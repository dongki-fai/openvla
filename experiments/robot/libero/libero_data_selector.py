import os
import math
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

# ---- Config ----
IN_ROOT  = "/workspace/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
OUT_ROOT = IN_ROOT.replace("modified_libero_rlds", "closing_and_opening_gripper_libero_rlds")
GRIPPER_CLOSED_VAL = 1.0          # action[-1] == 1.0 => closed
PAD_FRAC            = 0.025       # 5% padding on both sides
ACTION_DIM, STATE_DIM, JOINT_DIM = 7, 8, 7
SKIP_EMPTY_EPISODES  = True
SLICE_MODE = 'cluster'  # 'full', 'window', or 'cluster'
CLUSTER_K  = 7          # how many k‑means clusters
CLUSTER_M  = 2000       # max samples per cluster
os.makedirs(OUT_ROOT, exist_ok=True)
CLUSTER_MAP = {}

# Describe per-step fields (so we can slice quickly)
SPEC = {
    "steps/reward":                     ("float", 1),
    "steps/discount":                   ("float", 1),
    "steps/action":                     ("float", ACTION_DIM),
    "steps/observation/state":          ("float", STATE_DIM),
    "steps/observation/joint_state":    ("float", JOINT_DIM),
    "steps/is_first":                   ("int",   1),
    "steps/is_last":                    ("int",   1),
    "steps/is_terminal":                ("int",   1),
    "steps/observation/image":          ("bytes", 1),
    "steps/observation/wrist_image":    ("bytes", 1),
    "steps/language_instruction":       ("bytes", 1),  # usually single entry
}

def slice_example(orig_ex, start, end, cluster_k=None):
    """Return new SequenceExample with steps in [start, end] (inclusive)."""
    ctx_in  = orig_ex.context.feature
    new_ex  = tf.train.SequenceExample()
    ctx_out = new_ex.context.feature

    # copy episode-level (non-steps) features
    for k in ctx_in.keys():
        if not k.startswith("steps/"):
            ctx_out[k].CopyFrom(ctx_in[k])

    # Add cluster_k as a new context feature if given
    if cluster_k is not None:
        # int64_list for a single integer
        ctx_out["cluster_id"].int64_list.value.append(int(cluster_k))

    idx_range = range(start, end + 1)

    for key, (kind, stride) in SPEC.items():
        if key not in ctx_in:  # skip absent keys
            continue
        src = ctx_in[key]
        if kind == "float":
            vals = src.float_list.value
            if stride == 1:
                out = [vals[i] for i in idx_range]
            else:
                out = []
                for i in idx_range:
                    out.extend(vals[i*stride:(i+1)*stride])
            ctx_out[key].float_list.value[:] = out

        elif kind == "int":
            vals = src.int64_list.value
            ctx_out[key].int64_list.value[:] = [vals[i] for i in idx_range]

        elif kind == "bytes":
            vals = src.bytes_list.value
            if key == "steps/language_instruction" and len(vals) == 1:
                ctx_out[key].bytes_list.value[:] = vals  # keep as-is
            else:
                ctx_out[key].bytes_list.value[:] = [vals[i] for i in idx_range]

    return new_ex

def gather_cluster_states():
    global CLUSTER_MAP
    print("[Cluster] Gathering all states for k-means…")
    locs = []    # list of (fname, ep_idx, t)

    tfrecord_paths = [
        os.path.join(IN_ROOT, f)
        for f in sorted(os.listdir(IN_ROOT))
        if ".tfrecord" in f
    ]
    # === Load all episodes into a list first ===
    raw_dataset = list(tf.data.TFRecordDataset(tfrecord_paths))
    total_episodes = len(raw_dataset)
    print(f"Total episodes in file: {total_episodes}")

    states_list = []
    global_ep_idx_list = []
    state_idx_in_ep_list = []
    global_ep_idx = 0
    for ep in range(total_episodes):
        raw_record = raw_dataset[ep]
        ex = tf.train.SequenceExample()
        ex.ParseFromString(raw_record.numpy())
        ctx = ex.context.feature

        state_flat = ctx["steps/observation/state"].float_list.value
        state = np.asarray(state_flat, dtype=np.float32).reshape(-1, 8)
        states_list.append(state)
        global_ep_idx_list.extend([global_ep_idx] * state.shape[0])
        state_idx_in_ep_list.extend(range(state.shape[0]))
        global_ep_idx += 1

    # Stack into one array of shape (total_steps, 8)
    states_all = np.vstack(states_list)
    print(f"[Cluster] Collected {states_all.shape[0]} total state vectors.")

    print(f"[Cluster] Running k-means with K={CLUSTER_K}…")
    km      = KMeans(n_clusters=CLUSTER_K, init="k-means++", random_state=0)
    labels  = km.fit_predict(states_all)
    print("  cluster sizes:", np.bincount(labels))

    picks = []
    for c in range(CLUSTER_K):
        idxs = np.where(labels == c)[0]
        if len(idxs) > CLUSTER_M:
            picks.append(np.random.choice(idxs, CLUSTER_M, replace=False))
        else:
            picks.append(idxs)
    
    # # build cluster->episode->timesteps map
    for cluster_idx, cluster in enumerate(picks):
        for flat_idx in cluster:
            ep_idx = global_ep_idx_list[flat_idx]
            state_idx = state_idx_in_ep_list[flat_idx]
            CLUSTER_MAP.setdefault(ep_idx, []).append((cluster_idx, state_idx))


    print(f"[Cluster] Collected {len(picks)} clusters.")


if SLICE_MODE == 'cluster':
    gather_cluster_states()

# ---- Main loop over shards ----
global_ep_idx = -1
for fname in sorted(f for f in os.listdir(IN_ROOT) if ".tfrecord" in f):
    in_path  = os.path.join(IN_ROOT, fname)
    out_path = os.path.join(OUT_ROOT, fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[Shard] {fname}")
    ds     = tf.data.TFRecordDataset(in_path)
    writer = tf.io.TFRecordWriter(out_path)

    n_in = n_out = 0
    for ep_idx, raw in enumerate(ds):
        global_ep_idx += 1
        n_in += 1
        ex  = tf.train.SequenceExample(); ex.ParseFromString(raw.numpy())
        ctx = ex.context.feature

        if SLICE_MODE == 'cluster':
            if global_ep_idx not in CLUSTER_MAP:
                continue
            timesteps = CLUSTER_MAP[global_ep_idx]
            # Slice the example to only include the specified timestep
            for cluster_k, t in timesteps:
                new_ex = slice_example(ex, t, t, cluster_k=cluster_k)
                writer.write(new_ex.SerializeToString())
                n_out += 1
            continue

        actions = np.asarray(ctx["steps/action"].float_list.value, np.float32).reshape(-1, ACTION_DIM)
        T = actions.shape[0]
        mask = np.isclose(actions[:, -1], GRIPPER_CLOSED_VAL)

        if not mask.any():
             continue
        first_close = int(np.argmax(mask))            # first closed step
        last_close = int(T - 1 - np.argmax(mask[::-1])) # last closed step
        pad   = max(1, math.ceil(PAD_FRAC * T))

        if SLICE_MODE == 'full':
            # from first→last close, plus pad on each end
            start = max(0,          first_close - pad)
            end   = min(T - 1, last_close + pad)

            new_ex = slice_example(ex, start, end)
            writer.write(new_ex.SerializeToString())
            n_out += 1

        elif SLICE_MODE == 'window':
            # only around the first close
            first_close_start = max(0, first_close - pad)
            first_close_end   = min(T - 1, first_close + pad)

            new_ex = slice_example(ex, first_close_start, first_close_end)
            writer.write(new_ex.SerializeToString())
            n_out += 1

            # # only around the last close
            # last_close_start = max(0, last_close - pad)
            # last_close_end   = min(T - 1, last_close + pad)

            # new_ex = slice_example(ex, last_close_start, last_close_end)
            # writer.write(new_ex.SerializeToString())
            # n_out += 1
        else:
            raise ValueError(f"Unknown SLICE_MODE={SLICE_MODE!r}")


    writer.close()
    print(f"  Episodes in: {n_in}, written: {n_out}")

print("\n[Done] New TFRecords saved under:", OUT_ROOT)
