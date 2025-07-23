import os
import math
import numpy as np
import tensorflow as tf

# ---- Config ----
IN_ROOT  = "/workspace/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
OUT_ROOT = IN_ROOT.replace("modified_libero_rlds", "closed_gripper_libero_rlds")
GRIPPER_CLOSED_VAL = 1.0          # action[-1] == 1.0 => closed
PAD_FRAC            = 0.025       # 2.5% padding on both sides
ACTION_DIM, STATE_DIM, JOINT_DIM = 7, 8, 7
SKIP_EMPTY_EPISODES  = True

os.makedirs(OUT_ROOT, exist_ok=True)

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

def slice_example(orig_ex, start, end):
    """Return new SequenceExample with steps in [start, end] (inclusive)."""
    ctx_in  = orig_ex.context.feature
    new_ex  = tf.train.SequenceExample()
    ctx_out = new_ex.context.feature

    # copy episode-level (non-steps) features
    for k in ctx_in.keys():
        if not k.startswith("steps/"):
            ctx_out[k].CopyFrom(ctx_in[k])

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

# ---- Main loop over shards ----
for fname in sorted(f for f in os.listdir(IN_ROOT) if ".tfrecord" in f):
    in_path  = os.path.join(IN_ROOT, fname)
    out_path = os.path.join(OUT_ROOT, fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[Shard] {fname}")
    ds     = tf.data.TFRecordDataset(in_path)
    writer = tf.io.TFRecordWriter(out_path)

    n_in = n_out = 0
    for raw in ds:
        n_in += 1
        ex  = tf.train.SequenceExample(); ex.ParseFromString(raw.numpy())
        ctx = ex.context.feature

        actions = np.asarray(ctx["steps/action"].float_list.value, np.float32).reshape(-1, ACTION_DIM)
        T = actions.shape[0]
        mask = np.isclose(actions[:, -1], GRIPPER_CLOSED_VAL)

        if not mask.any():
             continue

        first = np.argmax(mask)                       # first True
        last  = T - 1 - np.argmax(mask[::-1])         # last True
        pad   = max(1, math.ceil(PAD_FRAC * T))
        start = max(0, first - pad)
        end   = min(T - 1, last + pad)

        new_ex = slice_example(ex, start, end)
        writer.write(new_ex.SerializeToString())
        n_out += 1

    writer.close()
    print(f"  Episodes in: {n_in}, written: {n_out}")

print("\n[Done] New TFRecords saved under:", OUT_ROOT)
