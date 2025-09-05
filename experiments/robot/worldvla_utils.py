""" Utilities for working with WorldVLA models."""

from PIL import Image
import torch
import numpy as np

from worldvla.model import ChameleonXLLMXForConditionalGeneration_ck as WorldVLA_Loader
from worldvla.data.pre_tokenize_action import ItemProcessor
from transformers import GenerationConfig

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_worldvla(cfg):

    model = WorldVLA_Loader.from_pretrained(
        cfg.pretrained_checkpoint,
        max_position_embeddings=4096,
        mask_image_logits=True,
        dropout=0.0,
        z_loss_weight=0.0,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    model.to(DEVICE).eval()

    ## TODO: Implement this cleanly
    if cfg.pruned_inference:
        FILTER_FOR = 'model.layers'  
        SKIP_LAYERS = ['vqmodel', 'lm_head', 'projector']
        # SKIP_LAYERS += ["." + str(i) + "." for i in range(24,32)]  # Skip first 16 layers of language model

        from experiments.robot.pruning_utils import attach_sparse_kernel, wrap_linears_with_svd
        # print(SKIP_LAYERS)
        # Attach sparse kernel
        model = attach_sparse_kernel(model, filter_for=FILTER_FOR, skip_layers=SKIP_LAYERS)

        # Load SVD factors and wrap linears
        if cfg.svd_factors_path is not None:
            model = wrap_linears_with_svd(model, cfg.svd_factors_path, filter_for=FILTER_FOR, skip_layers=SKIP_LAYERS, dtype=torch.bfloat16, device="cuda")


    return model

def get_worldvla_processor(cfg, target_size=256):
    processor = ItemProcessor(target_size=target_size, pretrained_checkpoint=cfg.pretrained_checkpoint)
    return processor

def unnorm_min_max_worldvla(action):
    min_values = np.array([-0.9375, -0.9375, -0.9375, -0.32571429, -0.375, -0.375, -1.0])
    max_values = np.array([0.9375, 0.9375, 0.9375, 0.375, 0.375, 0.375, 1.0])
    if action.shape[0] > 7:
        action = action[:7]
        
    unnorm_action = (action + 1) / 2 * (max_values - min_values + 1e-8) + min_values
    
    return unnorm_action

def get_worldvla_model_inputs(current_image, task_label, processor, history_image=[]):

    prompt = task_label.lower()

    conv = {
        "conversations":[
            {
                "from": "human",
                "value": "What action should the robot take to " + prompt + "?" + "<|image|>" * len(history_image[-1:]) + "<|image|>"
            },
        ],
        "image": history_image[-1:] + [current_image],
        "action": [],
    }

    tokens = processor.process_item(conv, training_mode=False)

    input_ids = torch.tensor(tokens, dtype=torch.int64, device=DEVICE).unsqueeze(0)

    return input_ids


def get_worldvla_action(vla, obs, task_label, processor, history_type, action_steps):

    current_image = obs['full_image']
    history_image = obs['history_image']  # already a list of PIL.Image
    current_image = Image.fromarray(current_image)

    if history_type != "2h_1a_img_only":
        raise NotImplementedError("History type not integrated. Please refer back to https://github.com/alibaba-damo-academy/WorldVLA/blob/3b71772b739dab954262a1e07193d34b6b53a3ba/worldvla/libero_util/Chameleon_utils.py#L386")
    
    input_ids = get_worldvla_model_inputs(current_image, task_label, processor, history_image)

    generation_config = GenerationConfig(max_new_tokens=action_steps*12,
                                        max_length=vla.config.max_position_embeddings,
                                        temperature=1,
                                        top_k=None,
                                        do_sample=False,
                                        eos_token_id=[8710],
                                    )

    dis_action = vla.generate_dis_ma(input_ids, generation_config)
    
    action_chunk = []
    for i in range(len(dis_action)):
        action_chunk.append(dis_action[i].cpu().float().detach().numpy())

    return action_chunk