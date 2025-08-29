""" Utilities for working with CogACT VLA models."""

from PIL import Image
import torch
import numpy as np
from vla import load_vla as load_cogact

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_cogact_vla(cfg):

    if 'large' in cfg.pretrained_checkpoint.lower():
        action_model_type = 'DiT-L'
    elif 'base' in cfg.pretrained_checkpoint.lower():
        action_model_type = 'DiT-B'
    elif 'small' in cfg.pretrained_checkpoint.lower():
        action_model_type = 'DiT-S'
    else:
        raise ValueError(f"Unknown model type in checkpoint path: {cfg.pretrained_checkpoint}")

    model = load_cogact(
        cfg.pretrained_checkpoint,
        load_for_training=False,
        action_model_type=action_model_type,  # choose from ['DiT-S', 'DiT-B', 'DiT-L'] to match the model weight
        future_action_window_size=15,
    )
    # Load vlm to bf16 to save memory
    model.vlm = model.vlm.to(torch.bfloat16)
    model.to(DEVICE).eval()
    return model

def get_cogact_model_inputs(image, task_label):
    raise NotImplementedError("CogACT model input function not implemented.")

def get_cogact_action(vla, obs, task_label, unnorm_key):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    prompt = task_label.lower()

    # Predict Action (7-DoF; un-normalize for RT-1 google robot data, i.e., fractal20220817_data)
    actions, _ = vla.predict_action(
                image,
                prompt,
                unnorm_key=unnorm_key,              # input your unnorm_key of the dataset
                cfg_scale = 1.5,                    # cfg from 1.5 to 7 also performs well
                use_ddim = True,                    # use DDIM sampling
                num_ddim_steps = 10,                # number of steps for DDIM sampling
            )

    # results in 7-DoF actions of 16 steps with shape [16, 7]

    return actions[0]
