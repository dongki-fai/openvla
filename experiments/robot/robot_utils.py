"""Utils for evaluating robot policies in various environments."""

import os
import random
import time

import numpy as np
import torch

def import_neccessary_libraries(model_family: str):
    """Imports neccessary libraries for a given model family."""
    if model_family == "openvla":
        from experiments.robot.openvla_utils import (
            get_vla,
            get_openvla_processor,
            get_vla_action,
        )
        globals().update(locals())
    elif model_family == "cogact":
        from experiments.robot.cogact_utils import (
            get_cogact_vla,
            get_cogact_action
        )
        globals().update(locals())
    elif model_family == "worldvla":
        from experiments.robot.worldvla_utils import (
            get_worldvla,
            get_worldvla_processor,
            get_worldvla_action, 
            unnorm_min_max_worldvla,
        )
        globals().update(locals())
    elif model_family == "molmoact":
        from experiments.robot.molmoact_utils import (
            get_molmoact_vla,
            get_molmoact_processor,
            get_molmoact_action
        )
        globals().update(locals())
    else:
        raise ValueError("Unexpected `model_family` found in config.")

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg, wrap_diffusion_policy_for_droid=False):
    """Load model for evaluation."""
    if cfg.model_family == "openvla":
        model = get_vla(cfg)
    elif cfg.model_family == "cogact":
        model = get_cogact_vla(cfg)
    elif cfg.model_family == "worldvla":
        model = get_worldvla(cfg)
    elif cfg.model_family == "molmoact":
        model = get_molmoact_vla(cfg)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model

def get_processor(cfg):

    processor = None
    if cfg.model_family == "openvla":
        processor = get_openvla_processor(cfg)
    elif cfg.model_family == "cogact":
        processor = None
    elif cfg.model_family == "molmoact":
        processor = get_molmoact_processor(cfg)
    elif cfg.model_family == "worldvla":
        processor = get_worldvla_processor(cfg)

    return processor

def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family in ["openvla", "cogact", "molmoact"]:
        resize_size = 224
    elif cfg.model_family == "worldvla":
        resize_size = 256
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, processor=None):
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        assert action.shape == (ACTION_DIM,)
    elif cfg.model_family == "cogact":
        action = get_cogact_action(model, obs, task_label, cfg.unnorm_key)
    elif cfg.model_family == "worldvla":
        action = get_worldvla_action(model, obs, task_label, processor, cfg.history_type, cfg.action_steps)
    elif cfg.model_family == "molmoact":
        action = get_molmoact_action(model, processor, obs, task_label)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action

def post_process_action(cfg, action):

    if cfg.model_family in ["openvla", "cogact", "molmoact"]:
        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
        action = normalize_gripper_action(action, binarize=True)

        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        action = invert_gripper_action(action)
    elif cfg.model_family == "worldvla":
        # Un-normalize action
        action = unnorm_min_max_worldvla(action)
    else:
        raise ValueError("Unexpected `model_family` found in config.")

    return action


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
