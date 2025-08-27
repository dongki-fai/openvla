from transformers import AutoProcessor #, AutoModelForImageTextToText
import torch
from PIL import Image
import numpy as np

# ckpt = "allenai/MolmoAct-7B-D-0812"

def get_molmoact_vla(cfg):

    model = None
    # # load the model
    # model = AutoModelForImageTextToText.from_pretrained(
    #     cfg.pretrained_checkpoint,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,
    #     device_map="cuda",
    # )

    # model = model.to("cuda")

    return model

def get_molmoact_processor(cfg):
    # load the processor
    processor = AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        padding_side="left",
        use_fast=True
    )
    return processor

def get_molmoact_action(vla, processor, obs, task_label):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    instruction = task_label.lower()

    # strictly follow this reasoning prompt
    prompt = (
        f"The task is {instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {instruction}, "
        "let's think through it step by step. "
        "First, what is the depth map for the first image? "
        "Second, what is the trajectory of the end effector in the first image? "
        "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
        "along with other images from different camera views as additional information, "
        "what is the action that the robot should take?"
    )

    # apply chat template
    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [dict(type="text", text=prompt)]
            }
        ], 
        tokenize=False, 
        add_generation_prompt=True,
    )

    # process the image and text
    inputs = processor(
        images=image,
        text=text,
        padding=True,
        return_tensors="pt",
    )

    # move inputs to the correct device
    inputs = {k: v.to(vla.device) for k, v in inputs.items()}

    # generate output
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generated_ids = vla.generate(**inputs, max_new_tokens=256)

    # only get generated tokens; decode them to text
    generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
    generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # # print the generated text
    # print(f"generated text: {generated_text}")

    # >>>  The depth map of the first image is ... The trajectory of the end effector in the first image is ...
    #      Based on these information, along with other images from different camera views as additional information,
    #      the action that the robot should take is ...

    # # parse out all depth perception tokens
    # depth = model.parse_depth(generated_text)
    # print(f"generated depth perception tokens: {depth}")

    # # >>>  [ "<DEPTH_START><DEPTH_1><DEPTH_2>...<DEPTH_END>" ]

    # # parse out all visual reasoning traces
    # trace = model.parse_trace(generated_text)
    # print(f"generated visual reasoning trace: {trace}")

    # # >>>  [ [[242, 115], [140, 77], [94, 58], [140, 44], [153, 26]]] ]

    # parse out all actions, unnormalizing with key of "molmoact"
    action = vla.parse_action(generated_text, unnorm_key="molmoact")
    # print(f"generated action: {action[0]}")

    # >>>  [ [0.0732076061122558, 0.08228153779226191, -0.027760173818644346, 
    #         0.15932856272248652, -0.09686601126895233, 0.043916773912953344, 
    #         0.996078431372549] ]

    # Convert to numpy array
    action = np.array(action[0], dtype=np.float32)
    return action 
