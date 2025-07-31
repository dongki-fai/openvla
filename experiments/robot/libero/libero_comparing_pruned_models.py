import torch
import pandas as pd
import pdb
import json
import csv
from torch import nn
from transformers import AutoModelForCausalLM

from transformers import AutoModelForCausalLM
from experiments.robot.robot_utils import get_model

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Setup dummy config with checkpoint
class DummyConfig():
    def __init__(self, pretrained_checkpoint):
        self.model_family = "openvla"
        self.pretrained_checkpoint = pretrained_checkpoint
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.pruned_inference = False
        self.load_to_cpu = True


def compare_models(model1, model2, device='cuda'):
    # Move models to appropriate devices
    model1.eval()
    model2 = model2.state_dict()

    # Collect data for comparison
    layer_names = []
    pct_diffs = []
    num_diffs = []

    # Perform patch under no_grad to avoid in-place grad errors
    with torch.no_grad():
        for name, module in model1.named_modules():
            if isinstance(module, nn.Linear):
                
                model1_W = module.weight
                model2_W = model2[name + ".weight"]

                # Calculate percentage of weights that are different
                diff_mask = model1_W != model2_W

                num_diff = int(diff_mask.sum().item())
                total    = model1_W.numel()
                pct_diff = num_diff / total * 100

                print(f"Layer: {name} {pct_diff:.2f}% differ ({num_diff}/{total})")

                # collect data
                layer_names.append(name)
                pct_diffs.append(pct_diff)
                num_diffs.append(num_diff)

    return layer_names, pct_diffs, num_diffs



def plot_layer_diffs(layer_names, pct_diffs, num_diffs):
    """
    Creates two line plots:
      1) Percentage of differing weights vs. layer index
      2) Count of differing weights vs. layer index

    Line segments and markers are colored by layer type,
    and we skip some x‑tick labels for readability.
    """
    # Color definitions
    lang_color   = '#F15A29'
    proj_color   = '#2980B9'
    vision_color = '#9B59B6'
    default_color = '#999999'
    hatch = '//'

    # Map each layer to a color
    colors = []
    for name in layer_names:
        if 'language_model' in name:
            colors.append(lang_color)
        elif 'projector' in name:
            colors.append(proj_color)
        elif 'vision_backbone' in name:
            colors.append(vision_color)
        else:
            colors.append(default_color)

    x = np.arange(len(layer_names))

    # Determine tick spacing (show ~10 ticks max)
    tick_step = max(1, len(x) // 10)
    tick_positions = x[::tick_step]

    # Legend handles
    legend_handles = [
        mpatches.Patch(facecolor=lang_color, edgecolor='black', hatch=hatch, label='Language model'),
        mpatches.Patch(facecolor=proj_color, edgecolor='black', hatch=hatch, label='Projector'),
        mpatches.Patch(facecolor=vision_color, edgecolor='black', hatch=hatch, label='Vision backbone'),
    ]

    # --- Plot 1: Percentage different ---
    fig, ax = plt.subplots(figsize=(12, 6))
    # Draw colored line segments
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], [pct_diffs[i], pct_diffs[i+1]], color=colors[i], linewidth=2)
    ax.scatter(x, pct_diffs, color=colors, s=30, edgecolors='black', linewidth=0.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(i) for i in tick_positions], rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Percentage Different (%)', fontsize=12)
    ax.set_title('Layer Weight Difference (%) by Index', fontsize=14, weight='bold')
    ax.set_ylim(0, max(pct_diffs) * 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=legend_handles, fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig('layer_diff_percentage_line.png', dpi=300)
    plt.show()

    # --- Plot 2: Count of weights different ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], [num_diffs[i], num_diffs[i+1]], color=colors[i], linewidth=2)
    ax.scatter(x, num_diffs, color=colors, s=30, edgecolors='black', linewidth=0.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(i) for i in tick_positions], rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Number of Weights Different', fontsize=12)
    ax.set_title('Layer Weight Difference Counts by Index', fontsize=14, weight='bold')
    ax.set_ylim(0, max(num_diffs) * 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=legend_handles, fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig('layer_diff_count_line.png', dpi=300)
    plt.show()

# Load the pruned OpenVLA model
print("[*] Loading pruned OpenVLA model 1...")
path_to_model1 = "/workspace/models/openvla-7b-pruned-2_4-Wanda-pruned-full_model-unprioritzed-calibset-15000"
cfg = DummyConfig(path_to_model1)
model1 = get_model(cfg)
model1 = model1.float()

print("[*] Loading pruned OpenVLA model 2...")
path_to_model2 = "/workspace/models/openvla-7b-pruned-2_4-Wanda-pruned-full_model-Closed_Gripper_Data_2_5_Window"
# path_to_dense_model = "/workspace/openvla/pruned_model_nudged"
cfg = DummyConfig(path_to_model2)
model2 = get_model(cfg)
model2 = model2.float()

names, pct, num = compare_models(model1, model2, device='cuda')
plot_layer_diffs(names, pct, num)

pdb.set_trace()
