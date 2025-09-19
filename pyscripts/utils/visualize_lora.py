#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2023 Cihan Xiao (Johns Hopkins University)

"""
Visualize the LoRA weights in a 2D heatmap.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration
from peft.tuners.lora.layer import LoraLayer
from tqdm import tqdm


def visualize_matrix(matrix: np.ndarray, name: str, model_dir: Path, ymax: float = None, ymin: float = None):
    out_dir = model_dir / "lora_weights"
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(matrix, cmap='RdBu')
    if ymax is not None and ymin is not None:
        # Set the limit of the heatmap
        c.set_clim(vmin=ymin, vmax=ymax)
    ax.set_title(name)
    fig.colorbar(c, ax=ax)
    plt.savefig(out_dir / f"{name}.png")
    plt.close()


def visualize(peft_model: Path, scaler: float = 5e1):
    # Step 1: Load the LoRA model
    print(f"Loading model from {peft_model}")
    peft_config = PeftConfig.from_pretrained(peft_model)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model)

    # Step 2: Get each LoraLayer
    # Get all LoraLayer instances in the model
    lora_layers = [(".".join(name.split(".")[3:]), layer) for name, layer in model.named_modules(
    ) if isinstance(layer, LoraLayer)]
    encoder_layers = [(name, layer)
                      for name, layer in lora_layers if "encoder." in name]
    decoder_layers = [(name, layer)
                      for name, layer in lora_layers if "decoder." in name]
    encoder_delta_weights = {name: layer.get_delta_weight(
        layer.active_adapter).numpy() for name, layer in encoder_layers}
    decoder_delta_weights = {name: layer.get_delta_weight(
        layer.active_adapter).numpy() for name, layer in decoder_layers}

    # Step 3: Visualize the weights
    # Normalize the y-axis to be the same for all layers (both encoder and decoder)
    ymax = max([np.max(weights) for weights in encoder_delta_weights.values()])
    ymin = min([np.min(weights) for weights in encoder_delta_weights.values()])
    _ymax = np.exp(ymax * scaler)
    _ymin = np.exp(ymin * scaler)
    for name, weights in tqdm(encoder_delta_weights.items()):
        # The weights are exponentiated to highlight the differences
        exp_weights = np.exp(weights * scaler)
        visualize_matrix(exp_weights, name, peft_model, _ymax, _ymin)

    ymax = max([np.max(weights) for weights in decoder_delta_weights.values()])
    ymin = min([np.min(weights) for weights in decoder_delta_weights.values()])
    _ymax = np.exp(ymax * scaler)
    _ymin = np.exp(ymin * scaler)

    for name, weights in tqdm(decoder_delta_weights.items()):
        # The weights are exponentiated to highlight the differences
        exp_weights = np.exp(weights * scaler)
        visualize_matrix(exp_weights, name, peft_model, _ymax, _ymin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path,
                        default="/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_large-v2_merged/tus/train-cts_sp/asr/lora",
                        help="Path to the LoRA model.")

    args = parser.parse_args()
    visualize(peft_model=args.model)


if __name__ == '__main__':
    main()
