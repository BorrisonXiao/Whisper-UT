#!/usr/bin/env python3
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_from_disk
from peft import PeftModel, PeftConfig

def merge_model(lora_path: Path, outdir: Path):
    # Step 1: Load the lora model
    print(f"Loading model from {lora_path}")
    peft_config = PeftConfig.from_pretrained(lora_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path)
    processor = WhisperProcessor.from_pretrained(lora_path)
    model = PeftModel.from_pretrained(model, lora_path)
    
    # Step 2: Merge the lora weights and save the new model
    print(f"Saving model to {outdir}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(outdir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", type=Path, default=Path("/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_large-v2_merged/ara/train-cts_sp/mtl/lora"))
    parser.add_argument("--outdir", type=Path, default=Path("/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_large-v2_merged/ara/train-cts_sp/mtl/lora/merged_model"))
    args = parser.parse_args()

    merge_model(args.lora_path, args.outdir)

if __name__ == "__main__":
    main()