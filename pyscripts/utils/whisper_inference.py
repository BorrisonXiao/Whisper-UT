#!/usr/bin/env python
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

import whisper
import argparse
from pathlib import Path
from tqdm import tqdm

LANGS = {
    "ara": "Arabic",
    "kor": "Korean",
    "cmn": "Chinese",
    "spa": "Spanish",
    "rus": "Russian",
    "eng": "English",
}


def inference(keyfile, src_lang, tgt_lang, output_dir, model_name, task="transcribe"):
    model = whisper.load_model(model_name)
    print(f"model.device: {model.device}")
    print(f"torch.cuda.is_available(): {whisper.torch.cuda.is_available()}")

    # Load the keyfile
    with open(keyfile, "r") as f:
        keys = f.readlines()
    keys = [key.strip() for key in keys]

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "text"
    with open(output, "w") as f:
        for key in tqdm(keys):
            uttid, wavpath = key.split(maxsplit=1)
            result = model.transcribe(
                audio=wavpath, language=src_lang, task=task, temperature=0.0)
            print(uttid, result["text"], file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyfile", type=Path,
                        default="dump/raw/org/dev/wav_raw.scp",
                        help="Path to the keyfile")
    parser.add_argument("--src-lang", type=str, default="ara",
                        help="Source language")
    parser.add_argument("--tgt-lang", type=str, default="eng",
                        help="Target language")
    parser.add_argument("--output_dir", type=Path,
                        default="exp/st_whisper/logdir/inference_asr/dev/output.1",
                        help="Path to the output directory")
    parser.add_argument("--task", type=str, default="transcribe",
                        choices=["transcribe", "translate"],
                        help="Task to perform")
    parser.add_argument("--model_name", type=str, default="base")

    args = parser.parse_args()
    inference(keyfile=args.keyfile, src_lang=LANGS[args.src_lang], tgt_lang=LANGS[args.tgt_lang],
              output_dir=args.output_dir, model_name=args.model_name, task=args.task)


if __name__ == "__main__":
    main()
