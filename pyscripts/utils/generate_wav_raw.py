#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from local.data_prep.stm import Stm

"""
Convert the STM files in the data directory to wav_raw.scp for inference.
"""


def generate_wav_raw(stm, output_dir):
    with open(stm, "r") as f:
        _stm = Stm.parse(f)

    with open(output_dir / "wav_raw.scp", "w") as f:
        for utt in _stm:
            print(utt.utterance_id(stereo=True), utt.filename, sep=" ", file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, help="Input STM file")
    parser.add_argument("-o", "--output_dir", type=Path,
                        help="Output directory")
    args = parser.parse_args()

    generate_wav_raw(stm=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
