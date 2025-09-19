#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from local.data_prep.stm import Stm

"""
Convert the STM files in the data directory to wav_raw.scp for inference.
"""


def generate_text(stm, output):
    with open(stm, "r") as f:
        _stm = Stm.parse(f)

    with open(output, "w") as f:
        for utt in _stm:
            print(utt.utterance_id(stereo=True), utt.transcript, sep=" ", file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, help="Input STM file")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output text file")
    args = parser.parse_args()

    generate_text(stm=args.input, output=args.output)


if __name__ == "__main__":
    main()
