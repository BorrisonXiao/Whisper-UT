#!/usr/bin/env python3

import argparse
from pathlib import Path
from local.data_prep.stm import parse_StmUtterance


def filter_dev(input_wavscp: Path, output_wavscp: Path, ref_wavscp: Path):
    """
    Keep only the utterances that are in the reference wav.scp file.
    """
    if str(ref_wavscp).endswith(".stm"):
        # Convert stm to wav.scp style
        with ref_wavscp.open() as f:
            ref_lines = f.readlines()
        ref_lines = [line.strip() for line in ref_lines]
        # Convert each line to wav.scp style
        ref_keys = [parse_StmUtterance(line).utterance_id(stereo=True) for line in ref_lines]

    with input_wavscp.open() as f:
        input_lines = f.readlines()
    with ref_wavscp.open() as f:
        ref_lines = f.readlines()

    ref_keys = [line.split()[0] for line in ref_lines] if not str(ref_wavscp).endswith(".stm") else ref_keys
    output_lines = [line for line in input_lines if line.split()[0] in ref_keys]

    with output_wavscp.open("w") as f:
        f.writelines(output_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, help="Input keyfile")
    parser.add_argument("--output", "-o", type=Path, help="Output keyfile")
    parser.add_argument("--ref", "-r", type=Path, help="Reference keyfile")
    args = parser.parse_args()

    filter_dev(input_wavscp=args.input, output_wavscp=args.output, ref_wavscp=args.ref)


if __name__ == "__main__":
    main()