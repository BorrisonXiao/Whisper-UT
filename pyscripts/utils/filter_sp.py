#!/usr/bin/env python3

import argparse
from pathlib import Path


def filter_sp(input: Path, output: Path, speed: list, num_utts: int = 2000):
    """
    Filter out all the lines that contain _sp in the input keyfile.
    """
    with open(input, "r") as f:
        lines = f.readlines()
    with open(output, "w") as f:
        for line in lines[:num_utts]:
            line = line.strip()
            uttid = line.split(maxsplit=1)[0]
            # If the uttid does not start with "sp[speed]", then write it to the output file
            if not any([uttid.startswith(f"sp{_sp}") for _sp in speed]):
                f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, help="Input keyfile")
    parser.add_argument("--output", "-o", type=Path, help="Output keyfile")
    parser.add_argument("--speed", default=["0.9", "1.1"], help="Speed perturbation to filter out")
    parser.add_argument("--num-utts", default=2000, type=int, help="Number of utterances to keep")
    args = parser.parse_args()

    filter_sp(input=args.input, output=args.output, speed=args.speed, num_utts=args.num_utts)


if __name__ == "__main__":
    main()
