#!/usr/bin/env python
import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset1", type=Path, required=True,
                        help="Path to the dataset1 location.")
    parser.add_argument("--dset2", type=Path, required=True,
                        help="Path to the dataset2 location.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to the output location.")
    args = parser.parse_args()

    print(f"dataset1: {args.dset1}")
    print(f"dataset2: {args.dset2}")
    print(f"output: {args.output}")
    
    ds1 = load_from_disk(str(args.dset1))
    ds2 = load_from_disk(str(args.dset2))
    ds = concatenate_datasets([ds1, ds2])
    ds.save_to_disk(str(args.output))


if __name__ == "__main__":
    main()
