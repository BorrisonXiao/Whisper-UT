#!/usr/bin/env python
import os
import logging
from typing import List, Tuple, Dict
from datasets import load_from_disk
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dset", type=Path, required=True,
                        help="Path to the src dataset location.")
    parser.add_argument("--tgt-dset", type=Path, required=True,
                        help="Path to the target dataset location.")
    parser.add_argument("--asr-hyp", type=Path, required=True,
                        help="Path to the asr input.")
    args = parser.parse_args()
    
    asr_hyps = {}
    with open(args.asr_hyp, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uttid, hyp = line.split(" ", maxsplit=1)
            asr_hyps[uttid] = hyp
    src_ds = load_from_disk(args.src_dset)
    # Duplicate the dataset and replace the text with the ASR hypothesis, keep everything else intact
    tgt_ds = src_ds.map(lambda x: {"transcript": asr_hyps[x["uttid"]].strip()})
    tgt_ds.save_to_disk(args.tgt_dset)

if __name__ == "__main__":
    main()
