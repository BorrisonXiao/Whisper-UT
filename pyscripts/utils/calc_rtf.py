#!/usr/bin/env python3

import argparse
from pathlib import Path
import os
import re

LOGDIR = "/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_large-v2_merged/logdir"
LANGS = ["ara", "cmn", "kor", "rus", "spa"]
TRAIN_SETS = ["train-cts_sp", "train-all_sp"]
MODES = ["inference_asr", "inference_st"]
DSET_DUR = {
    "ara": {"dev1": 3, "dev2": 3.3, "iwslt22_test": 3.6, "fleurs": 8.1},
    "cmn": {"dev1": 4, "dev2": 4.5, "bbn_cts_bolt_test": 8.5, "fleurs": 13.9},
    "kor": {"dev1": 2.8, "dev2": 2.4, "uhura_test": 3.0, "fleurs": 9.8},
    "rus": {"dev1": 6.0, "dev2": 6.5, "uhura_test": 6.1, "fleurs": 11.4},
    "spa": {"dev1": 4.6, "dev2": 4.7, "fisher_test": 4.5, "callhome_test": 1.8, "fleurs": 13.0},
}

def calc_rtf(logdir: Path, langs: list = LANGS, train_sets: list = TRAIN_SETS, modes: list = MODES):
    """
    Compute the real-time-factor (RTF) of a model.
    """
    asr_avg_time = []
    st_avg_time = []
    avg_time_none = []
    avg_time_lora = []
    for mode in modes:
        for lang in langs:
            for train_set in train_sets:
                # If the directory doesn't exist, continue
                train_set_dir = logdir / mode / lang / train_set
                if not train_set_dir.exists():
                    continue
                for dset in os.listdir(train_set_dir):
                    if dset not in DSET_DUR[lang]:
                        continue
                    dsetdir = train_set_dir / dset
                    for peft in os.listdir(dsetdir):
                        _err = False
                        peftdir = dsetdir / peft
                        if Path(peftdir / "merged").is_dir():
                            peftdir = peftdir / "merged"
                        # Glob all files in the format decode.*.log
                        decode_logs = list(peftdir.glob("decode.*.log"))
                        if len(decode_logs) == 0:
                            continue
                        # For each log, get the time and sum them all up
                        total_time = 0
                        for decode_log in decode_logs:
                            with decode_log.open() as f:
                                lines = f.readlines()
                            # The second to the last line is the total time
                            # It's also in the format "# Accounting: time=742 threads=1"
                            try:
                                total_time += int(re.search(r"time=(\d+)", lines[-2]).group(1))
                            except AttributeError:
                                print(f"Error in {decode_log}, skipping...")
                                _err = True
                                break
                        if not _err:
                            print(f"{mode}/{lang}/{train_set}/{dset}/{peft}: {total_time / DSET_DUR[lang][dset] / 3600:.2f}")
                            if mode == "inference_asr":
                                asr_avg_time.append(total_time / DSET_DUR[lang][dset] / 3600)
                            elif mode == "inference_st":
                                st_avg_time.append(total_time / DSET_DUR[lang][dset] / 3600)
                            if "none" in peft:
                                avg_time_none.append(total_time / DSET_DUR[lang][dset] / 3600)
                            elif "lora" in peft:
                                avg_time_lora.append(total_time / DSET_DUR[lang][dset] / 3600)
    if len(asr_avg_time) > 0:
        print(f"Average time for asr: {sum(asr_avg_time) / len(asr_avg_time):.2f}")
    if len(st_avg_time) > 0:
        print(f"Average time for st: {sum(st_avg_time) / len(st_avg_time):.2f}")
    if len(avg_time_none) > 0:
        print(f"Average time for none: {sum(avg_time_none) / len(avg_time_none):.2f}")
    if len(avg_time_lora) > 0:
        print(f"Average time for lora: {sum(avg_time_lora) / len(avg_time_lora):.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", "-l", type=Path, help="Log directory", default=LOGDIR)
    parser.add_argument("--langs", "-L", type=str, nargs="+", help="Languages", default=LANGS)
    parser.add_argument("--train-sets", "-t", type=str, nargs="+", help="Train sets", default=TRAIN_SETS)
    parser.add_argument("--modes", "-m", type=str, nargs="+", help="Modes", default=MODES)
    args = parser.parse_args()
    print(args)

    calc_rtf(logdir=args.logdir, langs=args.langs, train_sets=args.train_sets, modes=args.modes)

if __name__ == "__main__":
    main()


