#!/usr/bin/env python
import os
from datasets import load_dataset
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from data_prep.stm import StmUtterance

AUDIO_SAMPLING_RATE = 16000


def utt2uttid(utt, stereo=False):
    dur = len(utt["audio"]["array"]) / AUDIO_SAMPLING_RATE
    stm_utt = StmUtterance(filename=utt["file"], channel='A', speaker=utt["client_id"],
                           start_time=0.0, stop_time=dur, transcript=utt["sentence"].strip('\"'))
    return stm_utt.utterance_id(stereo=stereo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to the data directory.")
    parser.add_argument("--save-dir", type=Path, required=True,
                        help="Path to the output location.")
    parser.add_argument("--lang", type=str, default="fr",
                        help="Language code for the dataset.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers for data loading.")
    args = parser.parse_args()

    lang = args.lang

    # Load the dataset
    dataset = load_dataset("covost2", f"{lang}_en", data_dir=args.data_dir)
    print(f"Dataset loaded: {dataset}")

    files = defaultdict(dict)
    spkid = defaultdict(dict)
    duration = defaultdict(dict)
    transcripts = defaultdict(dict)
    translations = defaultdict(dict)
    # for split in ["test", "validation", "train"]:
    for split in ["validation", "train"]:
    # for split in ["test"]:
        valid_indices = set()
        for i in tqdm(range(len(dataset[split]))):
            try:
                utt = dataset[split][i]
                valid_indices.add(i)

                dur = len(utt["audio"]["array"]) / AUDIO_SAMPLING_RATE
                dur_str = f"{int(dur * 100):08d}"
                uttid = utt2uttid(utt)
                files[split][uttid] = utt["file"]
                spkid[split][uttid] = utt["client_id"]
                # Compute the duration based on the sample rate and audio array length
                duration[split][uttid] = dur
                transcripts[split][uttid] = utt["sentence"].strip('\"')
                translations[split][uttid] = utt["translation"]
            except Exception as e:
                print(f"Error processing {split} split at index {i}: {e}")
        # Select only the valid indices
        dataset[split] = dataset[split].select(list(valid_indices))
        # Print the number of valid samples
        print(
            f"Number of valid samples in {split} split: {len(valid_indices)}")
        dataset[split] = dataset[split].map(lambda x: {"transcript": transcripts[split][utt2uttid(x)],
                                                       "src_lang": f"{lang}",
                                                       "tgt_lang": "en",
                                                       "uttid": utt2uttid(x)},
                                            remove_columns=["client_id", "file", "sentence"], num_proc=min(args.num_workers, os.cpu_count()))

        args.save_dir.mkdir(parents=True, exist_ok=True)
        dataset[split].save_to_disk(args.save_dir / f"{lang}.{split}")
        print(f"Datasets saved to {args.save_dir}/{lang}.{split}")

        # Store a wav.scp like file, kaldi-style text file, and a stm file for the dev and test split
        with open(args.save_dir / f"{split}.wav.scp", "w") as f:
            for uttid, file in files[split].items():
                print(f"{uttid} {file}", file=f)
        with open(args.save_dir / f"{split}.text", "w") as f:
            for uttid, transcript in transcripts[split].items():
                print(f"{uttid} {transcript}", file=f)
        with open(args.save_dir / f"{split}.src.stm", "w") as f:
            for uttid, transcript in transcripts[split].items():
                dur = duration[split][uttid]
                print(
                    f"{files[split][uttid]} A {spkid[split][uttid]} 0.00 {int(dur * 100) / 100:.2f} <O> {transcript}", file=f)
        with open(args.save_dir / f"{split}.tgt.stm", "w") as f:
            for uttid, translation in translations[split].items():
                dur = duration[split][uttid]
                print(
                    f"{files[split][uttid]} A {spkid[split][uttid]} 0.00 {int(dur * 100) / 100:.2f} <O> {translation}", file=f)


if __name__ == "__main__":
    main()
