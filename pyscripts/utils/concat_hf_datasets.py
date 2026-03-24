#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate one or more Hugging Face datasets, optionally save the "
            "result, update a summary json, and write a wav.scp-style keyfile."
        )
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Input Hugging Face dataset directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output Hugging Face dataset directory.",
    )
    parser.add_argument(
        "--wav-scp",
        type=Path,
        default=None,
        help="Optional wav.scp-style keyfile to write. Only the uttid column is used downstream.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary json to update with the dataset size.",
    )
    parser.add_argument(
        "--summary-key",
        type=str,
        default=None,
        help="Key to update inside --summary-json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output dataset and keyfile if they already exist.",
    )
    return parser.parse_args()


def load_dataset_bundle(paths):
    datasets = [load_from_disk(str(path)) for path in paths]
    reference_features = datasets[0].features
    for path, dataset in zip(paths[1:], datasets[1:]):
        if dataset.features != reference_features:
            raise ValueError(
                f"Feature mismatch for {path}. "
                "All datasets must share the same schema before concatenation."
            )
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def maybe_save_dataset(dataset, output_path: Path, force: bool):
    if output_path.exists():
        if not force:
            return load_from_disk(str(output_path))
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    return load_from_disk(str(output_path))


def write_keyfile(dataset, path: Path, force: bool):
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    uttids = dataset["uttid"]
    with open(path, "w", encoding="utf-8") as fout:
        for uttid in uttids:
            print(f"{uttid} {uttid}", file=fout)


def update_summary(dataset, summary_path: Path, key: str):
    summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as fin:
            summary = json.load(fin)
    summary[key] = len(dataset)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    if args.output is None and len(args.inputs) != 1:
        raise ValueError("When --output is omitted, exactly one input dataset must be provided.")

    dataset = load_dataset_bundle(args.inputs)

    if args.output is not None:
        dataset = maybe_save_dataset(dataset, args.output, args.force)

    if args.wav_scp is not None:
        write_keyfile(dataset, args.wav_scp, args.force)

    if args.summary_json is not None:
        if not args.summary_key:
            raise ValueError("--summary-key is required when --summary-json is provided.")
        update_summary(dataset, args.summary_json, args.summary_key)


if __name__ == "__main__":
    main()
