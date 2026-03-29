#!/usr/bin/env python3

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local.data_prep.stm import parse_StmUtterance


MS_SCALE = 1000
MERGED_DATASET_NAMES = {
    "3way": "train_3way_merged",
    "asr_only": "train_asr_only_merged",
    "st_only": "train_st_only_merged",
}


@dataclass(frozen=True)
class SegmentExample:
    audio_path: str
    channel: str
    speaker: str
    start_ms: int
    stop_ms: int
    transcript: str
    translation: str
    source_dataset: str
    split: str

    @property
    def start_time(self) -> float:
        return self.start_ms / MS_SCALE

    @property
    def stop_time(self) -> float:
        return self.stop_ms / MS_SCALE

    @property
    def duration(self) -> float:
        return max(0.0, self.stop_time - self.start_time)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create deterministic STM-first manifests from paired SR/ST STM files.",
    )
    parser.add_argument("--train-sr-stms", nargs="*", default=[])
    parser.add_argument("--train-st-stms", nargs="*", default=[])
    parser.add_argument("--dev-sr-stms", nargs="*", default=[])
    parser.add_argument("--dev-st-stms", nargs="*", default=[])
    parser.add_argument("--test-sr-stms", nargs="*", default=[])
    parser.add_argument("--test-st-stms", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--src-lang", type=str, default="ara")
    parser.add_argument("--tgt-lang", type=str, default="eng")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mean", type=float, default=15.0)
    parser.add_argument("--std", type=float, default=7.0)
    parser.add_argument("--t-min", type=float, default=5.0)
    parser.add_argument("--t-max", type=float, default=25.0)
    parser.add_argument("--ratios", nargs=3, type=float, default=[1.0, 0.0, 0.0])
    parser.add_argument("--allow-gap", type=str, default="false")
    return parser.parse_args()


def dataset_name_from_path(path: Path) -> str:
    stem_parts = path.stem.split(".")
    if len(stem_parts) < 3:
        raise ValueError(f"Unexpected STM filename format: {path}")
    return "_".join(stem_parts[2:])


def mark_as_original(dataset_name: str) -> str:
    return f"{dataset_name}_org"


def utterance_key(utt) -> Tuple[str, str, int, int]:
    return (
        str(Path(utt.filename)),
        utt.channel,
        int(round(utt.start_time * MS_SCALE)),
        int(round(utt.stop_time * MS_SCALE)),
    )


def parse_stm_file(path: Path):
    with open(path, "r", encoding="utf-8") as fin:
        return [parse_StmUtterance(line) for line in fin if line.strip()]


def align_pair(sr_path: Path, st_path: Path, split: str) -> List[SegmentExample]:
    sr_dataset = dataset_name_from_path(sr_path)
    st_dataset = dataset_name_from_path(st_path)
    if sr_dataset != st_dataset:
        raise ValueError(f"Mismatched STM datasets: {sr_path} vs {st_path}")

    sr_rows = parse_stm_file(sr_path)
    st_rows = parse_stm_file(st_path)
    sr_map = {utterance_key(utt): utt for utt in sr_rows}
    st_map = {utterance_key(utt): utt for utt in st_rows}

    missing_in_st = sorted(set(sr_map) - set(st_map))
    missing_in_sr = sorted(set(st_map) - set(sr_map))
    if missing_in_st or missing_in_sr:
        raise ValueError(
            f"SR/ST STM mismatch for {sr_dataset}: "
            f"{len(missing_in_st)} missing in ST, {len(missing_in_sr)} missing in SR"
        )

    aligned = []
    for key in sorted(sr_map.keys(), key=lambda item: (item[0], item[1], item[2], item[3])):
        sr_utt = sr_map[key]
        st_utt = st_map[key]
        aligned.append(
            SegmentExample(
                audio_path=key[0],
                channel=key[1],
                speaker=sr_utt.speaker,
                start_ms=key[2],
                stop_ms=key[3],
                transcript=sr_utt.transcript.strip(),
                translation=st_utt.transcript.strip(),
                source_dataset=sr_dataset,
                split=split,
            )
        )
    return aligned


def combine_training_examples(examples: Iterable[SegmentExample]) -> List[SegmentExample]:
    combined: Dict[Tuple[str, str, int, int], SegmentExample] = {}
    for example in examples:
        key = (example.audio_path, example.channel, example.start_ms, example.stop_ms)
        if key in combined:
            existing = combined[key]
            if existing.transcript != example.transcript or existing.translation != example.translation:
                raise ValueError(f"Conflicting duplicate training segment for key={key}")
            continue
        combined[key] = example
    return sorted(
        combined.values(),
        key=lambda item: (item.audio_path, item.channel, item.start_ms, item.stop_ms, item.speaker),
    )


def normalize_ratios(ratios: Sequence[float]) -> List[float]:
    if len(ratios) != 3:
        raise ValueError("Expected exactly 3 ratios.")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Ratios must be non-negative.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one ratio must be positive.")
    return [ratio / total for ratio in ratios]


def sample_target_duration(rng: random.Random, mean: float, std: float, t_min: float, t_max: float) -> float:
    if std <= 0:
        return max(t_min, min(mean, t_max))
    while True:
        candidate = rng.gauss(mean, std)
        if t_min <= candidate <= t_max:
            return candidate


def merged_uttid(example_group: Sequence[SegmentExample]) -> str:
    first = example_group[0]
    last = example_group[-1]
    recording_id = Path(first.audio_path).stem.replace(".", "_")
    return f"merged-{recording_id}-{first.channel}_{first.start_ms:08d}_{last.stop_ms:08d}"


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def merged_duration(example_group: Sequence[SegmentExample], allow_gap: bool) -> float:
    if allow_gap:
        return max(0.0, example_group[-1].stop_time - example_group[0].start_time)
    return sum(example.duration for example in example_group)


def serialize_record(
    example_group: Sequence[SegmentExample],
    dataset_name: str,
    supervision_mode: str,
    src_lang: str,
    tgt_lang: str,
    allow_gap: bool,
):
    first = example_group[0]
    last = example_group[-1]
    transcript = " ".join(example.transcript for example in example_group if example.transcript).strip()
    translation = " ".join(example.translation for example in example_group if example.translation).strip()

    if supervision_mode == "asr_only":
        translation = ""
    elif supervision_mode == "st_only":
        transcript = ""

    return {
        "uttid": merged_uttid(example_group) if len(example_group) > 1 else passthrough_uttid(first),
        "dataset_name": dataset_name,
        "split": example_group[0].split,
        "supervision_mode": supervision_mode,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "audio_path": first.audio_path,
        "channel": first.channel,
        "speaker": first.speaker,
        "start_time": first.start_time,
        "stop_time": last.stop_time,
        "duration": merged_duration(example_group, allow_gap),
        "transcript": transcript,
        "translation": translation,
        "source_datasets": sorted({example.source_dataset for example in example_group}),
        "segments": [
            {
                "speaker": example.speaker,
                "start_time": example.start_time,
                "stop_time": example.stop_time,
                "transcript": example.transcript,
                "translation": example.translation,
            }
            for example in example_group
        ],
    }


def passthrough_uttid(example: SegmentExample) -> str:
    recording_id = Path(example.audio_path).stem.replace(".", "_")
    return f"{example.speaker}-{recording_id}-{example.channel}_{example.start_ms:08d}_{example.stop_ms:08d}"


def build_merged_training_records(
    examples: Sequence[SegmentExample],
    src_lang: str,
    tgt_lang: str,
    seed: int,
    mean: float,
    std: float,
    t_min: float,
    t_max: float,
    allow_gap: bool,
) -> List[dict]:
    rng = random.Random(seed)
    grouped: Dict[Tuple[str, str], List[SegmentExample]] = defaultdict(list)
    for example in examples:
        grouped[(example.audio_path, example.channel)].append(example)

    merged_records = []
    for _, group in sorted(grouped.items(), key=lambda item: item[0]):
        group = sorted(group, key=lambda item: (item.start_ms, item.stop_ms, item.speaker))
        bucket: List[SegmentExample] = []
        bucket_target = sample_target_duration(rng, mean, std, t_min, t_max)
        bucket_duration = 0.0

        for example in group:
            if bucket and bucket_duration + example.duration > bucket_target:
                merged_records.append(
                    serialize_record(
                        bucket,
                        MERGED_DATASET_NAMES["3way"],
                        "3way",
                        src_lang,
                        tgt_lang,
                        allow_gap,
                    )
                )
                bucket = []
                bucket_duration = 0.0
                bucket_target = sample_target_duration(rng, mean, std, t_min, t_max)

            bucket.append(example)
            bucket_duration += example.duration

        if bucket:
            merged_records.append(
                serialize_record(
                    bucket,
                    MERGED_DATASET_NAMES["3way"],
                    "3way",
                    src_lang,
                    tgt_lang,
                    allow_gap,
                )
            )

    return merged_records


def split_training_records(records: Sequence[dict], ratios: Sequence[float], seed: int) -> Dict[str, List[dict]]:
    normalized = normalize_ratios(ratios)
    assignments = [
        (MERGED_DATASET_NAMES["3way"], "3way"),
        (MERGED_DATASET_NAMES["asr_only"], "asr_only"),
        (MERGED_DATASET_NAMES["st_only"], "st_only"),
    ]
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    if normalized[0] == 1.0 and normalized[1] == 0.0 and normalized[2] == 0.0:
        only_records = []
        for record in shuffled:
            new_record = dict(record)
            new_record["dataset_name"] = MERGED_DATASET_NAMES["3way"]
            new_record["supervision_mode"] = "3way"
            only_records.append(new_record)
        return {MERGED_DATASET_NAMES["3way"]: sorted(only_records, key=lambda item: item["uttid"])}

    raw_counts = [len(shuffled) * ratio for ratio in normalized]
    counts = [int(math.floor(value)) for value in raw_counts]
    remainder = len(shuffled) - sum(counts)
    ranked_indices = sorted(
        range(len(raw_counts)),
        key=lambda idx: (raw_counts[idx] - counts[idx], -idx),
        reverse=True,
    )
    for idx in ranked_indices[:remainder]:
        counts[idx] += 1

    outputs: Dict[str, List[dict]] = {}
    cursor = 0
    for idx, (dataset_name, supervision_mode) in enumerate(assignments):
        if counts[idx] == 0:
            continue
        subset = []
        for record in shuffled[cursor: cursor + counts[idx]]:
            new_record = dict(record)
            new_record["dataset_name"] = dataset_name
            new_record["supervision_mode"] = supervision_mode
            if supervision_mode == "asr_only":
                new_record["translation"] = ""
            elif supervision_mode == "st_only":
                new_record["transcript"] = ""
            subset.append(new_record)
        outputs[dataset_name] = sorted(subset, key=lambda item: item["uttid"])
        cursor += counts[idx]
    return outputs


def build_passthrough_records(examples: Sequence[SegmentExample], src_lang: str, tgt_lang: str) -> Dict[str, List[dict]]:
    outputs: Dict[str, List[dict]] = defaultdict(list)
    for example in examples:
        record = {
            "uttid": passthrough_uttid(example),
            "dataset_name": mark_as_original(example.source_dataset),
            "split": example.split,
            "supervision_mode": "3way",
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "audio_path": example.audio_path,
            "channel": example.channel,
            "speaker": example.speaker,
            "start_time": example.start_time,
            "stop_time": example.stop_time,
            "duration": example.duration,
            "transcript": example.transcript,
            "translation": example.translation,
            "source_datasets": [example.source_dataset],
            "segments": [
                {
                    "speaker": example.speaker,
                    "start_time": example.start_time,
                    "stop_time": example.stop_time,
                    "transcript": example.transcript,
                    "translation": example.translation,
                }
            ],
        }
        outputs[mark_as_original(example.source_dataset)].append(record)

    return {
        dataset_name: sorted(records, key=lambda item: item["uttid"])
        for dataset_name, records in outputs.items()
    }


def write_manifest(path: Path, records: Sequence[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_stats(path: Path, stats: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, indent=2, ensure_ascii=False)


def collect_pairs(sr_paths: Sequence[str], st_paths: Sequence[str], split: str) -> List[SegmentExample]:
    if len(sr_paths) != len(st_paths):
        raise ValueError(f"{split}: expected the same number of SR and ST STMs.")
    aligned = []
    for sr_path, st_path in zip(sr_paths, st_paths):
        aligned.extend(align_pair(Path(sr_path), Path(st_path), split))
    return aligned


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allow_gap = parse_bool(args.allow_gap)

    train_examples = combine_training_examples(
        collect_pairs(args.train_sr_stms, args.train_st_stms, "train")
    )
    dev_examples = collect_pairs(args.dev_sr_stms, args.dev_st_stms, "dev")
    test_examples = collect_pairs(args.test_sr_stms, args.test_st_stms, "test")

    merged_training = build_merged_training_records(
        examples=train_examples,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        seed=args.seed,
        mean=args.mean,
        std=args.std,
        t_min=args.t_min,
        t_max=args.t_max,
        allow_gap=allow_gap,
    )
    training_outputs = split_training_records(merged_training, args.ratios, args.seed)
    passthrough_outputs = build_passthrough_records(dev_examples + test_examples, args.src_lang, args.tgt_lang)

    manifest_count = 0
    for dataset_name, records in {**training_outputs, **passthrough_outputs}.items():
        write_manifest(args.output_dir / f"{args.src_lang}.{dataset_name}.jsonl", records)
        manifest_count += len(records)

    stats = {
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "seed": args.seed,
        "gaussian": {
            "mean": args.mean,
            "std": args.std,
            "t_min": args.t_min,
            "t_max": args.t_max,
        },
        "allow_gap": allow_gap,
        "ratios": normalize_ratios(args.ratios),
        "counts": {
            "train_input_segments": len(train_examples),
            "train_merged_segments": len(merged_training),
            **{name: len(records) for name, records in training_outputs.items()},
            **{name: len(records) for name, records in passthrough_outputs.items()},
        },
        "manifest_records": manifest_count,
    }
    write_stats(args.output_dir / "stats.json", stats)


if __name__ == "__main__":
    main()
