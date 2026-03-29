#!/usr/bin/env python3

import argparse
import audioop
import io
import json
import os
import shutil as py_shutil
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, Features, Value
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local.data_prep.stm import StmUtterance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize STM manifests into audio, STM, and HF datasets.",
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--stm-dir", type=Path, required=True)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--src-lang", type=str, default="ara")
    parser.add_argument("--tgt-lang", type=str, default="eng")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--audio-format", type=str, default="flac")
    parser.add_argument("--allow-gap", type=str, default="false")
    return parser.parse_args()


def load_manifest(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def channel_to_index(channel: str) -> int:
    return 0 if channel == "A" else 1


def load_segment_from_sph(audio_path: str, channel: str, start_time: float, stop_time: float):
    if py_shutil.which("sph2pipe") is None:
        raise FileNotFoundError("sph2pipe is not installed or not on PATH")
    command = [
        "sph2pipe",
        "-f",
        "wav",
        "-p",
        "-c",
        str(channel_to_index(channel) + 1),
        "-t",
        f"{start_time}:{stop_time}",
        audio_path,
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_array, sampling_rate = sf.read(io.BytesIO(result.stdout), dtype="float32")
    return audio_array, sampling_rate


def load_segment_from_file(audio_path: str, channel: str, start_time: float, stop_time: float):
    with sf.SoundFile(audio_path) as sound_file:
        start_frame = max(0, int(round(start_time * sound_file.samplerate)))
        stop_frame = max(start_frame, int(round(stop_time * sound_file.samplerate)))
        frames = stop_frame - start_frame
        sound_file.seek(start_frame)
        audio_array = sound_file.read(frames=frames, dtype="float32", always_2d=True)
        audio_array = audio_array[:, min(channel_to_index(channel), audio_array.shape[1] - 1)]
        return audio_array, sound_file.samplerate


def load_audio_segment(audio_path: str, channel: str, start_time: float, stop_time: float, target_sr: int):
    try:
        audio_array, sampling_rate = load_segment_from_file(audio_path, channel, start_time, stop_time)
    except Exception as error:
        suffix = Path(audio_path).suffix.lower()
        if suffix != ".sph":
            raise
        try:
            audio_array, sampling_rate = load_segment_from_sph(audio_path, channel, start_time, stop_time)
        except Exception:
            raise RuntimeError(
                f"Failed to read audio segment from {audio_path}. "
                "soundfile could not open it, and sph2pipe fallback is unavailable."
            ) from error

    if sampling_rate != target_sr:
        audio_array = resample_audio(audio_array, sampling_rate, target_sr)
        sampling_rate = target_sr
    return audio_array, sampling_rate


def resample_audio(audio_array: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio_array
    pcm16 = np.clip(audio_array, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
    converted, _ = audioop.ratecv(pcm16, 2, 1, source_sr, target_sr, None)
    resampled = np.frombuffer(converted, dtype=np.int16).astype(np.float32) / 32767.0
    return resampled


def materialize_record(task: Tuple[int, dict, str, int, str, bool]):
    index, record, dataset_audio_dir, target_sr, audio_format, allow_gap = task
    dataset_audio_dir = Path(dataset_audio_dir)
    dataset_audio_dir.mkdir(parents=True, exist_ok=True)
    output_audio_path = dataset_audio_dir / f"{record['uttid']}.{audio_format}"

    if not output_audio_path.exists():
        if allow_gap:
            merged_audio, _ = load_audio_segment(
                audio_path=record["audio_path"],
                channel=record["channel"],
                start_time=float(record["start_time"]),
                stop_time=float(record["stop_time"]),
                target_sr=target_sr,
            )
        else:
            audio_segments = []
            for segment in record["segments"]:
                audio_array, _ = load_audio_segment(
                    audio_path=record["audio_path"],
                    channel=record["channel"],
                    start_time=segment["start_time"],
                    stop_time=segment["stop_time"],
                    target_sr=target_sr,
                )
                audio_segments.append(audio_array)

            if not audio_segments:
                raise ValueError(f"No audio segments found for {record['uttid']}")

            merged_audio = audio_segments[0] if len(audio_segments) == 1 else np.concatenate(audio_segments)
        sf.write(output_audio_path, merged_audio, target_sr)

    return {
        "index": index,
        "uttid": record["uttid"],
        "audio_path": str(output_audio_path),
        "transcript": record["transcript"],
        "translation": record["translation"],
        "src_lang": record["src_lang"],
        "tgt_lang": record["tgt_lang"],
        "supervision_mode": record["supervision_mode"],
        "source_split": record["split"],
        "speaker": record["speaker"],
        "duration": record["duration"],
    }


def create_dataset(rows: List[dict], dataset_path: Path, sampling_rate: int):
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    features = Features(
        {
            "audio": Audio(sampling_rate=sampling_rate),
            "uttid": Value(dtype="string"),
            "transcript": Value(dtype="string"),
            "translation": Value(dtype="string"),
            "src_lang": Value(dtype="string"),
            "tgt_lang": Value(dtype="string"),
            "supervision_mode": Value(dtype="string"),
            "source_split": Value(dtype="string"),
        }
    )
    payload = {
        "audio": [row["audio_path"] for row in rows],
        "uttid": [row["uttid"] for row in rows],
        "transcript": [row["transcript"] for row in rows],
        "translation": [row["translation"] for row in rows],
        "src_lang": [row["src_lang"] for row in rows],
        "tgt_lang": [row["tgt_lang"] for row in rows],
        "supervision_mode": [row["supervision_mode"] for row in rows],
        "source_split": [row["source_split"] for row in rows],
    }
    dataset = Dataset.from_dict(payload, features=features)
    dataset.save_to_disk(str(dataset_path))


def write_stms(rows: List[dict], dataset_name: str, stm_dir: Path, src_lang: str, tgt_lang: str):
    language_dir = stm_dir / src_lang
    language_dir.mkdir(parents=True, exist_ok=True)
    sr_path = language_dir / f"sr.{src_lang}-{src_lang}.{dataset_name}.stm"
    st_path = language_dir / f"st.{src_lang}-{tgt_lang}.{dataset_name}.stm"

    with open(sr_path, "w", encoding="utf-8") as sr_file, open(st_path, "w", encoding="utf-8") as st_file:
        for row in rows:
            speaker = row["speaker"]
            if speaker == "":
                speaker = row["uttid"].split("-")[0]
            sr_utt = StmUtterance(
                filename=row["audio_path"],
                channel="A",
                speaker=speaker,
                start_time=0.0,
                stop_time=max(0.01, float(row["duration"])),
                transcript=row["transcript"],
            )
            st_utt = StmUtterance(
                filename=row["audio_path"],
                channel="A",
                speaker=speaker,
                start_time=0.0,
                stop_time=max(0.01, float(row["duration"])),
                transcript=row["translation"],
            )
            print(sr_utt, file=sr_file)
            print(st_utt, file=st_file)


def save_summary(summary: Dict[str, int], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    allow_gap = parse_bool(args.allow_gap)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.stm_dir.mkdir(parents=True, exist_ok=True)
    args.hf_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = sorted(args.manifest_dir.glob(f"{args.src_lang}.*.jsonl"))
    if not manifest_paths:
        raise FileNotFoundError(f"No manifests found under {args.manifest_dir}")

    summary = {}
    for manifest_path in manifest_paths:
        dataset_name = manifest_path.stem.split(".", maxsplit=1)[1]
        records = load_manifest(manifest_path)
        dataset_audio_dir = args.audio_dir / dataset_name
        print(
            f"Materializing {dataset_name}: {len(records)} utterances "
            f"with {max(1, args.num_workers)} workers"
        )

        tasks = [
            (index, record, str(dataset_audio_dir), args.sampling_rate, args.audio_format, allow_gap)
            for index, record in enumerate(records)
        ]
        with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
            rows = list(
                tqdm(
                    executor.map(materialize_record, tasks),
                    total=len(tasks),
                    desc=f"Stage2 {dataset_name}",
                    unit="utt",
                )
            )
        rows = sorted(rows, key=lambda item: item["index"])

        hf_dataset_path = args.hf_dir / f"{args.src_lang}.{dataset_name}"
        create_dataset(rows, hf_dataset_path, args.sampling_rate)
        write_stms(rows, dataset_name, args.stm_dir, args.src_lang, args.tgt_lang)
        summary[dataset_name] = len(rows)

    save_summary(summary, args.hf_dir / f"{args.src_lang}.summary.json")


if __name__ == "__main__":
    main()
