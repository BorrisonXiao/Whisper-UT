#!/usr/bin/env python3

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score Whisper predictions against a Hugging Face dataset split."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory created by save_to_disk.")
    parser.add_argument("--hyp-file", type=Path, required=True, help="Hypothesis file with lines: uttid text")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["asr", "st", "mt"],
        help="Which reference field to score against.",
    )
    parser.add_argument("--score-dir", type=Path, required=True, help="Directory to store scores and aligned text.")
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Apply simple whitespace normalization before scoring.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def maybe_normalize(text: str, enabled: bool) -> str:
    return normalize_text(text) if enabled else text.strip()


def read_hypotheses(path: Path, normalize: bool) -> dict:
    hypotheses = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            uttid = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            hypotheses[uttid] = maybe_normalize(text, normalize)
    return hypotheses


def levenshtein_distance(ref, hyp):
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            substitution = previous[j - 1] + (ref_item != hyp_item)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def corpus_error_rate(refs, hyps, level="word"):
    total_edits = 0
    total_units = 0
    for ref, hyp in zip(refs, hyps):
        if level == "char":
            ref_units = list(ref)
            hyp_units = list(hyp)
        else:
            ref_units = ref.split()
            hyp_units = hyp.split()
        total_edits += levenshtein_distance(ref_units, hyp_units)
        total_units += len(ref_units)
    return 100.0 * total_edits / max(1, total_units)


def ngrams(tokens, order):
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def corpus_bleu(refs, hyps, max_order=4):
    matches_by_order = [0] * max_order
    totals_by_order = [0] * max_order
    ref_length = 0
    hyp_length = 0

    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)
        for order in range(1, max_order + 1):
            ref_ngrams = ngrams(ref_tokens, order)
            hyp_ngrams = ngrams(hyp_tokens, order)
            overlap = hyp_ngrams & ref_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            totals_by_order[order - 1] += max(0, len(hyp_tokens) - order + 1)

    precisions = []
    for matches, total in zip(matches_by_order, totals_by_order):
        precisions.append((matches + 1.0) / (total + 1.0))

    if hyp_length == 0:
        bleu = 0.0
        bp = 0.0
    else:
        bp = 1.0 if hyp_length > ref_length else math.exp(1.0 - float(ref_length) / hyp_length)
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_order)

    return {
        "bleu": bleu * 100.0,
        "brevity_penalty": bp,
        "ref_length": ref_length,
        "hyp_length": hyp_length,
        "precisions": [p * 100.0 for p in precisions],
    }


def main():
    args = parse_args()
    dataset = load_from_disk(str(args.dataset))
    hypotheses = read_hypotheses(args.hyp_file, args.normalize_text)
    ref_field = "transcript" if args.task == "asr" else "translation"

    refs = []
    hyps = []
    aligned_rows = []
    missing = 0

    for row in dataset:
        uttid = row["uttid"]
        ref = maybe_normalize(row[ref_field], args.normalize_text)
        hyp = hypotheses.get(uttid, "")
        if uttid not in hypotheses:
            missing += 1
        refs.append(ref)
        hyps.append(hyp)
        aligned_rows.append({"uttid": uttid, "ref": ref, "hyp": hyp})

    args.score_dir.mkdir(parents=True, exist_ok=True)
    with open(args.score_dir / "refs.txt", "w", encoding="utf-8") as ref_out:
        for row in aligned_rows:
            print(f"{row['uttid']} {row['ref']}", file=ref_out)
    with open(args.score_dir / "hyps.txt", "w", encoding="utf-8") as hyp_out:
        for row in aligned_rows:
            print(f"{row['uttid']} {row['hyp']}", file=hyp_out)

    metrics = {
        "task": args.task,
        "dataset": str(args.dataset),
        "hyp_file": str(args.hyp_file),
        "num_utts": len(aligned_rows),
        "missing_hypotheses": missing,
    }

    if args.task == "asr":
        metrics["wer"] = corpus_error_rate(refs, hyps, level="word")
        metrics["cer"] = corpus_error_rate(refs, hyps, level="char")
    else:
        metrics.update(corpus_bleu(refs, hyps))

    with open(args.score_dir / "metrics.json", "w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2, ensure_ascii=False)

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
