#!/usr/bin/env python3
"""
Isolated NER analysis: corpus stats + dev/test metrics + optional error dump.

Does not train; use after `python train_ner.py` to inspect checkpoint behavior.

Example:
  python analyze_ner.py --train datasets/scierc/train.json --dev datasets/scierc/dev.json \\
    --ner-model artifacts/ner --errors-jsonl .cache/ner_errors.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from domain.entity import scierc_ner_label_to_bio_suffix
from pipeline.evaluation.baseline_evaluator import BaselineEvaluator
from pipeline.ner.predictor import SciBERTNER


def corpus_stats(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_docs = len(rows)
    n_sents = 0
    n_toks = 0
    n_ent = 0
    by_label: Counter[str] = Counter()
    toks_per_sent: list[int] = []

    for row in rows:
        for sent_id, tokens in enumerate(row["sentences"]):
            n_sents += 1
            n_toks += len(tokens)
            toks_per_sent.append(len(tokens))
            for start_tok, end_tok, lab in row["ner"][sent_id]:
                n_ent += 1
                by_label[scierc_ner_label_to_bio_suffix(lab)] += 1

    print(f"File: {path}")
    print(f"  documents: {n_docs}")
    print(f"  sentences: {n_sents}")
    print(f"  tokens: {n_toks} (avg {n_toks / n_sents:.1f} / sentence)")
    print(f"  entities: {n_ent} (avg {n_ent / n_sents:.2f} / sentence)")
    print("  entities by type:")
    for lab, c in sorted(by_label.items(), key=lambda x: -x[1]):
        print(f"    {lab}: {c}")
    long = sum(1 for t in toks_per_sent if t > 128)
    print(f"  sentences with >128 tokens: {long} ({100 * long / n_sents:.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze SciERC NER corpus and checkpoint metrics.")
    p.add_argument("--train", type=str, default=None, help="Optional train.jsonl for corpus stats.")
    p.add_argument("--dev", type=str, default="datasets/scierc/dev.json")
    p.add_argument("--test", type=str, default=None, help="Optional second split to evaluate.")
    p.add_argument("--ner-model", type=str, default="artifacts/ner")
    p.add_argument("--errors-jsonl", type=str, default=None, help="Write FP/FN samples from --dev.")
    p.add_argument("--error-limit", type=int, default=150)
    args = p.parse_args()

    if args.train:
        print("=== Train corpus ===")
        corpus_stats(Path(args.train))
        print()

    print("=== Dev corpus ===")
    corpus_stats(Path(args.dev))
    print()

    ner = SciBERTNER(local_model_dir=args.ner_model)
    ev = BaselineEvaluator(ner=ner, re=None)

    print("=== NER metrics (dev) ===")
    micro, report = ev.evaluate_ner(
        args.dev,
        verbose=True,
        error_limit=args.error_limit,
        error_jsonl=args.errors_jsonl,
    )
    print(
        f"Summary: P={micro.precision:.4f} R={micro.recall:.4f} F1={micro.f1:.4f} | "
        f"truncated_sents={report.n_truncated_sentences}/{report.n_sentences}"
    )

    if args.test:
        print()
        print("=== NER metrics (test) ===")
        micro_t, report_t = ev.evaluate_ner(Path(args.test), verbose=True)
        print(
            f"Summary: P={micro_t.precision:.4f} R={micro_t.recall:.4f} F1={micro_t.f1:.4f} | "
            f"truncated_sents={report_t.n_truncated_sentences}/{report_t.n_sentences}"
        )

    print()
    print("Hints if recall is low:")
    print("  - Increase epochs / check dev loss in Trainer output")
    print("  - SciERC has nested/overlapping mentions; plain BIO cannot fit all")
    print("  - Try max_length 512 if many truncated sentences")
    print("  - Compare per-type F1: one rare class can dominate FN")


if __name__ == "__main__":
    main()
