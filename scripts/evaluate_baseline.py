import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Allow running as `python evaluate_baseline.py` without installation.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline.evaluation.baseline_evaluator import BaselineEvaluator
from nlp.ner.predictor import SciBERTNER
from nlp.re.predictor import SciBERTRE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline evaluation for SciBERT NER and RE.")
    parser.add_argument("--split", default="datasets/scierc/test.json", help="SciERC split path (jsonl).")
    parser.add_argument("--ner-model", default="artifacts/ner", help="NER local checkpoint dir.")
    parser.add_argument("--re-model", default="artifacts/re", help="RE local checkpoint dir.")
    parser.add_argument(
        "--re-entities",
        choices=["gold", "pred"],
        default="gold",
        help="Use gold entities for RE eval or predicted entities (end-to-end).",
    )
    parser.add_argument(
        "--ner-verbose",
        action="store_true",
        help="Print NER corpus stats, per-type P/R/F1, truncation counts.",
    )
    parser.add_argument(
        "--ner-errors",
        type=int,
        default=0,
        metavar="N",
        help="Collect up to N NER FP/FN examples (preview in verbose; use --ner-errors-jsonl to save all).",
    )
    parser.add_argument(
        "--ner-errors-jsonl",
        type=str,
        default=None,
        help="Write NER error samples (FP/FN) to this JSONL path.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write full evaluation statistics to this JSON file (UTF-8).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save JSON to .cache/evaluation/baseline_eval_<timestamp>.json (ignored if --output is set).",
    )
    return parser.parse_args()


def _fmt(name: str, p: float, r: float, f1: float) -> str:
    return f"{name}: P={p:.4f} R={r:.4f} F1={f1:.4f}"


def _prf_dict(p: float, r: float, f1: float) -> dict[str, float]:
    return {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6)}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    ner = SciBERTNER(local_model_dir=args.ner_model)
    re = SciBERTRE(local_model_dir=args.re_model)
    evaluator = BaselineEvaluator(ner=ner, re=re)

    logging.info("Running NER baseline evaluation on %s", args.split)
    want_report = args.ner_verbose or args.ner_errors > 0 or args.ner_errors_jsonl is not None

    ner_report_serializable: dict | None = None

    if want_report:
        micro, report = evaluator.evaluate_ner(
            args.split,
            verbose=args.ner_verbose,
            error_limit=args.ner_errors,
            error_jsonl=args.ner_errors_jsonl,
        )
        ner_scores = micro
        print(_fmt("NER", micro.precision, micro.recall, micro.f1))
        if args.ner_verbose:
            print(
                f"  [counts] gold_entities={report.n_gold_entities} pred_entities={report.n_pred_entities} "
                f"sentences={report.n_sentences} truncated_sentences={report.n_truncated_sentences} "
                f"pred_char_align_fallback={report.n_pred_char_align_fallback}"
            )
            print("  [per-type]")
            for t, m in sorted(report.per_type.items()):
                print(f"    {t}: P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f}")
            ner_report_serializable = {
                "counts": {
                    "gold_entities": report.n_gold_entities,
                    "pred_entities": report.n_pred_entities,
                    "sentences": report.n_sentences,
                    "truncated_sentences": report.n_truncated_sentences,
                    "pred_char_align_fallback": report.n_pred_char_align_fallback,
                },
                "per_type": {
                    t: _prf_dict(m.precision, m.recall, m.f1)
                    for t, m in sorted(report.per_type.items())
                },
            }
        if report.error_samples:
            preview = min(5, len(report.error_samples))
            print(f"  [error preview] {preview} of {len(report.error_samples)} samples")
            for s in report.error_samples[:preview]:
                print(f"    {s['kind']} type={s['type']} tok={s['span_tok']} text={s['text']!r}")
    else:
        ner_scores = evaluator.evaluate_ner(args.split)
        assert not isinstance(ner_scores, tuple)
        print(_fmt("NER", ner_scores.precision, ner_scores.recall, ner_scores.f1))

    use_gold = args.re_entities == "gold"
    logging.info(
        "Running RE baseline evaluation on %s (entities=%s)",
        args.split,
        args.re_entities,
    )
    re_scores = evaluator.evaluate_re(args.split, use_gold_entities=use_gold)
    print(_fmt("RE", re_scores.precision, re_scores.recall, re_scores.f1))

    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    elif args.save:
        eval_dir = ROOT / ".cache" / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stem = Path(args.split).stem
        out_path = eval_dir / f"baseline_eval_{stem}_{ts}.json"

    if out_path is not None:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "split": str(Path(args.split).expanduser().resolve()),
            "ner_model": str(Path(args.ner_model).expanduser().resolve()),
            "re_model": str(Path(args.re_model).expanduser().resolve()),
            "re_entities": args.re_entities,
            "ner": _prf_dict(ner_scores.precision, ner_scores.recall, ner_scores.f1),
            "re": _prf_dict(re_scores.precision, re_scores.recall, re_scores.f1),
        }
        if ner_report_serializable is not None:
            payload["ner_diagnostics"] = ner_report_serializable

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote evaluation statistics to %s", out_path)


if __name__ == "__main__":
    main()
