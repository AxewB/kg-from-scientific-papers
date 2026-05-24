from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from domain.entity import scierc_ner_label_to_bio_suffix
from datasets.scierc_loader import load_scierc_split, ner_global_to_local
from nlp.ner.predictor import SciBERTNER
from nlp.re.predictor import SciBERTRE

lg = logging.getLogger(__name__)


@dataclass
class PRF1:
    precision: float
    recall: float
    f1: float


@dataclass
class NERDiagnosticReport:
    micro: PRF1
    per_type: dict[str, PRF1] = field(default_factory=dict)
    n_gold_entities: int = 0
    n_pred_entities: int = 0
    n_sentences: int = 0
    n_truncated_sentences: int = 0
    n_pred_char_align_fallback: int = 0
    error_samples: list[dict] = field(default_factory=list)


def _compute_prf1(tp: int, fp: int, fn: int) -> PRF1:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF1(precision=precision, recall=recall, f1=f1)


def _token_char_ranges(tokens: list[str]) -> list[tuple[int, int]]:
    """Exclusive character spans for each token in ``" ".join(tokens)``."""
    ranges: list[tuple[int, int]] = []
    offset = 0
    for i, tok in enumerate(tokens):
        start = offset
        end = start + len(tok)
        ranges.append((start, end))
        offset = end + (1 if i + 1 < len(tokens) else 0)
    return ranges


def _char_span_to_token_span_inclusive(
    tokens: list[str], char_start: int, char_end: int
) -> tuple[int, int] | None:
    """
    Map a predicted [char_start, char_end) span (HF-style, end exclusive) to SciERC
    inclusive token indices (start_tok, end_tok).

    Prefer exact alignment to token boundaries; fall back to maximal contiguous overlap.
    """
    ranges = _token_char_ranges(tokens)
    if not ranges:
        return None

    # Exact: find i,j with ranges[i][0] == char_start and ranges[j][1] == char_end
    for i, (ts, te) in enumerate(ranges):
        if ts != char_start:
            continue
        for j in range(i, len(ranges)):
            if ranges[j][1] == char_end:
                return i, j
            if ranges[j][1] > char_end:
                break

    # Overlap: tokens intersecting [char_start, char_end)
    overlapping: list[int] = []
    for i, (ts, te) in enumerate(ranges):
        if ts < char_end and te > char_start:
            overlapping.append(i)
    if not overlapping:
        return None
    lo, hi = overlapping[0], overlapping[-1]
    if hi - lo + 1 != len(overlapping):
        return None
    return lo, hi


class BaselineEvaluator:
    def __init__(self, ner: SciBERTNER, re: SciBERTRE | None = None) -> None:
        self.ner = ner
        self.re = re

    def evaluate_ner(
        self,
        split_path: str | Path,
        *,
        verbose: bool = False,
        error_limit: int = 0,
        error_jsonl: str | Path | None = None,
    ) -> PRF1 | tuple[PRF1, NERDiagnosticReport]:
        """
        Entity-level micro P/R/F1 in **token space** (SciERC native).

        Predictions use ``Entity.start_tok`` / ``end_tok`` when set (no lossy char round-trip).

        Returns ``(PRF1, NERDiagnosticReport)`` if ``verbose``, ``error_limit > 0``, or ``error_jsonl`` is set.

        If ``error_jsonl`` is set but ``error_limit`` is 0, up to 100 samples are collected.
        """
        if error_jsonl is not None and error_limit == 0:
            error_limit = 100
        need_report = verbose or error_limit > 0 or error_jsonl is not None
        path = Path(split_path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        tp = fp = fn = 0
        tp_t: dict[str, int] = defaultdict(int)
        fp_t: dict[str, int] = defaultdict(int)
        fn_t: dict[str, int] = defaultdict(int)
        n_gold = n_pred = n_sents = 0
        n_trunc = 0
        n_align_fallback = 0
        errors: list[dict] = []

        for doc_i, row in enumerate(tqdm(rows, desc="NER evaluation", unit="doc")):
            sentences = row["sentences"]
            ner = row["ner"]
            for sentence_id, tokens in enumerate(sentences):
                n_sents += 1
                if self.ner.was_truncated(tokens):
                    n_trunc += 1

                pred_entities = self.ner.predict_from_tokens(tokens, sentence_id=sentence_id)

                gold_set: set[tuple[int, int, int, str]] = set()
                for g_s, g_e, label in ner[sentence_id]:
                    loc = ner_global_to_local(sentences, sentence_id, g_s, g_e)
                    if loc is None:
                        continue
                    start_tok, end_tok = loc
                    gold_set.add(
                        (
                            sentence_id,
                            start_tok,
                            end_tok,
                            scierc_ner_label_to_bio_suffix(label),
                        )
                    )

                pred_set: set[tuple[int, int, int, str]] = set()
                for e in pred_entities:
                    if e.start_tok is not None and e.end_tok is not None:
                        pred_set.add((e.sentence_id, e.start_tok, e.end_tok, e.label.value))
                    else:
                        aligned = _char_span_to_token_span_inclusive(tokens, e.start, e.end)
                        n_align_fallback += 1
                        if aligned is None:
                            continue
                        st, et = aligned
                        pred_set.add((e.sentence_id, st, et, e.label.value))

                n_gold += len(gold_set)
                n_pred += len(pred_set)

                inter = pred_set & gold_set
                tp += len(inter)
                fp += len(pred_set - gold_set)
                fn += len(gold_set - pred_set)

                for key in inter:
                    tp_t[key[3]] += 1
                for key in pred_set - gold_set:
                    fp_t[key[3]] += 1
                for key in gold_set - pred_set:
                    fn_t[key[3]] += 1

                if error_limit > 0 and len(errors) < error_limit:
                    sentence = " ".join(tokens)
                    for key in gold_set - pred_set:
                        if len(errors) >= error_limit:
                            break
                        _, st, et, lab = key
                        text = " ".join(tokens[st : et + 1])
                        errors.append(
                            {
                                "kind": "fn",
                                "doc_idx": doc_i,
                                "sentence_id": sentence_id,
                                "span_tok": [st, et],
                                "type": lab,
                                "text": text,
                                "sentence_preview": sentence[:200],
                            }
                        )
                    for key in pred_set - gold_set:
                        if len(errors) >= error_limit:
                            break
                        _, st, et, lab = key
                        text = " ".join(tokens[st : et + 1])
                        errors.append(
                            {
                                "kind": "fp",
                                "doc_idx": doc_i,
                                "sentence_id": sentence_id,
                                "span_tok": [st, et],
                                "type": lab,
                                "text": text,
                                "sentence_preview": sentence[:200],
                            }
                        )

        micro = _compute_prf1(tp, fp, fn)
        if not need_report:
            return micro

        all_types = sorted(set(tp_t) | set(fp_t) | set(fn_t))
        per_type: dict[str, PRF1] = {}
        for t in all_types:
            per_type[t] = _compute_prf1(tp_t[t], fp_t[t], fn_t[t])

        report = NERDiagnosticReport(
            micro=micro,
            per_type=per_type,
            n_gold_entities=n_gold,
            n_pred_entities=n_pred,
            n_sentences=n_sents,
            n_truncated_sentences=n_trunc,
            n_pred_char_align_fallback=n_align_fallback,
            error_samples=errors[:error_limit],
        )

        if error_jsonl:
            out = Path(error_jsonl)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                for e in report.error_samples:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            lg.info("Wrote %d NER error samples to %s", len(report.error_samples), out)

        return micro, report

    def evaluate_ner_simple(self, split_path: str | Path) -> PRF1:
        """Backward-compatible: micro F1 only."""
        result = self.evaluate_ner(split_path)
        return result[0] if isinstance(result, tuple) else result

    def evaluate_re(self, split_path: str | Path, use_gold_entities: bool = True) -> PRF1:
        if self.re is None:
            raise ValueError("Relation model is not configured (pass SciBERTRE to BaselineEvaluator).")
        docs = load_scierc_split(split_path)
        tp = fp = fn = 0

        for doc in tqdm(docs, desc="RE evaluation", unit="doc"):
            for sentence_id, tokens in enumerate(doc.sentences):
                sentence = " ".join(tokens)

                if use_gold_entities:
                    entities = [e for e in doc.entities if e.sentence_id == sentence_id]
                else:
                    entities = self.ner.predict_from_tokens(tokens, sentence_id=sentence_id)

                pred_relations = self.re.predict(sentence, entities)
                gold_relations = [
                    r for r in doc.relations
                    if r.head.sentence_id == sentence_id and r.tail.sentence_id == sentence_id
                ]

                pred_set: set[tuple[int, int, int, int, int, str]] = set()
                for r in pred_relations:
                    ha = _char_span_to_token_span_inclusive(tokens, r.head.start, r.head.end)
                    ta = _char_span_to_token_span_inclusive(tokens, r.tail.start, r.tail.end)
                    if ha is None or ta is None:
                        continue
                    h0, h1 = ha
                    t0, t1 = ta
                    pred_set.add(
                        (sentence_id, h0, h1, t0, t1, r.type.value),
                    )

                gold_set: set[tuple[int, int, int, int, int, str]] = set()
                for r in gold_relations:
                    ha = _char_span_to_token_span_inclusive(tokens, r.head.start, r.head.end)
                    ta = _char_span_to_token_span_inclusive(tokens, r.tail.start, r.tail.end)
                    if ha is None or ta is None:
                        continue
                    gold_set.add(
                        (sentence_id, ha[0], ha[1], ta[0], ta[1], r.type.value),
                    )

                tp += len(pred_set & gold_set)
                fp += len(pred_set - gold_set)
                fn += len(gold_set - pred_set)

        return _compute_prf1(tp, fp, fn)
