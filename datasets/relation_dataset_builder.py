from __future__ import annotations

import json
from pathlib import Path

from datasets.scierc_loader import ner_global_to_local


def mark_entity_pair(
    tokens: list[str],
    head_span: tuple[int, int],
    tail_span: tuple[int, int],
) -> str:
    out: list[str] = []
    for idx, tok in enumerate(tokens):
        if idx == head_span[0]:
            out.append("[E1]")
        if idx == tail_span[0]:
            out.append("[E2]")

        out.append(tok)

        if idx == head_span[1]:
            out.append("[/E1]")
        if idx == tail_span[1]:
            out.append("[/E2]")
    return " ".join(out)


def build_relation_examples(record: dict, include_none: bool = True) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for sent_idx, tokens in enumerate(record["sentences"]):
        entities = record["ner"][sent_idx]
        gold_relations = record["relations"][sent_idx]
        sentences = record["sentences"]

        rel_map: dict[tuple[tuple[int, int], tuple[int, int]], str] = {}
        for h_start, h_end, t_start, t_end, rel_type in gold_relations:
            hh = ner_global_to_local(sentences, sent_idx, h_start, h_end)
            tt = ner_global_to_local(sentences, sent_idx, t_start, t_end)
            if hh is None or tt is None:
                continue
            rel_map[(hh, tt)] = rel_type.upper()

        spans: list[tuple[int, int]] = []
        for g_s, g_e, _ in entities:
            loc = ner_global_to_local(sentences, sent_idx, g_s, g_e)
            if loc is not None:
                spans.append(loc)
        for head in spans:
            for tail in spans:
                if head == tail:
                    continue
                label = rel_map.get((head, tail), "NONE")
                if label == "NONE" and not include_none:
                    continue
                text = mark_entity_pair(tokens, head, tail)
                examples.append({"text": text, "label": label})
    return examples


def build_relation_dataset(path: str | Path, include_none: bool = True) -> list[dict[str, str]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples: list[dict[str, str]] = []
    for row in rows:
        examples.extend(build_relation_examples(row, include_none=include_none))
    return examples
