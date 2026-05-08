from __future__ import annotations

import json
from pathlib import Path


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

        rel_map: dict[tuple[tuple[int, int], tuple[int, int]], str] = {}
        for h_start, h_end, t_start, t_end, rel_type in gold_relations:
            rel_map[((h_start, h_end), (t_start, t_end))] = rel_type.upper()

        spans = [(start, end) for start, end, _ in entities]
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
