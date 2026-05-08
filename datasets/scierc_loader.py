from __future__ import annotations

"""
SciERC jsonl: ``ner[s]`` and ``relations[s]`` use **document-global** token indices
(cumulative over ``sentences[0]..sentences[s]``). Convert to sentence-local with
``ner_global_to_local`` before spans, BIO, or string joins.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from domain.entity import Entity, EntityType, scierc_ner_label_to_bio_suffix
from domain.relation import Relation, RelationType


@dataclass
class SciERCDocument:
    sentences: list[list[str]]
    entities: list[Entity]
    relations: list[Relation]


def scierc_sentence_token_base(sentences: list[list[str]], sent_idx: int) -> int:
    """Cumulative token offset at start of sentence ``sent_idx`` (SciERC uses **document-global** token indices)."""
    return sum(len(sentences[i]) for i in range(sent_idx))


def ner_global_to_local(
    sentences: list[list[str]], sent_idx: int, global_start: int, global_end: int
) -> tuple[int, int] | None:
    base = scierc_sentence_token_base(sentences, sent_idx)
    ls, le = global_start - base, global_end - base
    L = len(sentences[sent_idx])
    if 0 <= ls <= le < L:
        return ls, le
    return None


def _token_span_to_char_span(tokens: list[str], start_tok: int, end_tok: int) -> tuple[int, int]:
    start = sum(len(t) + 1 for t in tokens[:start_tok])
    end = sum(len(t) + 1 for t in tokens[: end_tok + 1]) - 1
    return start, end


def _map_entity_type(label: str) -> EntityType:
    return EntityType(scierc_ner_label_to_bio_suffix(label))


def _map_relation_type(label: str) -> RelationType | None:
    mapping = {
        "USED-FOR": RelationType.USED_FOR,
        "PART-OF": RelationType.PART_OF,
        "FEATURE-OF": RelationType.FEATURE_OF,
        "HYPONYM-OF": RelationType.HYPONYM_OF,
        "CONJUNCTION": RelationType.CONJUNCTION,
        "COMPARE": RelationType.COMPARE,
    }
    return mapping.get(label.upper())


def parse_scierc_record(record: dict) -> SciERCDocument:
    sentences: list[list[str]] = record["sentences"]
    ner = record["ner"]
    rel = record["relations"]

    entities: list[Entity] = []
    ent_index: dict[tuple[int, int, int], Entity] = {}

    for sentence_id, sentence_ents in enumerate(ner):
        tokens = sentences[sentence_id]
        for g_start, g_end, label in sentence_ents:
            local = ner_global_to_local(sentences, sentence_id, g_start, g_end)
            if local is None:
                continue
            start_tok, end_tok = local
            start, end = _token_span_to_char_span(tokens, start_tok, end_tok)
            text = " ".join(tokens[start_tok : end_tok + 1])
            entity = Entity(
                text=text,
                label=_map_entity_type(label),
                start=start,
                end=end,
                sentence_id=sentence_id,
            )
            entities.append(entity)
            ent_index[(sentence_id, start_tok, end_tok)] = entity

    relations: list[Relation] = []
    for sentence_id, sentence_rels in enumerate(rel):
        for h_start, h_end, t_start, t_end, r_type in sentence_rels:
            rel_type = _map_relation_type(r_type)
            if rel_type is None:
                continue

            hh = ner_global_to_local(sentences, sentence_id, h_start, h_end)
            tt = ner_global_to_local(sentences, sentence_id, t_start, t_end)
            if hh is None or tt is None:
                continue
            head = ent_index.get((sentence_id, hh[0], hh[1]))
            tail = ent_index.get((sentence_id, tt[0], tt[1]))
            if head is None or tail is None:
                continue

            relations.append(Relation(head=head, tail=tail, type=rel_type))

    return SciERCDocument(
        sentences=sentences,
        entities=entities,
        relations=relations,
    )


def load_scierc_split(path: str | Path) -> list[SciERCDocument]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [parse_scierc_record(row) for row in rows]
