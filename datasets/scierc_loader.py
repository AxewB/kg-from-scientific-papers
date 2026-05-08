from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from domain.entity import Entity, EntityType
from domain.relation import Relation, RelationType


@dataclass
class SciERCDocument:
    sentences: list[list[str]]
    entities: list[Entity]
    relations: list[Relation]


def _token_span_to_char_span(tokens: list[str], start_tok: int, end_tok: int) -> tuple[int, int]:
    start = sum(len(t) + 1 for t in tokens[:start_tok])
    end = sum(len(t) + 1 for t in tokens[: end_tok + 1]) - 1
    return start, end


def _map_entity_type(label: str) -> EntityType:
    mapping = {
        "Task": EntityType.TASK,
        "Method": EntityType.METHOD,
        "Material": EntityType.MATERIAL,
        "Metric": EntityType.METRIC,
        "OtherScientificTerm": EntityType.OTHER,
        "Generic": EntityType.GENERIC,
    }
    return mapping.get(label, EntityType.GENERIC)


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
        for start_tok, end_tok, label in sentence_ents:
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

            head = ent_index.get((sentence_id, h_start, h_end))
            tail = ent_index.get((sentence_id, t_start, t_end))
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
