from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import pipeline

from domain.entity import Entity
from domain.relation import Relation, RelationType


class SciBERTRE:
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        local_model_dir: str = "artifacts/re",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.max_length = max_length
        self._pipe: Any | None = None

    def _lazy_init(self) -> None:
        if self._pipe is None:
            model_source = (
                self.local_model_dir
                if Path(self.local_model_dir).exists()
                else self.model_name
            )
            common = dict(model=model_source, tokenizer=model_source)
            try:
                self._pipe = pipeline(
                    "text-classification",
                    truncation=True,
                    max_length=self.max_length,
                    **common,
                )
            except TypeError:
                self._pipe = pipeline("text-classification", **common)

    def _mark_entities(self, sentence: str, head: Entity, tail: Entity) -> str:
        s = sentence
        spans = sorted(
            [
                (head.start, head.end, "[E1]", "[/E1]"),
                (tail.start, tail.end, "[E2]", "[/E2]"),
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        for start, end, left, right in spans:
            s = f"{s[:start]}{left} {s[start:end]} {right}{s[end:]}"
        return s

    def _map_label(self, raw: str) -> RelationType | None:
        normalized = raw.upper().replace("_", "-")
        mapping = {
            "USED-FOR": RelationType.USED_FOR,
            "PART-OF": RelationType.PART_OF,
            "FEATURE-OF": RelationType.FEATURE_OF,
            "HYPONYM-OF": RelationType.HYPONYM_OF,
            "CONJUNCTION": RelationType.CONJUNCTION,
            "COMPARE": RelationType.COMPARE,
        }
        return mapping.get(normalized)

    def predict(self, sentence: str, entities: list[Entity]) -> list[Relation]:
        if len(entities) < 2 or not sentence.strip():
            return []

        self._lazy_init()
        assert self._pipe is not None

        relations: list[Relation] = []
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i == j:
                    continue

                marked = self._mark_entities(sentence, head, tail)
                try:
                    prediction = self._pipe(
                        marked,
                        truncation=True,
                        max_length=self.max_length,
                    )[0]
                except TypeError:
                    prediction = self._pipe(marked)[0]
                label = self._map_label(str(prediction.get("label", "NONE")))

                if label is None:
                    continue

                relations.append(
                    Relation(
                        head=head,
                        tail=tail,
                        type=label,
                    )
                )
        return relations
