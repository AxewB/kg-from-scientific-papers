from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import pipeline

from domain.entity import Entity, EntityType


class SciBERTNER:
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        local_model_dir: str = "artifacts/ner",
    ) -> None:
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self._pipe: Any | None = None

    def _lazy_init(self) -> None:
        if self._pipe is None:
            model_source = (
                self.local_model_dir
                if Path(self.local_model_dir).exists()
                else self.model_name
            )
            self._pipe = pipeline(
                "token-classification",
                model=model_source,
                tokenizer=model_source,
                aggregation_strategy="simple",
            )

    def _map_label(self, raw: str) -> EntityType:
        label = raw.replace("B-", "").replace("I-", "").upper()
        mapping = {
            "TASK": EntityType.TASK,
            "METHOD": EntityType.METHOD,
            "MATERIAL": EntityType.MATERIAL,
            "METRIC": EntityType.METRIC,
            "OTHER-SCIENTIFIC-TERM": EntityType.OTHER,
            "GENERIC": EntityType.GENERIC,
        }
        return mapping.get(label, EntityType.GENERIC)

    def predict(self, sentence: str, sentence_id: int = 0) -> list[Entity]:
        if not sentence.strip():
            return []

        self._lazy_init()
        assert self._pipe is not None

        raw_entities = self._pipe(sentence)
        entities: list[Entity] = []

        for item in raw_entities:
            start = int(item.get("start", 0))
            end = int(item.get("end", 0))

            entities.append(
                Entity(
                    text=sentence[start:end],
                    label=self._map_label(str(item.get("entity_group", "GENERIC"))),
                    start=start,
                    end=end,
                    sentence_id=sentence_id,
                )
            )

        return entities
