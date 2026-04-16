from dataclasses import dataclass

from domain.entity import Entity


@dataclass(slots=True)
class SentenceEntities:
    text: str
    entities: list[Entity]
