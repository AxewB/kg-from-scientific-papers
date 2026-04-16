from dataclasses import dataclass
from typing import List

from domain.entity import Entity


@dataclass(slots=True)
class SentenceEntities:
    text: str
    entities: List[Entity]
