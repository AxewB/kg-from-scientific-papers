from dataclasses import dataclass

from pydantic import BaseModel

from domain.entity import Entity


@dataclass(slots=True)
class SentenceEntities(BaseModel):
    text: str
    entities: list[Entity]
