from pydantic import BaseModel

from domain.entity import Entity


class SentenceEntities(BaseModel):
    text: str
    entities: list[Entity]
