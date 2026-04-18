from dataclasses import dataclass
from pydantic import BaseModel

from domain.relation import RelationTriple


@dataclass(slots=True)
class Sentence(BaseModel):
    text: str
    relations: list[RelationTriple]
