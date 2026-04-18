from pydantic import BaseModel

from domain.relation import RelationTriple


class Sentence(BaseModel):
    text: str
    relations: list[RelationTriple]
