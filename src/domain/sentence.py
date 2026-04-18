from dataclasses import dataclass

from domain.relation import RelationTriple


@dataclass(slots=True)
class Sentence:
    text: str
    relations: list[RelationTriple]
