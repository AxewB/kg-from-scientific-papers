from dataclasses import dataclass

from domain.relation import Relation


@dataclass(slots=True)
class Sentence:
    text: str
    relations: list[Relation]
