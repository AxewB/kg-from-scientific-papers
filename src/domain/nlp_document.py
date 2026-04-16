from dataclasses import dataclass

from domain.sentence import Sentence
from domain.sentence_entities import SentenceEntities


@dataclass
class NLPDocument:
    sentences: list[str]
    entities: list[SentenceEntities]
    relations: list[Sentence]
