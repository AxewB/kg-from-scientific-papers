from pydantic import BaseModel

from domain.sentence import Sentence
from domain.sentence_entities import SentenceEntities


class NLPResult(BaseModel):
    entities: list[SentenceEntities]
    relations: list[Sentence]
