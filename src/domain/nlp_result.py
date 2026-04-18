from pydantic import BaseModel

from domain.sentence import Sentence
from domain.sentence_entities import SentenceEntities


class NLPResult(BaseModel):
    clean_text: str
    sentences: list[str]
    entities: list[SentenceEntities]
    relations: list[Sentence]
