from spacy.language import Language
from spacy.tokens.span import Span

from domain.entity import Entity
from domain.sentence_entities import SentenceEntities


class NERExtractor:
    def __init__(self, nlp: Language):
        self.nlp: Language = nlp

    # private

    def _convert(self, ent: Span) -> Entity:
        return Entity(
            text=ent.text,
            label=ent.label_,
            start=ent.start_char,
            end=ent.end_char,
        )

    # public

    def extract(self, text: str) -> SentenceEntities:
        doc = self.nlp(text)

        entities = [self._convert(ent) for ent in doc.ents]

        return SentenceEntities(
            text=text,
            entities=entities,
        )

    def extract_batch(self, texts: list[str]) -> list[SentenceEntities]:
        results: list[SentenceEntities] = []

        for doc in self.nlp.pipe(texts, batch_size=32):
            entities = [self._convert(ent) for ent in doc.ents]

            results.append(
                SentenceEntities(
                    text=doc.text,
                    entities=entities,
                )
            )

        return results
