from spacy.language import Language
from spacy.tokens.doc import Doc
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

    def _process_doc(self, doc: Doc, text: str | None = None) -> SentenceEntities:
        entities: list[Entity] = [self._convert(ent) for ent in doc.ents]

        return SentenceEntities(
            text=text or doc.text,
            entities=entities,
        )

    # public

    def extract(self, text: str) -> SentenceEntities:
        doc: Doc = self.nlp(text)
        return self._process_doc(doc, text)

    def extract_batch(self, texts: list[str]) -> list[SentenceEntities]:
        docs = self.nlp.pipe(texts, batch_size=32)

        results: list[SentenceEntities] = [self._process_doc(doc) for doc in docs]

        return results
