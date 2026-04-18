import logging
from spacy.language import Language
from spacy.tokens import Doc, Span

from domain.entity import Entity
from domain.sentence_entities import SentenceEntities


lg = logging.getLogger(__name__)

class NERExtractor:
    def __init__(self, nlp: Language):
        self.nlp: Language = nlp

    # private

    def _convert(self, ent: Span) -> Entity:
        return Entity(
            text=ent.text,
            label=ent.label_,
            start_char=ent.start_char,
            end_char=ent.end_char,
            start_token=ent.start,
            end_token=ent.end,
        )

    def _process_doc(self, doc: Doc) -> SentenceEntities:
        return SentenceEntities(
            text=doc.text,
            entities=[self._convert(ent) for ent in doc.ents],
        )

    # public

    def extract(self, text: str) -> SentenceEntities:
        doc = self.nlp(text)
        return self._process_doc(doc)

    def extract_batch(self, texts: list[str]) -> list[SentenceEntities]:
        lg.info("Extracting...")
        return [
            self._process_doc(doc)
            for doc in self.nlp.pipe(texts, batch_size=64, n_process=1)
        ]
