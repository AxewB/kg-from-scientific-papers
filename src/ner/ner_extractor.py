import logging

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from domain.entity import Entity
from domain.ir import BlockIR
from domain.sentence_entities import SentenceEntities

lg = logging.getLogger(__name__)

ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"}

class NERExtractor:
    def __init__(self, nlp_model: str = "en_core_web_trf", allowed_labels: set[str] = ALLOWED_LABELS):

        self.nlp: Language = spacy.load(nlp_model)
        self.allowed_labels = allowed_labels

    def _process_doc(self, doc: Doc, block_id: str | None = None) -> SentenceEntities:
        return SentenceEntities(
            text=doc.text,
            entities=[
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    start_token=ent.start,
                    end_token=ent.end,
                )
                for ent in doc.ents
            ],
            source_block_id=block_id,
        )

    def extract_str(self, texts: list[str]) -> list[SentenceEntities]:
        return [
            self._process_doc(doc)
            for doc in self.nlp.pipe(texts, batch_size=64, n_process=1)
        ]

    def extract_blocks(self, blocks: list[BlockIR]) -> list[SentenceEntities]:
        results: list[SentenceEntities] = []

        for block in blocks:
            if block.type != "text" or not block.text:
                continue

            doc = self.nlp(block.text)
            results.append(self._process_doc(doc, block_id=getattr(block, "id", None)))

        return results


    def is_valid_entity(self, text: str) -> bool:
        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ in self.allowed_labels:
                return True

        return False
