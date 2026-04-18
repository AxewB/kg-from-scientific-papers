import logging
from spacy.language import Language
from spacy.tokens import Doc

from domain.entity import Entity
from domain.ir import BlockIR
from domain.sentence_entities import SentenceEntities


lg = logging.getLogger(__name__)

class NERExtractor:
    def __init__(self, nlp: Language):
        self.nlp = nlp

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
