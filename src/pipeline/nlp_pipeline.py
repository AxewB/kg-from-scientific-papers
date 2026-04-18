import logging

from domain.ir import BlockIR, DocumentIR
from domain.nlp_result import NLPResult
from ner.ner_extractor import NERExtractor
from relation_extraction.relation_extractor import RelationExtractor

lg = logging.getLogger(__name__)


class NLPPipeline:
    def __init__(
        self,
        ner: NERExtractor,
        relation_extractor: RelationExtractor,
    ):
        self.ner = ner
        self.relation_extractor = relation_extractor

    def process(self, doc: DocumentIR) -> NLPResult:
        lg.info("1. IR traversal")

        blocks: list[BlockIR] = []
        for section in doc.sections:
            for block in section.blocks:
                blocks.append(block)

        lg.info("2. NER (IR-aware)")
        entities = self.ner.extract_blocks(blocks)

        lg.info("3. RE (IR-aware)")
        relations = self.relation_extractor.extract_blocks(blocks)

        return NLPResult(
            entities=entities,
            relations=relations,
        )
