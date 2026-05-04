import logging

from domain.ir import DocumentIR
from domain.nlp_result import NLPResult
from pipeline.entity_normalizer import EntityNormalizer
from pipeline.entity_resolver import EntityResolver
from relation_extraction.relation_extractor import RelationExtractor

lg = logging.getLogger(__name__)


class NLPPipeline:
    def __init__(
        self,
        relation_extractor: RelationExtractor,
    ):
        self.relation_extractor = relation_extractor
        self.normalizer = EntityNormalizer()
        self.resolver = EntityResolver()

    def process(self, doc: DocumentIR) -> NLPResult:
        lg.info("1. IR traversal")

        blocks = [b for s in doc.sections for b in s.blocks]

        lg.info("2. RE (REBEL + NER filter)")
        relations = self.relation_extractor.extract_blocks(blocks)

        lg.info("3. Post-processing (normalization + dedup)")
        relations = self._postprocess_relations(relations)

        return NLPResult(
            entities=[],
            relations=relations,
        )

    def _postprocess_relations(self, relations):
        dedup = set()
        cleaned = []

        for sent in relations:
            new_relations = []

            for r in sent.relations:
                subj = self.resolver.register(r.subject)
                obj = self.resolver.register(r.target)

                key = (subj, r.relation, obj)

                if key in dedup:
                    continue

                dedup.add(key)

                r.subject = subj
                r.target = obj

                new_relations.append(r)

            sent.relations = new_relations
            cleaned.append(sent)

        return cleaned
