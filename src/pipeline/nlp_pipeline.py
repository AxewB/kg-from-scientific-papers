import logging

from tqdm import tqdm

from domain.ir import DocumentIR
from domain.entity import Entity
from domain.nlp_result import NLPResult
from domain.relation import Relation
from pipeline.ner.predictor import SciBERTNER
from pipeline.relation_extraction.predictor import SciBERTRE

lg = logging.getLogger(__name__)


class NLPPipeline:
    def __init__(
        self,
        ner: SciBERTNER,
        relation_extractor: SciBERTRE,
    ):
        self.ner = ner
        self.relation_extractor = relation_extractor

    def process(self, doc: DocumentIR) -> NLPResult:
        lg.info("1. IR traversal + sentence segmentation")
        sentences = self._collect_sentences(doc)
        lg.info("Collected %d sentences for NLP processing", len(sentences))

        lg.info("2. SciBERT NER")
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        for sentence_id, sentence in tqdm(
            enumerate(sentences),
            total=len(sentences),
            desc="NLP sentence processing",
            unit="sent",
        ):
            entities = self.ner.predict(sentence, sentence_id=sentence_id)
            if not entities:
                continue

            all_entities.extend(entities)

            lg.debug("3. SciBERT relation extraction")
            relations = self.relation_extractor.predict(sentence, entities)
            all_relations.extend(relations)

        return NLPResult(entities=all_entities, relations=all_relations)

    def _collect_sentences(self, doc: DocumentIR) -> list[str]:
        sentences: list[str] = []
        for section in doc.sections:
            for block in section.blocks:
                if block.type != "text" or not block.text:
                    continue
                for sentence in block.text.split("."):
                    candidate = sentence.strip()
                    if candidate:
                        sentences.append(candidate + ".")
        return sentences
