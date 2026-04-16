from dataclasses import dataclass

from domain.sentence import Sentence
from domain.sentence_entities import SentenceEntities
from ner.ner_extractor import NERExtractor
from relation_extraction.re import RelationExtractor
from text_processing.cleaner import TextCleaner
from text_processing.sentence_splitter import SentenceSplitter


@dataclass(slots=True)
class NLPResult:
    clean_text: str
    sentences: list[str]
    entities: list[SentenceEntities]
    relations: list[Sentence]


class NLPPipeline:
    def __init__(
        self,
        ner: NERExtractor,
        re: RelationExtractor,
        cleaner: TextCleaner | None = None,
        splitter: SentenceSplitter | None = None,
    ):
        self.ner: NERExtractor = ner
        self.re: RelationExtractor = re
        self.cleaner: TextCleaner = cleaner or TextCleaner()
        self.splitter: SentenceSplitter = splitter or SentenceSplitter()

    def process(self, text: str) -> NLPResult:
        """
        text: str - full text from file

        return: NLPResult - formatted and cleaned text
        """
        # 1 - clean
        clean_text = self.cleaner.clear(text)

        # 2 - split
        sentences = self.splitter.split(clean_text)

        # 3 - extract
        entities = self.ner.extract_batch(sentences)
        relations = self.re.extract_batch(sentences)

        return NLPResult(
            clean_text=clean_text,
            sentences=sentences,
            entities=entities,
            relations=relations,
        )
