import logging
from domain.nlp_result import NLPResult
from ner.ner_extractor import NERExtractor
from relation_extraction.relation_extractor import RelationExtractor
from text_processing.cleaner import TextCleaner
from text_processing.sentence_splitter import SentenceSplitter

lg = logging.getLogger(__name__)

class NLPPipeline:
    def __init__(
        self,
        ner: NERExtractor,
        relation_extractor: RelationExtractor,
        cleaner: TextCleaner | None = None,
        splitter: SentenceSplitter | None = None,
    ):
        self.ner: NERExtractor = ner
        self.relation_extractor: RelationExtractor = relation_extractor
        self.cleaner: TextCleaner = cleaner or TextCleaner()
        self.splitter: SentenceSplitter = splitter or SentenceSplitter()

    def process(self, text: str) -> NLPResult:
        """
        text: str - full text from file

        return: NLPResult - formatted and cleaned text
        """
        lg.info("1. NLP: Clearing...")
        clean_text = self.cleaner.clear(text)

        lg.info("2. NLP: Splitting...")
        sentences = self.splitter.split(clean_text)

        lg.info("3. NLP: Extracting...")
        entities = self.ner.extract_batch(sentences)
        relations = self.relation_extractor.extract_batch(sentences)

        return NLPResult(
            clean_text=clean_text,
            sentences=sentences,
            entities=entities,
            relations=relations,
        )
