from pathlib import Path
from domain.parsed_paper import ParsedPaper
from text_processing.tei_parser import TEIParser
from extractors.extractor_base import ExtractorBase


class TEIExtractor(ExtractorBase):
    def __init__(self, root_dir: Path) -> None:
        super().__init__(root_dir)

    def extract(self, xml: str) -> ParsedPaper:
        parser = TEIParser(xml)
        parsed = parser.parse()

        return parsed
