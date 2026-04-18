from pathlib import Path
from typing import override

from domain.parsed_paper import ParsedPaper
from extractors.extractor_base import ExtractorBase
from text_processing.tei_parser import TEIParser


class TEIExtractor(ExtractorBase):
    def __init__(self, root_dir: Path) -> None:
        super().__init__(root_dir)

    @override
    def extract(self, xml: str) -> ParsedPaper:
        parser = TEIParser.from_xml(xml)

        parsed: ParsedPaper = parser.parse()

        return parser.parse()
