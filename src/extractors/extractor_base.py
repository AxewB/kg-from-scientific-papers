from abc import ABC, abstractmethod
from pathlib import Path

from domain.parsed_paper import ParsedPaper


class ExtractorBase(ABC):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    @abstractmethod
    def extract(self, xml: str) -> ParsedPaper:
        """Extract text"""
