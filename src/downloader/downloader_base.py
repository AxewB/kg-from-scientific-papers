import logging
from abc import ABC, abstractmethod
from pathlib import Path

from domain.category import Category
from domain.paper import Paper

lg = logging.getLogger(__name__)


class DownloaderBase(ABC):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir: Path = root_dir

    def get_local_papers(self) -> list[Paper]:
        papers: list[Paper] = []

        if not self.root_dir.exists():
            lg.info(f"Root directory {self.root_dir} doesn't exist")
            return papers

        for paper_dir in self.root_dir.iterdir():
            if not paper_dir.is_dir():
                continue

            paper_id = paper_dir.name
            pdf_path = paper_dir / f"{paper_id}.pdf"

            if pdf_path.exists():
                paper_categories: list[Category] = []

                paper = Paper(
                    id=paper_id,
                    path=pdf_path,
                    categories=paper_categories,
                )
                papers.append(paper)
                lg.info(f"Found local paper: {paper_id}")
            else:
                lg.info(f"No PDF found for {paper_id}")

        lg.info(f"Found {len(papers)} local papers")
        return papers

    @abstractmethod
    def download(self) -> list[Paper]:
        """Download files and return list of saved paths"""
