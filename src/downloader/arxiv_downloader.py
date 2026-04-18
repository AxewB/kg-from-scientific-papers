# arxiv lib docs:
# https://lukasschwab.me/arxiv.py/arxiv.html
import logging
import time
from pathlib import Path
from time import sleep
from typing import override
from urllib.error import ContentTooShortError

from arxiv import Client, Result, Search, SortCriterion

from domain import category
from domain.category import Category
from domain.paper import Paper
from downloader.downloader_base import DownloaderBase
from helpers.paths import paths

lg = logging.getLogger(__name__)

class ArxivDownloader(DownloaderBase):
    def __init__(
        self,
        categories: list[str],
        num_each: int = 3,
        download_dir: Path = paths.papers,
        category_registry: dict[str, Category] | None = None,
    ) -> None:
        """
        categories - categories dictionary (https://arxiv.org/category_taxonomy)
        num_each - how much papers of each category should be downloaded
        root_dir - parent dir where all papers should be saved
        """
        super().__init__(root_dir=download_dir)

        if category_registry is None:
            self.category_registry = category.CATEGORY_REGISTRY
        else:
            self.category_registry: dict[str, Category] = category_registry

        self.categories: list[str] = categories
        self.num_each: int = num_each

    @override
    def download(self) -> list[Paper]:
        client = Client()
        papers: list[Paper] = []

        for category_code in self.categories:
            category = self.category_registry[category_code]
            if not category:
                lg.info(
                    f"Category with code {category} doesn't exist in category registry"
                )
                continue

            lg.info(f'Category "{category.code}" ({category.name})')

            search = Search(
                query=category.code,
                max_results=self.num_each,
                sort_by=SortCriterion.SubmittedDate,
            )

            for r in client.results(search):
                paper_id = r.entry_id.split("/")[-1]

                paper_dir = self.root_dir / paper_id
                paper_dir.mkdir(parents=True, exist_ok=True)

                paper_name = f"{paper_id}.pdf"
                pdf_path = paper_dir / paper_name

                if not pdf_path.exists():
                    success = self._download_pdf_with_retry(
                        r,
                        str(paper_dir),
                        paper_name,
                    )

                    if not success:
                        lg.info(f"SKIP {paper_id}: Can't download")
                        continue

                    lg.info(f"Downloaded: {pdf_path}")
                else:
                    lg.info(f"Exists: {paper_id}")

                paper_categories: list[Category] = [
                    self.category_registry[c]
                    for c in r.categories
                    if c in self.category_registry
                ]

                paper = Paper(
                    id=paper_id,
                    path=pdf_path,
                    categories=paper_categories,
                )

                papers.append(paper)

            sleep(3)

        return papers

    def _download_pdf_with_retry(
        self,
        r: Result,
        paper_dir: str,
        paper_name: str,
        retries: int = 3,
    ) -> bool:
        """Returns true if download successful, else False"""
        for attempt in range(retries):
            try:
                _ = r.download_pdf(
                    dirpath=paper_dir,
                    filename=paper_name,
                )
                return True
            except ContentTooShortError:
                if attempt < retries - 1:
                    lg.info(f"[retry {attempt + 1}/{retries}] {paper_name}")
                    time.sleep(2)
                else:
                    return False
            except Exception as e:
                lg.info(f"[error] {paper_name}: {e}")
                return False

        return False

    def _safe_name(self, name: str) -> str:
        return "".join(c for c in name if c.isalnum() or c in "._- ").strip()
