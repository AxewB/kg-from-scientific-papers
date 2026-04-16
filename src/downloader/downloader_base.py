from abc import ABC, abstractmethod
from pathlib import Path

from domain.paper import Paper


class DownloaderBase(ABC):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir: Path = root_dir

    @abstractmethod
    def download(self) -> list[Paper]:
        """Download files and return list of saved paths"""
