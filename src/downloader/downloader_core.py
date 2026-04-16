from abc import ABC, abstractmethod
from pathlib import Path


class DownloaderBase(ABC):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    @abstractmethod
    def download(self) -> list[Path]:
        """Download files and return list of saved paths"""
