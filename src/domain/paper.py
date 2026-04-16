from dataclasses import dataclass
from pathlib import Path

from domain.category import Category


@dataclass
class Paper:
    id: str
    path: Path
    categories: list[Category]
