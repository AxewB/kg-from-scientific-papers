from dataclasses import dataclass
from typing import TypedDict


@dataclass
class ParsedPaper(TypedDict):
    title: str | None
    authors: list[str] | None
    abstract: str | None
    keywords: list[str] | None
    sections: list[dict[str, str | None]] | None
    full_text: str | None
