from typing import TypedDict


class ParsedPaper(TypedDict):
    title: str | None
    authors: list[str] | None
    abstract: str | None
    keywords: list[str] | None
    sections: list[dict[str, str | None]] | None
    full_text: str | None

