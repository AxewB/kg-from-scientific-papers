from typing import Dict, List, TypedDict


class ParsedPaper(TypedDict):
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    sections: List[Dict[str, str]]
    full_text: str
