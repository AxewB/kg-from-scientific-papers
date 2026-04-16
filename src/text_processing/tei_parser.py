import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from domain.parsed_paper import ParsedPaper

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


# Text Encoding Initiative Parser (XML standart)
class TEIParser:
    def __init__(self, xml: str):
        self.root = ET.fromstring(xml)
        self._remove_unwanted_tags()
        self._strip_refs()

    def get_title(self) -> Optional[str]:
        el = self.root.find(".//tei:titleStmt/tei:title", NS)
        return el.text.strip() if el is not None and el.text else None

    def get_authors(self) -> List[str]:
        authors = []

        for author in self.root.findall(".//tei:analytic/tei:author", NS):
            first = author.find(".//tei:forename", NS)
            last = author.find(".//tei:surname", NS)

            name_parts = []

            if first is not None and first.text:
                name_parts.append(first.text.strip())

            if last is not None and last.text:
                name_parts.append(last.text.strip())

            if name_parts:
                authors.append(" ".join(name_parts))

        return authors

    def get_abstract(self) -> Optional[str]:
        el = self.root.find(".//tei:profileDesc/tei:abstract", NS)

        if el is None:
            return None

        text = "".join(el.itertext()).strip()
        return text if text else None

    def get_keywords(self) -> List[str]:
        keywords = []

        for term in self.root.findall(".//tei:keywords/tei:term", NS):
            if term.text:
                keywords.append(term.text.strip())

        return keywords

    def get_sections(self) -> List[Dict[str, Optional[str]]]:
        sections = []

        for div in self.root.findall(".//tei:body/tei:div", NS):
            head = div.find("tei:head", NS)

            title = head.text.strip() if head is not None and head.text else None

            paragraphs = [
                "".join(p.itertext()).strip() for p in div.findall("tei:p", NS)
            ]

            text = "\n\n".join(p for p in paragraphs if p)

            if text:
                sections.append({"title": title, "text": text})

        return sections

    def get_full_text(self) -> str:
        paragraphs = []

        for p in self.root.findall(".//tei:body//tei:p", NS):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)

    def parse(self) -> ParsedPaper:
        return ParsedPaper(
            title=self.get_title() or None,
            authors=self.get_authors() or None,
            abstract=self.get_abstract() or None,
            keywords=self.get_keywords() or None,
            sections=self.get_sections() or None,
            full_text=self.get_full_text() or None,
        )

    # --- private methods

    def _strip_refs(self):
        for ref in self.root.findall(".//tei:ref", NS):
            parent = self._get_parent(ref)
            if parent is None:
                continue

            idx = list(parent).index(ref)

            # move text to tail
            if ref.tail:
                if idx > 0:
                    parent[idx - 1].tail = (parent[idx - 1].tail or "") + ref.tail
                else:
                    parent.text = (parent.text or "") + ref.tail

            parent.remove(ref)

    def _remove_unwanted_tags(self):
        remove_tags = [".//tei:formula", ".//tei:figure", ".//tei:table", ".//tei:note"]

        for path in remove_tags:
            for el in self.root.findall(path, NS):
                parent = self._get_parent(el)
                if parent is not None:
                    parent.remove(el)

    def _get_parent(self, child):
        for parent in self.root.iter():
            for c in parent:
                if c is child:
                    return parent
        return None
