import xml.etree.ElementTree as ET

from domain.ir import BlockIR, DocumentIR, DocumentMeta, SectionIR

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


# Text Encoding Initiative Parser (XML standart)
class TEIParser:
    """
    Parser for TEI standard of XML documents (e.g. GROBID returns such xml)
    """

    def __init__(self, root: ET.Element, *, _from_factory: bool = False):
        """
        Init method is not meant to be used outside TEIParser class.

        Use TEIParser.from_xml() instead of direct constructor
        """
        if not _from_factory:
            raise RuntimeError("Use TEIParser.from_xml() instead of direct constructor")

        self.root: ET.Element = root

    @classmethod
    def from_xml(cls, xml: str) -> "TEIParser":
        root = ET.fromstring(xml)

        parser = cls(root, _from_factory=True)

        # preprocessing
        parser._remove_unwanted_tags()
        parser._strip_refs()

        return parser

    def get_title(self) -> str | None:
        el = self.root.find(".//tei:titleStmt/tei:title", NS)
        return el.text.strip() if el is not None and el.text else None

    def get_authors(self) -> list[str]:
        authors: list[str] = []

        for author in self.root.findall(".//tei:analytic/tei:author", NS):
            first = author.find(".//tei:forename", NS)
            last = author.find(".//tei:surname", NS)

            name_parts: list[str] = []

            if first is not None and first.text:
                name_parts.append(first.text.strip())

            if last is not None and last.text:
                name_parts.append(last.text.strip())

            if name_parts:
                authors.append(" ".join(name_parts))

        return authors

    def get_abstract(self) -> str | None:
        el = self.root.find(".//tei:profileDesc/tei:abstract", NS)

        if el is None:
            return None

        text = "".join(el.itertext()).strip()
        return text if text else None

    def get_keywords(self) -> list[str]:
        keywords: list[str] = []

        for term in self.root.findall(".//tei:keywords/tei:term", NS):
            if term.text:
                keywords.append(term.text.strip())

        return keywords

    def get_sections(self) -> list[dict[str, str | None]]:
        sections: list[dict[str, str | None]] = []

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
        paragraphs: list[str] = []

        for p in self.root.findall(".//tei:body//tei:p", NS):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)

    def parse(self, doc_id: str) -> DocumentIR:
        return DocumentIR(
            doc_id=doc_id,
            meta=DocumentMeta(
                title=self.get_title(),
                authors=self.get_authors() or [],
                abstract=self.get_abstract(),
                keywords=self.get_keywords() or [],
            ),
            sections=self._build_sections(),
            references=[],
            formulas=[],
            raw_text=self.get_full_text(),
        )

    # private

    def _build_sections(self) -> list[SectionIR]:
        sections = []

        for idx, div in enumerate(self.root.findall(".//tei:body/tei:div", NS)):
            head = div.find("tei:head", NS)
            title = head.text.strip() if head is not None and head.text else None

            paragraphs = [
                "".join(p.itertext()).strip() for p in div.findall("tei:p", NS)
            ]

            text = "\n\n".join(p for p in paragraphs if p)

            blocks = []
            if text:
                blocks.append(BlockIR(type="text", text=text))

            sections.append(
                SectionIR(
                    title=title,
                    level=idx + 1,
                    blocks=blocks,
                )
            )

        return sections

    def _strip_refs(self):
        for ref in self.root.findall(".//tei:ref", NS):
            parent: ET.Element[str] | None = self._get_parent(ref)
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

    def _get_parent(self, child: ET.Element):
        for parent in self.root.iter():
            for c in parent:
                if c is child:
                    return parent
        return None
