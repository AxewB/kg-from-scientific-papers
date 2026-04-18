from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

BlockType: TypeAlias = Literal[
    "text",
    "formula",
    "table",
    "list",
    "figure_caption",
    "reference_block",
]


@dataclass(slots=True)
class SpanIR:
    text: str
    start: int
    end: int
    context_block_type: BlockType
    semantic_hint: str | None = None


@dataclass(slots=True)
class BlockIR:
    type: BlockType
    text: str | None = None
    spans: list[SpanIR] | None = None


@dataclass(slots=True)
class SectionIR:
    title: str | None
    level: int
    blocks: list[BlockIR]


@dataclass(slots=True)
class DocumentMeta:
    title: str | None
    authors: list[str]
    abstract: str | None
    keywords: list[str]


@dataclass(slots=True)
class ReferenceIR:
    id: str
    raw: str
    resolved_text: str | None


FormulaStructureType: TypeAlias = Literal[
    "ratio",
    "equation",
    "inequality",
    "expression",
]


@dataclass(slots=True)
class FormulaIR:
    raw: str
    latex: str | None
    variables: list[str]
    structure_type: FormulaStructureType


@dataclass(slots=True)
class TableIR:
    headers: list[str]
    rows: list[list[str]]
    flattened_spans: list[SpanIR]


@dataclass(slots=True)
class DocumentIR:
    doc_id: str
    meta: DocumentMeta
    sections: list[SectionIR]
    references: list[ReferenceIR]
    formulas: list[FormulaIR]
    raw_text: str | None
