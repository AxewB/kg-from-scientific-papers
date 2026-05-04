import json
import logging
from pathlib import Path
from typing import Any

from data_extraction.grobid import GrobidClient
from domain.ir import BlockIR, DocumentIR, DocumentMeta, SectionIR
from domain.nlp_result import NLPResult
from domain.paper import Paper
from downloader.downloader_base import DownloaderBase
from helpers.paper_state import PaperState
from pipeline.graph_sink import Neo4jSink
from pipeline.nlp_pipeline import NLPPipeline
from text_processing.tei_parser import TEIParser

lg = logging.getLogger(__name__)


class Workflow:
    def __init__(
        self,
        downloader: DownloaderBase,
        grobid: GrobidClient,
        pipeline: NLPPipeline,
        sink: Neo4jSink,
    ):
        self.downloader = downloader
        self.grobid = grobid
        self.pipeline = pipeline
        self.sink = sink

    def run(self):
        papers: list[Paper] = self.downloader.download()

        if not self.grobid.is_alive():
            raise RuntimeError("Grobid is offline")

        total_papers = len(papers)

        for i, paper in enumerate(papers):
            lg.info(f"Processing paper {i + 1} out of {total_papers}")
            try:
                state = PaperState(paper.path.parent)

                self._step_grobid(state, paper.path)
                result = self._step_nlp(state)
                self._step_neo4j(paper, result)

            except Exception as e:
                lg.info(f"[skip paper {paper.id}] {e}")
                continue

    # --- steps

    def _step_grobid(self, state: PaperState, paper_path: Path) -> None:
        lg.info("Doing GROBID step...")

        if state.is_valid_tei():
            return

        tei_xml = self.grobid.process_fulltext(paper_path)

        if not tei_xml or not tei_xml.strip():
            raise ValueError("Empty TEI from GROBID")

        state.tei.write_text(tei_xml, encoding="utf-8")

    def _step_nlp(self, state: PaperState) -> NLPResult:
        lg.info("Doing NLP step...")

        if state.nlp.exists():
            return self._load_nlp(state)

        tei_text = state.tei.read_text(encoding="utf-8")

        parser = TEIParser.from_xml(tei_text)
        parsed = parser.parse()

        doc_ir = self._to_document_ir(parsed, doc_id=state.dir.name)

        lg.info("Starting NLP processing...")
        result = self.pipeline.process(doc_ir)

        state.nlp.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return result

    def _step_neo4j(self, paper: Paper, result: NLPResult) -> None:
        lg.info("Doing Neo4j step...")
        self.sink.write(paper, result)

    # --- helpers

    def _load_nlp(self, state: PaperState) -> NLPResult:
        data = json.loads(state.nlp.read_text(encoding="utf-8"))
        return NLPResult.model_validate(data)

    def _normalize(self, text: str | None) -> str | None:
        if not text:
            return text
        return " ".join(text.split())

    def _to_document_ir(self, parsed: dict[str, Any], doc_id: str) -> DocumentIR:
        sections_data = parsed.get("sections") or []
        sections: list[SectionIR] = []

        for idx, section in enumerate(sections_data):
            text = self._normalize(section.get("text"))

            blocks: list[BlockIR] = []
            if text:
                blocks.append(BlockIR(type="text", text=text))

            sections.append(
                SectionIR(
                    title=self._normalize(section.get("title")),
                    level=idx + 1,
                    blocks=blocks,
                )
            )

        return DocumentIR(
            doc_id=doc_id,
            meta=DocumentMeta(
                title=self._normalize(parsed.get("title")),
                authors=parsed.get("authors") or [],
                abstract=self._normalize(parsed.get("abstract")),
                keywords=parsed.get("keywords") or [],
            ),
            sections=sections,
            references=[],  # пока не используешь
            formulas=[],  # осознанно игнорируешь
            raw_text=self._normalize(parsed.get("full_text")),
        )
