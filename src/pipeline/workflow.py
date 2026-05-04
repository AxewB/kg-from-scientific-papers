import json
import logging
from pathlib import Path

from data_extraction.grobid import GrobidClient
from domain.ir import DocumentIR
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
        # papers: list[Paper] = self.downloader.download()
        papers: list[Paper] = self.downloader.get_local_papers()

        if not self.grobid.is_alive():
            raise RuntimeError("Grobid is offline")

        total_papers = len(papers)

        for i, paper in enumerate(papers):
            lg.info(f"Processing paper {i + 1} out of {total_papers}")
            try:
                state = PaperState(paper.path.parent)

                self._step_grobid(state, paper.path)
                doc_ir, result = self._step_nlp(state)
                self._step_neo4j(paper, doc_ir, result)

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

    def _step_nlp(self, state: PaperState) -> tuple[DocumentIR, NLPResult]:
        lg.info("Doing NLP step...")

        tei_text = state.tei.read_text(encoding="utf-8")
        parser = TEIParser.from_xml(tei_text)
        doc_ir = parser.parse(doc_id=state.dir.name)

        if state.nlp.exists():
            result = self._load_nlp(state)
            return doc_ir, result

        lg.info("Starting NLP processing...")
        result = self.pipeline.process(doc_ir)

        state.nlp.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return doc_ir, result

    def _step_neo4j(self, paper: Paper, doc_ir: DocumentIR, result: NLPResult) -> None:
        lg.info("Doing Neo4j step...")
        self.sink.write(paper, doc_ir, result)

    # --- helpers

    def _load_nlp(self, state: PaperState) -> NLPResult:
        data = json.loads(state.nlp.read_text(encoding="utf-8"))
        return NLPResult.model_validate(data)

    def _normalize(self, text: str | None) -> str | None:
        if not text:
            return text
        return " ".join(text.split())
