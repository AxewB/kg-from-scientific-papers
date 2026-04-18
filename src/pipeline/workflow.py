import json
import logging
from dataclasses import asdict
from pathlib import Path

from data_extraction.grobid import GrobidClient
from domain.nlp_result import NLPResult
from domain.paper import Paper
from downloader.downloader_base import DownloaderBase
from extractors.extractor_base import ExtractorBase
from helpers.paper_state import PaperState
from pipeline.graph_sink import Neo4jSink
from pipeline.nlp_pipeline import NLPPipeline

lg = logging.getLogger(__name__)


class Workflow:
    def __init__(
        self,
        downloader: DownloaderBase,
        extractor: ExtractorBase,
        grobid: GrobidClient,
        pipeline: NLPPipeline,
        sink: Neo4jSink,
    ):
        self.downloader: DownloaderBase = downloader
        self.extractor: ExtractorBase = extractor
        self.grobid: GrobidClient = grobid
        self.pipeline: NLPPipeline = pipeline
        self.sink: Neo4jSink = sink

    def run(self):
        papers: list[Paper] = self.downloader.download()

        if not self.grobid.is_alive():
            raise RuntimeError("Grobid is offline")

        for paper in papers:
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

        _ = state.tei.write_text(tei_xml, encoding="utf-8")

    def _step_nlp(self, state: PaperState) -> NLPResult:
        lg.info("Doing NLP step...")
        tei_text = state.tei.read_text(encoding="utf-8")

        parsed = self.extractor.extract(tei_text)
        text = parsed.get("full_text")

        if not text:
            raise ValueError("Empty extracted text")

        # either we read or we count
        if state.nlp.exists():
            return self._load_nlp(state)

        lg.info("Starting NLP processing...")
        result = self.pipeline.process(text)

        _ = state.nlp.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return result

    def _step_neo4j(self, paper: Paper, result: NLPResult) -> None:
        lg.info("Doing Neo4j step...")
        self.sink.write(paper, result)

    # --- helpers

    def _load_nlp(self, state: PaperState) -> NLPResult:
        data = json.loads(state.nlp.read_text(encoding="utf-8"))  # pyright: ignore[reportAny]
        return NLPResult.model_validate(data)
