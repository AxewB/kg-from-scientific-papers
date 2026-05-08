import json
import logging
from pathlib import Path
import time

from data_extraction.grobid import GrobidClient
from domain.ir import DocumentIR
from domain.nlp_result import NLPResult
from domain.paper import Paper
from downloader.downloader_base import DownloaderBase
from helpers.paper_state import PaperState
from pipeline.graph_sink import Neo4jSink
from pipeline.kg_analyzer import KGAnalyzer
from pipeline.metrics_collector import MetricsCollector, PaperMetrics
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
        self.downloader: DownloaderBase = downloader
        self.grobid: GrobidClient = grobid
        self.pipeline: NLPPipeline = pipeline
        self.sink: Neo4jSink = sink

        self.metrics: MetricsCollector = MetricsCollector()
        self.analyzer: KGAnalyzer = KGAnalyzer()

    def run(self):
        papers: list[Paper] = self.downloader.download()

        if not self.grobid.is_alive():
            raise RuntimeError("Grobid is offline")

        total_papers = len(papers)

        for i, paper in enumerate(papers):
            lg.info(f"Processing paper {i + 1} out of {total_papers}")
            try:
                state = PaperState(paper.path.parent)
                paper_metrics: PaperMetrics = self.metrics.start_paper(paper.id)

                # --- GROBID ---
                stop = self.metrics.time_stage(paper_metrics, "grobid")
                try:
                    self._step_grobid(state, paper.path)
                finally:
                    stop()

                # --- NLP ---
                stop = self.metrics.time_stage(paper_metrics, "nlp_wall")
                try:
                    doc_ir, result, nlp_time, from_cache = self._step_nlp(state)
                finally:
                    stop()

                # сохраняем "реальную" стоимость NLP
                paper_metrics.stage_time["nlp"] = nlp_time
                paper_metrics.extra["nlp_from_cache"] = from_cache
                paper_metrics.extra["nlp_time"] = nlp_time

                # --- ANALYSIS ---
                analysis = self.analyzer.analyze(result)

                paper_metrics.sentences = len(result.relations)
                paper_metrics.relations = analysis["relations"]
                paper_metrics.entities = analysis["entities"]

                # дополнительные метрики
                paper_metrics.extra.update(analysis)

                text = doc_ir.raw_text or ""
                paper_metrics.extra["text_length_chars"] = len(text)
                paper_metrics.extra["text_length_tokens_est"] = len(text.split())

                # --- NEO4J ---
                stop = self.metrics.time_stage(paper_metrics, "neo4j")
                try:
                    self._step_neo4j(paper, doc_ir, result)
                finally:
                    stop()

            except Exception as e:
                lg.info(f"[skip paper {paper.id}] {e}")
                continue

        self.metrics.save_raw()
        self.metrics.standard_analysis()
        self.metrics.plot_stage_times()
        self.metrics.plot_complexity()
        self.metrics.plot_distributions()

    # --- steps

    def _step_grobid(self, state: PaperState, paper_path: Path) -> None:
        lg.info("Doing GROBID step...")

        if state.is_valid_tei():
            return

        tei_xml = self.grobid.process_fulltext(paper_path)

        if not tei_xml or not tei_xml.strip():
            raise ValueError("Empty TEI from GROBID")

        state.tei.write_text(tei_xml, encoding="utf-8")

    def _step_nlp(self, state: PaperState) -> tuple[DocumentIR, NLPResult, float, bool]:
        lg.info("Doing NLP step...")

        tei_text = state.tei.read_text(encoding="utf-8")
        parser = TEIParser.from_xml(tei_text)
        doc_ir = parser.parse(doc_id=state.dir.name)

        # --- CACHE ---
        if state.nlp.exists():
            raw = json.loads(state.nlp.read_text(encoding="utf-8"))

            result = NLPResult.model_validate(raw["data"])
            nlp_time = raw["meta"]["nlp_time"]

            return doc_ir, result, nlp_time, True

        # --- REAL RUN ---
        start = time.perf_counter()

        lg.info("Starting NLP processing...")
        result = self.pipeline.process(doc_ir)

        nlp_time = time.perf_counter() - start

        payload = {
            "meta": {
                "nlp_time": nlp_time,
            },
            "data": result.model_dump(),
        }

        state.nlp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return doc_ir, result, nlp_time, False

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
