import json
from dataclasses import asdict
from enum import Enum, auto
from pathlib import Path

from data_extraction.grobid import GrobidClient
from domain.paper import Paper
from downloader.downloader_base import DownloaderBase
from extractors.extractor_base import ExtractorBase
from helpers.paper_state import PaperState
from pipeline.graph_sink import Neo4jSink
from pipeline.nlp_pipeline import NLPPipeline, NLPResult


class WorkflowPipe(Enum):
    DOWNLOADER = auto()
    NLP = auto()
    GROBID = auto()
    NEO4J = auto()


class Workflow:
    def __init__(
        self,
        downloader: DownloaderBase,
        extractor: ExtractorBase,
        grobid: GrobidClient,
        pipeline: NLPPipeline,
        sink: Neo4jSink,
        skip_pipes: list[WorkflowPipe] | None = None,
    ):
        self.downloader: DownloaderBase = downloader
        self.extractor: ExtractorBase = extractor
        self.grobid: GrobidClient = grobid
        self.pipeline: NLPPipeline = pipeline
        self.sink: Neo4jSink = sink

        self.skip_pipes: list[WorkflowPipe] | None = skip_pipes

    def run(self):
        papers: list[Paper] = self._step_download()

        for paper in papers:
            try:
                state = PaperState(paper.path.parent)  # getting pdf directory

                # 1. GROBID
                grobid = self._step_grobid(state, paper.path)
                if not grobid:
                    continue

                # 2. NLP
                nlp_result = self._step_nlp(state)
                if not nlp_result:
                    continue

                # 3. Neo4j
                self._step_neo4j(paper, nlp_result)

            except Exception as e:
                print(f"[skip paper {paper.id}] {e}")
                continue

    def _step_download(self) -> list[Paper]:
        if self.skip_pipes and WorkflowPipe.DOWNLOADER in self.skip_pipes:
            print("Skipping `download` pipeline step")
            return []

        return self.downloader.download()

    def _step_grobid(self, state: PaperState, paper_path: Path) -> bool:
        if self.skip_pipes and WorkflowPipe.GROBID in self.skip_pipes:
            print("Skipping `GROBID` pipeline step")
            return True

        if not self.grobid.is_alive():
            raise RuntimeError("Grobid is offline")

        if not state.is_valid_tei():
            tei_xml = self.grobid.process_fulltext(paper_path)

            if not tei_xml or not tei_xml.strip():
                return False

            _ = state.tei.write_text(tei_xml, encoding="utf-8")

        return True

    def _step_nlp(self, state: PaperState) -> NLPResult | None:
        if self.skip_pipes and WorkflowPipe.NLP in self.skip_pipes:
            print("Skipping `GROBID` pipeline step")
            return None  # WARN: need to change return values of disabled features to some standart

        parsed = self.extractor.extract(state.tei.read_text())
        text = parsed["full_text"]

        if not text:
            return None

        if not state.nlp.exists():
            result = self.pipeline.process(text)
            _ = state.nlp.write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            return self.pipeline.process(text)

    def _step_neo4j(self, paper: Paper, result: NLPResult):
        if self.skip_pipes and WorkflowPipe.NEO4J in self.skip_pipes:
            print("Skipping `NEO4J` pipeline step")
            return # WARN: need to change return values of disabled features to some standart

        self.sink.write(paper, result)


def is_valid_tei(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        content = path.read_text(encoding="utf-8")

        if not content.strip():
            return False

        if not content.lstrip().startswith("<?xml"):
            return False

        return True
    except Exception:
        return False
