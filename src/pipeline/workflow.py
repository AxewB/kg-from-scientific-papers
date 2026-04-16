import json
from dataclasses import asdict
from pathlib import Path

from data_extraction.grobid import GrobidClient
from domain.paper import Paper
from downloader.downloader_base import DownloaderBase
from extractors.extractor_base import ExtractorBase
from helpers.paper_state import PaperState
from pipeline.graph_sink import Neo4jSink
from pipeline.nlp_pipeline import NLPPipeline


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
                state = PaperState(paper.path.parent)  # getting pdf directory

                # 1. GROBID
                if not state.is_valid_tei():
                    tei_xml = self.grobid.process_fulltext(paper.path)

                    if not tei_xml or not tei_xml.strip():
                        continue

                    state.tei.write_text(tei_xml, encoding="utf-8")

                # 2. NLP
                parsed = self.extractor.extract(state.tei.read_text())
                text = parsed["full_text"]

                if not text:
                    continue

                if not state.nlp.exists():
                    result = self.pipeline.process(text)
                    _ = state.nlp.write_text(
                        json.dumps(asdict(result), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                else:
                    result = self.pipeline.process(text)

                # 3. Neo4j
                self.sink.write(paper, result)

            except Exception as e:
                print(f"[skip paper {paper.id}] {e}")
                continue


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
