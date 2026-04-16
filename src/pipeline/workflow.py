import json
from dataclasses import asdict

from data_extraction.grobid import GrobidClient
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
        self.downloader = downloader
        self.extractor = extractor
        self.grobid = grobid
        self.pipeline = pipeline
        self.sink = sink

    def run(self):
        papers = self.downloader.download()

        if not self.grobid.is_alive():
            raise RuntimeError("Grobid is offline")

        for pdf_path in papers:
            state = PaperState(pdf_path.parent)

            # 1. GROBID
            if not state.tei().exists():
                tei_xml = self.grobid.process_fulltext(pdf_path)
                state.tei().write_text(tei_xml or "", encoding="utf-8")

            # 2. NLP
            parsed = self.extractor.extract(state.tei().read_text())
            text = parsed["full_text"]

            if not state.nlp().exists():
                if not text:
                    continue

                result = self.pipeline.process(text)

                state.nlp().write_text(
                    json.dumps(asdict(result), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                result = self.pipeline.process(text)

            # 3. Neo4j
            self.sink.write(state.dir.name, result)
