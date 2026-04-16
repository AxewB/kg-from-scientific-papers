import sys
from pathlib import Path

import spacy

# Allow running the project as `python main.py` without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_extraction.grobid import GrobidClient
from db.neo4j_writer import Neo4jGraphWriter
from downloader.arxiv_downloader import ArxivDownloader
from extractors.tei_extractor import TEIExtractor
from helpers import logger
from helpers.paths import paths
from ner.ner_extractor import NERExtractor
from pipeline.graph_sink import Neo4jSink
from pipeline.nlp_pipeline import NLPPipeline
from pipeline.workflow import Workflow
from relation_extraction.re import RelationExtractor


def main():
    lg = logger.init_logger(paths.log_file())

    print("Initializing downloader...")
    downloader = ArxivDownloader(
        categories=[
            "cs.AI",
            "cs.AR",
            "cs.CC",
        ],
        num_each=3,
    )

    nlp_model = "en_core_web_trf"

    print(f"Initializing NLP model {nlp_model}...")
    nlp = spacy.load(nlp_model)

    print("Initializing NER and RE extractors...")
    ner = NERExtractor(nlp)
    re = RelationExtractor(nlp)

    print("Configuring NLP pipeline...")
    pipeline = NLPPipeline(ner, re)

    print("Setting up Neo4j configuration...")
    db = Neo4jGraphWriter(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="",
    )

    print("Creating sink for Neo4j...")
    sink = Neo4jSink(db)

    print("Creating text extractor...")
    extractor = TEIExtractor(root_dir=paths.papers)

    print("Configuring GROBID...")
    grobid = GrobidClient()

    print("Setting up workflow...")
    workflow = Workflow(
        downloader=downloader,
        extractor=extractor,
        grobid=grobid,
        pipeline=pipeline,
        sink=sink,
    )

    print("Running workflow...")
    workflow.run()

    print("Closing Neo4j connection...")
    db.close()


if __name__ == "__main__":
    main()
