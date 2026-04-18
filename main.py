import logging
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
    # initiating global logger
    logger.init_logger(paths.log_file())

    # getting logger by app name
    lg = logging.getLogger(__name__)

    lg.info("Initializing downloader...")
    downloader = ArxivDownloader(
        categories=[
            "cs.AI",
            # "cs.AR",
            # "cs.CC",
        ],
        num_each=1,
    )

    nlp_model = "en_core_web_trf"

    lg.info(f"Initializing NLP model {nlp_model}...")
    nlp = spacy.load(nlp_model)

    lg.info("Initializing NER and RE extractors...")
    ner = NERExtractor(nlp)
    re = RelationExtractor(nlp)

    lg.info("Configuring NLP pipeline...")
    pipeline = NLPPipeline(ner, re)

    lg.info("Setting up Neo4j configuration...")
    db = Neo4jGraphWriter(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="",
    )

    lg.info("Creating sink for Neo4j...")
    sink = Neo4jSink(db)

    lg.info("Creating text extractor...")
    extractor = TEIExtractor(root_dir=paths.papers)

    lg.info("Configuring GROBID...")
    grobid = GrobidClient()

    lg.info("Setting up workflow...")
    workflow = Workflow(
        downloader=downloader,
        extractor=extractor,
        grobid=grobid,
        pipeline=pipeline,
        sink=sink,
    )

    lg.info("Running workflow...")
    workflow.run()

    lg.info("Closing Neo4j connection...")
    db.close()


if __name__ == "__main__":
    main()
