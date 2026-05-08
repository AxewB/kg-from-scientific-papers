import logging
import sys
from pathlib import Path

import spacy

# Allow running the project as `python main.py` without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_extraction.grobid import GrobidClient
from db.neo4j_writer import Neo4jGraphWriter
from downloader.arxiv_downloader import ArxivDownloader
from helpers import logger
from helpers.paths import paths
from ner.ner_extractor import NERExtractor
from pipeline.graph_sink import Neo4jSink
from pipeline.nlp_pipeline import NLPPipeline
from pipeline.workflow import Workflow
from relation_extraction.relation_extractor import RelationExtractor

# Добавить передаваемые параметры
# Добавить проверку параметров


def main():
    logger.init_logger(paths.log_file())  # initiating global logger

    # getting logger by file name
    lg = logging.getLogger(__name__)

    lg.info("Initializing downloader...")
    downloader = ArxivDownloader(
        categories=[
            "cs.AI",
            "cs.AR",
            "cs.CC",
            # "math.SG",
            # "math.SP",
            # "q-fin.CP",
            # "q-fin.EC",
            # "q-fin.GN",
        ],
        num_each=20,
    )

    lg.info("Initializing NER and RE extractors...")
    ner = NERExtractor()
    re = RelationExtractor(ner)

    lg.info("Configuring NLP pipeline...")
    pipeline = NLPPipeline(re)

    lg.info("Setting up Neo4j configuration...")
    db = Neo4jGraphWriter(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="",
    )

    lg.info("Creating sink for Neo4j...")
    sink = Neo4jSink(db)

    lg.info("Configuring GROBID...")
    grobid = GrobidClient()

    lg.info("Setting up workflow...")
    workflow = Workflow(
        downloader=downloader,
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
