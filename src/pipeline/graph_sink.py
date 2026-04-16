from db.neo4j_writer import Neo4jGraphWriter
from pipeline.nlp_pipeline import NLPResult


class Neo4jSink:
    def __init__(self, db_writer: Neo4jGraphWriter):
        self.db = db_writer

    def write(self, paper_name: str, result: NLPResult) -> None:
        triples: list[dict] = []

        for sentence in result.relations:
            for rel in sentence.relations:
                if not rel.relation:
                    continue

                triples.append(
                    {
                        "subject": rel.subject,
                        "predicate": rel.relation,
                        "object": rel.target,
                    }
                )

        self.db.write_triples(triples)
