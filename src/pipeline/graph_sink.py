import logging
from db.neo4j_writer import Neo4jGraphWriter
from domain.kg_triple import KGTriple
from domain.paper import Paper
from pipeline.nlp_pipeline import NLPResult

lg = logging.getLogger(__name__)

class Neo4jSink:
    def __init__(self, db_writer: Neo4jGraphWriter):
        self.db: Neo4jGraphWriter = db_writer

    def write(self, paper: Paper, result: NLPResult) -> None:
        triples: list[KGTriple] = []

        # 1. paper -> category -> domain
        for category in paper.categories:
            triples.append(
                KGTriple(
                    subject=paper.id,
                    predicate="HAS_CATEGORY",
                    object=category.code,
                    subject_type="Paper",
                    object_type="Category",
                    paper=paper.id,
                )
            )

            triples.append(
                KGTriple(
                    subject=category.code,
                    predicate="PART_OF",
                    object=category.parent.value,
                    subject_type="Category",
                    object_type="Domain",
                )
            )

        lg.warning(f"Relations type: {type(result.relations[0])}")

        # 2. NLP relations
        for sentence in result.relations:
            for rel in sentence.relations:
                if not rel.relation:
                    continue

                triples.append(
                    KGTriple(
                        subject=rel.subject,
                        predicate=rel.relation,
                        object=rel.target,
                        subject_type="Entity",
                        object_type="Entity",
                        paper=paper.id,
                    )
                )

        self.db.write_triples(triples)
