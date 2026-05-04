import logging
from neo4j import GraphDatabase

from domain.ir import DocumentMeta

lg = logging.getLogger(__name__)

class Neo4jGraphWriter:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    # def write_triples(self, triples):
    #     if not triples:
    #         return
    #
    #     payload = [t.__dict__ for t in triples]
    #
    #     with self.driver.session() as session:
    #         session.execute_write(self._write_batch, payload)

    def write_paper(self, paper_id: str, meta: DocumentMeta):
        lg.info("Writing paper")
        with self.driver.session() as session:
            _ = session.run(
                """
                MERGE (p:Paper {id: $id})
                SET p.title = $title,
                    p.abstract = $abstract,
                    p.authors = $authors,
                    p.keywords = $keywords
                """,
                {
                    "id": paper_id,
                    "title": meta.title,
                    "abstract": meta.abstract,
                    "authors": meta.authors,
                    "keywords": meta.keywords,
                },
            )
    def write_sentences(self, paper_id, sentences):
        lg.info("Writing sentences")

        payload = [
            {"id": f"{paper_id}::s{i}", "text": s.text}
            for i, s in enumerate(sentences)
        ]

        query = """
        UNWIND $rows AS row
        MERGE (s:Sentence {id: row.id})
        SET s.text = row.text

        WITH row, s
        MERGE (p:Paper {id: $paper_id})
        MERGE (p)-[:HAS_SENTENCE]->(s)
        """

        with self.driver.session() as s:
            s.run(query, rows=payload, paper_id=paper_id)

    def write_entities(self, entities: dict[str, str]):
        lg.info("Writing entities")

        payload = [{"id": k, "label": v} for k, v in entities.items()]

        query = """
        UNWIND $rows AS row
        MERGE (e:Entity {id: row.id})
        SET e.label = row.label
        """

        with self.driver.session() as s:
            s.run(query, rows=payload)

    def write_mentions(self, mentions):

        lg.info("Writing mentions")
        query = """
        UNWIND $rows AS row
        MATCH (s:Sentence {id: row.sentence_id})
        MATCH (e:Entity {id: row.entity_id})
        MERGE (s)-[:MENTIONS]->(e)
        """

        with self.driver.session() as s:
            s.run(query, rows=mentions)

    def write_relations(self, relations):

        lg.info("Writing relations")
        query = """
        UNWIND $rows AS row
        MATCH (e1:Entity {id: row.subj})
        MATCH (e2:Entity {id: row.obj})

        MERGE (e1)-[r:RELATION {
            type: row.rel,
            paper_id: row.paper_id,
            sentence_id: row.sentence_id
        }]->(e2)
        """

        with self.driver.session() as s:
            s.run(query, rows=relations)

    # @staticmethod
    # def _write_batch(tx, triples):
    #     # print(triples)
    #     query = """
    #     UNWIND $triples AS t
    #
    #     MERGE (a:Entity {id: t.subject_id})
    #     SET a.label = coalesce(t.subject_label, t.subject_id)
    #
    #     MERGE (b:Entity {id: t.object_id})
    #     SET b.label = coalesce(t.object_label, t.object_id)
    #
    #     MERGE (a)-[r:RELATION {type: t.predicate}]->(b)
    #     SET r.paper_id = t.paper_id
    #
    #     RETURN count(*)
    #     """
    #     tx.run(query, triples=triples)
