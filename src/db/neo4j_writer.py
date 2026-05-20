import logging

from neo4j import GraphDatabase

from domain.ir import DocumentMeta

lg = logging.getLogger(__name__)


class Neo4jGraphWriter:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

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

    def write_entities(self, entities: dict[str, dict[str, str]]):
        lg.info("Writing entities")

        payload = [{"id": k, "text": v["text"], "type": v["type"]} for k, v in entities.items()]

        query = """
        UNWIND $rows AS row
        MERGE (e:Entity {id: row.id})
        SET e.text = row.text,
            e.type = row.type
        """

        with self.driver.session() as s:
            s.run(query, rows=payload)

    def write_mentions(self, mentions):

        lg.info("Writing mentions")
        query = """
        UNWIND $rows AS row
        MERGE (p:Paper {id: row.paper_id})
        MERGE (s:Sentence {id: row.sentence_id})
        MERGE (p)-[:HAS_SENTENCE]->(s)
        MATCH (e:Entity {id: row.entity_id})
        MERGE (s)-[:MENTIONS]->(e)
        """

        with self.driver.session() as s:
            s.run(query, rows=mentions)

    def write_relations(self, relations):
        lg.info("Writing relations")
        with self.driver.session() as s:
            for row in relations:
                rel_type = row["rel_type"]
                query = f"""
                MATCH (e1:Entity {{id: $subj}})
                MATCH (e2:Entity {{id: $obj}})
                MERGE (e1)-[r:{rel_type} {{
                    paper_id: $paper_id,
                    sentence_id: $sentence_id
                }}]->(e2)
                """
                s.run(
                    query,
                    subj=row["subj"],
                    obj=row["obj"],
                    paper_id=row["paper_id"],
                    sentence_id=row["sentence_id"],
                )
