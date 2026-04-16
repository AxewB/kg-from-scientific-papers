from dataclasses import asdict
from typing import Any

from neo4j import GraphDatabase

from domain.kg_triple import KGTriple


class Neo4jGraphWriter:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def write_triples(self, triples: list[KGTriple]) -> None:
        if not triples:
            return

        payload = [asdict(t) for t in triples]

        with self.driver.session() as session:
            session.execute_write(self._write_batch, payload)

    @staticmethod
    def _write_batch(tx, triples: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $triples AS t

        MERGE (s:Entity {name: t.subject, type: t.subject_type})
        MERGE (o:Entity {name: t.object, type: t.object_type})

        MERGE (s)-[r:RELATION {type: t.predicate}]->(o)

        RETURN count(*)
        """

        tx.run(query, triples=triples)
