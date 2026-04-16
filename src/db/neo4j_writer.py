from neo4j import GraphDatabase


class Neo4jGraphWriter:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def write_triples(self, triples: list[dict]) -> None:
        if not triples:
            return

        with self.driver.session() as session:
            session.execute_write(self._write_batch, triples)

    @staticmethod
    def _write_batch(tx, triples: list[dict]) -> None:
        query = """
        UNWIND $triples AS t
        MERGE (s:Entity {name: t.subject})
        MERGE (o:Entity {name: t.object})
        MERGE (s)-[r:RELATION {type: t.predicate}]->(o)
        """

        tx.run(query, triples=triples)
