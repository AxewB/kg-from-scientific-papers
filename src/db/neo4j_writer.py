from neo4j import GraphDatabase


class Neo4jGraphWriter:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def write_triples(self, triples):
        if not triples:
            return

        payload = [t.__dict__ for t in triples]

        with self.driver.session() as session:
            session.execute_write(self._write_batch, payload)

    @staticmethod
    def _write_batch(tx, triples):
        # print(triples)
        query = """
        UNWIND $triples AS t

        MERGE (a:Entity {id: t.subject_id})
        SET a.label = coalesce(t.subject_label, t.subject_id)

        MERGE (b:Entity {id: t.object_id})
        SET b.label = coalesce(t.object_label, t.object_id)

        MERGE (a)-[r:RELATION {type: t.predicate}]->(b)
        SET r.paper_id = t.paper_id

        RETURN count(*)
        """
        tx.run(query, triples=triples)
