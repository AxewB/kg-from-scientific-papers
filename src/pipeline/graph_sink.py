from domain.kg_triple import KGTriple
from pipeline.entity_resolver import EntityResolver


class Neo4jSink:
    def __init__(self, db_writer):
        self.db = db_writer
        self.resolver = EntityResolver()

    def write(self, paper, result):
        triples = []

        paper_id = paper.id

        for sent in result.relations:
            for rel in sent.relations:
                if not rel.relation:
                    continue

                subj = rel.subject.strip()
                obj = rel.target.strip()

                if not subj or not obj:
                    continue

                subj_id = self.resolver.get_id(subj)
                obj_id = self.resolver.get_id(obj)

                triples.append(
                    KGTriple(
                        subject_id=subj_id,
                        object_id=obj_id,
                        predicate=rel.relation,
                        paper_id=paper_id,
                        subject_label=subj,
                        object_label=obj,
                    )
                )

        self.db.write_triples(triples)
