import logging
from db.neo4j_writer import Neo4jGraphWriter
from domain.ir import DocumentIR
from domain.nlp_result import NLPResult
from domain.paper import Paper
from pipeline.entity_resolver import EntityResolver

lg = logging.getLogger(__name__)


class Neo4jSink:
    def __init__(self, db_writer):
        self.db: Neo4jGraphWriter = db_writer
        self.resolver = EntityResolver()

    def write(self, paper: Paper, doc_ir: DocumentIR, result: NLPResult):
        lg.info("Starting sink")

        self.db.write_paper(paper_id=paper.id, meta=doc_ir.meta)

        self.db.write_sentences(paper.id, result.relations)

        entities = {}
        mentions = []
        relations = []

        for i, sentence in enumerate(result.relations):
            sentence_id = f"{paper.id}::s{i}"

            for rel in sentence.relations:
                if not rel.relation:
                    continue

                subj_raw = rel.subject.strip()
                obj_raw = rel.target.strip()

                if not subj_raw or not obj_raw:
                    continue

                subj_id = self.resolver.get_id(subj_raw)
                obj_id = self.resolver.get_id(obj_raw)

                subj_label = self.resolver.resolve(subj_raw)
                obj_label = self.resolver.resolve(obj_raw)

                # entities
                entities[subj_id] = subj_label
                entities[obj_id] = obj_label

                # mentions
                mentions.append({"sentence_id": sentence_id, "entity_id": subj_id})
                mentions.append({"sentence_id": sentence_id, "entity_id": obj_id})

                # relations
                relations.append(
                    {
                        "subj": subj_id,
                        "obj": obj_id,
                        "rel": rel.relation,
                        "paper_id": paper.id,
                        "sentence_id": sentence_id,
                    }
                )

        self.db.write_entities(entities)
        self.db.write_mentions(mentions)
        self.db.write_relations(relations)
