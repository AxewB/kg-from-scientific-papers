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

        entities = {}
        mentions = []
        relations = []

        for entity in result.entities:
            ent_id = self.resolver.get_id(entity.text)
            entities[ent_id] = {"text": entity.text, "type": entity.label.value}
            mentions.append(
                {
                    "paper_id": paper.id,
                    "sentence_id": f"{paper.id}::s{entity.sentence_id}",
                    "entity_id": ent_id,
                }
            )

        for rel in result.relations:
            subj_id = self.resolver.get_id(rel.head.text)
            obj_id = self.resolver.get_id(rel.tail.text)
            relations.append(
                {
                    "subj": subj_id,
                    "obj": obj_id,
                    "rel_type": rel.type.value.replace("-", "_"),
                    "paper_id": paper.id,
                    "sentence_id": f"{paper.id}::s{rel.head.sentence_id}",
                }
            )

        self.db.write_entities(entities)
        self.db.write_mentions(mentions)
        self.db.write_relations(relations)
