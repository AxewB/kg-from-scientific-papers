import math
from collections import Counter

from domain.nlp_result import NLPResult
from pipeline.entity_resolver import EntityResolver


class KGAnalyzer:
    def __init__(self):
        self.resolver = EntityResolver()

    def analyze(self, result: NLPResult) -> dict[str, float]:
        sentences = result.relations

        total_sent = len(sentences)
        total_rel = sum(len(s.relations) for s in sentences)

        entity_ids = []
        relation_types = []

        sentences_with_rel = 0

        for s in sentences:
            if s.relations:
                sentences_with_rel += 1

            for r in s.relations:
                if not r.relation:
                    continue

                subj_id = self.resolver.get_id(r.subject)
                obj_id = self.resolver.get_id(r.target)

                entity_ids.append(subj_id)
                entity_ids.append(obj_id)

                relation_types.append(r.relation)

        total_entity_mentions = len(entity_ids)
        unique_entities = len(set(entity_ids))
        unique_relations = len(set(relation_types))

        # --- метрики ---

        avg_entity_mentions = (
            total_entity_mentions / total_sent if total_sent else 0
        )

        avg_relations = (
            total_rel / total_sent if total_sent else 0
        )

        relation_coverage = (
            sentences_with_rel / total_sent if total_sent else 0
        )

        unique_entity_ratio = (
            unique_entities / total_entity_mentions
            if total_entity_mentions else 0
        )

        # entropy
        rel_counter = Counter(relation_types)
        total = sum(rel_counter.values())

        entropy = 0.0
        if total > 0:
            for c in rel_counter.values():
                p = c / total
                entropy -= p * math.log2(p)

        return {
            # базовые
            "entity_mentions": total_entity_mentions,
            "unique_entities": unique_entities,
            "relations": total_rel,
            "unique_relations": unique_relations,

            # нормализованные
            "avg_entity_mentions_per_sentence": avg_entity_mentions,
            "avg_relations_per_sentence": avg_relations,
            "relation_coverage": relation_coverage,
            "unique_entity_ratio": unique_entity_ratio,

            # сложность
            "relation_entropy": entropy,
        }
