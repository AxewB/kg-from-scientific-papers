from domain.relation import Relation


def build_triples_from_relations(
    relations_by_sentence: list[list[Relation]],
) -> list[Relation]:
    """
    Преобразует список извлечённых отношений в тройки (subject, predicate, object).
    Отбрасывает отношения без глагола.
    """
    triples: list[Relation] = []

    for sentence_relations in relations_by_sentence:
        for rel in sentence_relations:
            triples.append(rel)
            # triples.append(
            #     {
            #         "subject": rel["entity1"],
            #         "predicate": verb,
            #         "object": rel["entity2"],
            #         "subject_label": rel.get("label1"),
            #         "object_label": rel.get("label2"),
            #         "sentence": rel.get("sentence"),
            #     }
            # )

    return triples
