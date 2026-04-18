from domain.relation import RelationTriple


def build_triples_from_relations(
    relations_by_sentence: list[list[RelationTriple]],
) -> list[RelationTriple]:
    """
    Преобразует список извлечённых отношений в тройки (subject, predicate, object).
    Отбрасывает отношения без глагола.
    """
    triples: list[RelationTriple] = []

    for sentence_relations in relations_by_sentence:
        for rel in sentence_relations:
            verb = rel.relation
            if not verb:
                continue  # пропускаем связи без явного действия

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
