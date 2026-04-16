from typing import Iterable

from spacy.language import Language

from domain.entity import Entity
from domain.entity_pair import EntityPair
from domain.relation import Relation
from domain.sentence import Sentence


class RelationExtractor:
    def __init__(self, nlp: Language):
        self.nlp = nlp

    # Utils

    def _is_valid_entity(self, text: str) -> bool:
        clean = text.replace(".", "").strip()
        return bool(clean) and not clean.isdigit()

    def _to_entity(self, ent) -> Entity:
        return Entity(
            text=ent.text,
            label=ent.label_,
            start=ent.start,
            end=ent.end,
        )

    # Core pipeline steps

    def _extract_entity_pairs(self, doc) -> Iterable[EntityPair]:
        entities = [self._to_entity(e) for e in doc.ents]

        for i, left in enumerate(entities):
            for right in entities[i + 1 :]:
                yield EntityPair(left, right)

    def _extract_verbs_between(self, doc, pair: EntityPair):
        return [
            tok for tok in doc[pair.left.end : pair.right.start] if tok.pos_ == "VERB"
        ]

    def _build_relation(
        self,
        pair: EntityPair,
        verbs,
        sentence: str,
    ) -> Relation | None:
        if not (
            self._is_valid_entity(pair.left.text)
            and self._is_valid_entity(pair.right.text)
        ):
            return None

        return Relation(
            subject=pair.left.text,
            subject_label=pair.left.label,
            target=pair.right.text,
            target_label=pair.right.label,
            relation=verbs[0].text if verbs else None,
            sentence=sentence,
        )

    # Public API

    def extract(self, text: str) -> Sentence:
        doc = self.nlp(text)
        relations: list[Relation] = []

        for pair in self._extract_entity_pairs(doc):
            verbs = self._extract_verbs_between(doc, pair)
            rel = self._build_relation(pair, verbs, text)
            if rel:
                relations.append(rel)

        return Sentence(text=text, relations=relations)

    def extract_batch(self, texts: list[str]) -> list[Sentence]:
        results: list[Sentence] = []

        for doc in self.nlp.pipe(texts, batch_size=32):
            relations: list[Relation] = []

            for pair in self._extract_entity_pairs(doc):
                verbs = self._extract_verbs_between(doc, pair)
                rel = self._build_relation(pair, verbs, doc.text)
                if rel:
                    relations.append(rel)

            results.append(Sentence(text=doc.text, relations=relations))

        return results
