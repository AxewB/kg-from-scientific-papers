from collections.abc import Iterable

from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from domain.entity import Entity
from domain.relation import RelationTriple
from domain.sentence import Sentence


class RelationExtractor:
    MAX_TOKEN_DISTANCE: int = 40

    def __init__(self, nlp: Language):
        self.nlp: Language = nlp

    # validation

    def _is_valid_entity(self, text: str) -> bool:
        text = text.strip()
        if len(text) < 2:
            return False
        if text.isdigit():
            return False
        return True

    # entity conversion

    def _to_entity(self, ent: Span) -> Entity:
        return Entity(
            text=ent.text,
            label=ent.label_,
            start_char=ent.start_char,
            end_char=ent.end_char,
            start_token=ent.start,
            end_token=ent.end,
        )

    # core logic

    def _entity_pairs(self, doc: Doc) -> Iterable[tuple[Entity, Entity, Span]]:
        for sent in doc.sents:
            entities = [self._to_entity(e) for e in sent.ents]

            for i, left in enumerate(entities):
                for right in entities[i + 1 :]:
                    if (right.start_token - left.end_token) > self.MAX_TOKEN_DISTANCE:
                        continue
                    yield left, right, sent

    def _find_relation_verb(
        self, doc: Doc, left: Entity, right: Entity
    ) -> Token | None:
        left_tok = doc[left.start_token]
        right_tok = doc[right.start_token]

        # dependency-based heuristic:
        # ищем общий предок
        ancestors: set[Token] = set()
        cur = left_tok

        while cur != cur.head:
            ancestors.add(cur.head)
            cur = cur.head

        cur = right_tok
        while cur != cur.head:
            if cur.head in ancestors and cur.head.pos_ == "VERB":
                return cur.head
            cur = cur.head

        return None

    def _build_relation(
        self,
        left: Entity,
        right: Entity,
        verb: Token | None,
        sentence: str,
    ) -> RelationTriple | None:

        if not (self._is_valid_entity(left.text) and self._is_valid_entity(right.text)):
            return None

        return RelationTriple(
            subject=left.text,
            subject_label=left.label,
            target=right.text,
            target_label=right.label,
            relation=verb.text if verb else None,
            sentence=sentence,
        )

    def _process_doc(self, doc: Doc) -> Sentence:
        relations: list[RelationTriple] = []

        for left, right, sent in self._entity_pairs(doc):
            verb = self._find_relation_verb(doc, left, right)
            rel = self._build_relation(left, right, verb, sent.text)
            if rel:
                relations.append(rel)

        return Sentence(text=doc.text, relations=relations)

    # public API

    def extract(self, text: str) -> Sentence:
        doc = self.nlp(text)
        return self._process_doc(doc)

    def extract_batch(self, texts: list[str]) -> list[Sentence]:
        return [
            self._process_doc(doc)
            for doc in self.nlp.pipe(texts, batch_size=64, n_process=2)
        ]
