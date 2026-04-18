from dataclasses import dataclass


@dataclass(slots=True)
class RelationTriple:
    subject: str
    subject_label: str
    target: str
    target_label: str
    relation: str | None
    sentence: str
