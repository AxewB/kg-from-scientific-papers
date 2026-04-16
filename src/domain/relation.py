from dataclasses import dataclass


@dataclass(slots=True)
class Relation:
    subject: str
    subject_label: str
    target: str
    target_label: str
    relation: str | None
    sentence: str
