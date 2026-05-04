from dataclasses import dataclass


@dataclass
class KGTriple:
    subject_id: str
    object_id: str
    predicate: str
    paper_id: str | None
    subject_label: str | None = None
    object_label: str | None = None
