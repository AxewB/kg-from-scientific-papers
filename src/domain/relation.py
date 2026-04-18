from dataclasses import dataclass
from pydantic import BaseModel


@dataclass(slots=True)
class RelationTriple(BaseModel):
    subject: str
    subject_label: str
    target: str
    target_label: str
    relation: str | None
    sentence: str
