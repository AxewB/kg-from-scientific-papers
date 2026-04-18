from pydantic import BaseModel


class RelationTriple(BaseModel):
    subject: str
    subject_label: str
    target: str
    target_label: str
    relation: str | None
    sentence: str
