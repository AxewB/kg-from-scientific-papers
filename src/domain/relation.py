from pydantic import BaseModel


class RelationTriple(BaseModel):
    subject: str
    subject_label: str | None
    target: str
    target_label: str | None
    relation: str | None
    sentence: str
