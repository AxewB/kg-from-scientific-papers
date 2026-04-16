from dataclasses import dataclass
from typing import Literal

NodeType = Literal["Paper", "Category", "Domain", "Entity"]


@dataclass
class KGTriple:
    subject: str
    predicate: str
    object: str

    subject_type: NodeType = "Entity"
    object_type: NodeType = "Entity"

    paper: str | None = None
