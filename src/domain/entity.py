from enum import Enum

from pydantic import BaseModel


class EntityType(str, Enum):
    TASK = "TASK"
    METHOD = "METHOD"
    MATERIAL = "MATERIAL"
    METRIC = "METRIC"
    OTHER = "OTHER-SCIENTIFIC-TERM"
    GENERIC = "GENERIC"


class Entity(BaseModel):
    text: str
    label: EntityType
    start: int
    end: int
    sentence_id: int
