from dataclasses import dataclass
from pydantic import BaseModel


@dataclass(slots=True)
class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
