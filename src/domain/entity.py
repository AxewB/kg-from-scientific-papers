from dataclasses import dataclass


@dataclass(slots=True)
class Entity:
    text: str
    label: str
    start: int
    end: int
