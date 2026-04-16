from dataclasses import dataclass

from domain.entity import Entity


@dataclass(slots=True)
class EntityPair:
    left: Entity
    right: Entity
