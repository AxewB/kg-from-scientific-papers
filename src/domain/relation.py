from enum import Enum

from pydantic import BaseModel

from domain.entity import Entity


class RelationType(str, Enum):
    USED_FOR = "USED-FOR"
    PART_OF = "PART-OF"
    FEATURE_OF = "FEATURE-OF"
    HYPONYM_OF = "HYPONYM-OF"
    CONJUNCTION = "CONJUNCTION"
    COMPARE = "COMPARE"


class Relation(BaseModel):
    head: Entity
    tail: Entity
    type: RelationType
