from pydantic import BaseModel

from domain.entity import Entity


class NERPrediction(BaseModel):
    entities: list[Entity]
