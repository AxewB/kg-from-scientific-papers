from pydantic import BaseModel

from domain.relation import Relation


class REPrediction(BaseModel):
    relations: list[Relation]
