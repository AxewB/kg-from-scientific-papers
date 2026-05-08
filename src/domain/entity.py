from enum import Enum

from pydantic import BaseModel


class EntityType(str, Enum):
    TASK = "TASK"
    METHOD = "METHOD"
    MATERIAL = "MATERIAL"
    METRIC = "METRIC"
    OTHER = "OTHER-SCIENTIFIC-TERM"
    GENERIC = "GENERIC"


def scierc_ner_label_to_bio_suffix(label: str) -> str:
    """
    Map SciERC dataset NER label strings to BIO suffixes that match EntityType.value.

    Training must use the same strings as evaluation gold (Entity.label.value),
    not raw label.upper() (e.g. OtherScientificTerm -> OTHER-SCIENTIFIC-TERM).
    """
    mapping = {
        "Task": EntityType.TASK.value,
        "Method": EntityType.METHOD.value,
        "Material": EntityType.MATERIAL.value,
        "Metric": EntityType.METRIC.value,
        "OtherScientificTerm": EntityType.OTHER.value,
        "Generic": EntityType.GENERIC.value,
    }
    return mapping.get(label, EntityType.GENERIC.value)


class Entity(BaseModel):
    text: str
    label: EntityType
    start: int
    end: int
    sentence_id: int
    # SciERC word indices (inclusive), set when produced from predict_from_tokens
    start_tok: int | None = None
    end_tok: int | None = None
