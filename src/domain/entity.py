from pydantic import BaseModel


class Entity(BaseModel):
    """
    Represents a named entity extracted from text.

    This structure keeps both character-level and token-level offsets
    to support different stages of the NLP pipeline:

    - Character offsets (`start_char`, `end_char`) are used for:
        * mapping back to the original text
        * serialization (e.g., JSON)
        * UI highlighting

    - Token offsets (`start_token`, `end_token`) are used for:
        * dependency parsing
        * relation extraction
        * interaction with spaCy `Doc`

    Attributes:
        text: Original entity text as it appears in the document.
        label: Entity type (e.g., PERSON, ORG, GPE).
        start_char: Start position in the original text (in characters).
        end_char: End position in the original text (in characters).
        start_token: Start token index in the spaCy Doc.
        end_token: End token index in the spaCy Doc.
    """

    text: str
    label: str

    start_char: int
    end_char: int

    start_token: int
    end_token: int
