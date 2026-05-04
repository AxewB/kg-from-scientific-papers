import re
from collections import defaultdict


class EntityNormalizer:
    def __init__(self):
        self.alias_map: dict[str, str] = {}

    def normalize_text(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def register(self, entity: str) -> str:
        norm = self.normalize_text(entity)

        if norm in self.alias_map:
            return self.alias_map[norm]

        self.alias_map[norm] = entity.strip()
        return self.alias_map[norm]

    def resolve(self, entity: str) -> str:
        norm = self.normalize_text(entity)
        return self.alias_map.get(norm, entity.strip())
