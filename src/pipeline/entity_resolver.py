import re
import hashlib
from dataclasses import dataclass, field


@dataclass
class EntityStats:
    canonical: str
    count: int = 0


class EntityResolver:
    """
    Stateful entity resolution layer:
    - normalizes surface forms
    - maps aliases to canonical form
    - provides stable identifiers for KG nodes
    """

    def __init__(self):
        self._alias_map: dict[str, str] = {}
        self._stats: dict[str, EntityStats] = {}

    # -------------------------
    # normalization
    # -------------------------

    def normalize(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    # -------------------------
    # stable id generation
    # -------------------------

    def _make_id(self, canonical: str) -> str:
        """
        Stable deterministic ID for KG nodes.
        Prevents duplication even if labels differ slightly.
        """
        return hashlib.md5(canonical.encode("utf-8")).hexdigest()

    # -------------------------
    # main API
    # -------------------------

    def register(self, entity: str) -> str:
        """
        Returns canonical entity form.
        """
        norm = self.normalize(entity)

        if norm in self._alias_map:
            canonical = self._alias_map[norm]
            self._stats[canonical].count += 1
            return canonical

        canonical = entity.strip()
        self._alias_map[norm] = canonical

        self._stats[canonical] = EntityStats(canonical=canonical, count=1)

        return canonical

    def resolve(self, entity: str) -> str:
        norm = self.normalize(entity)
        return self._alias_map.get(norm, entity.strip())

    def get_id(self, entity: str) -> str:
        canonical = self.resolve(entity)
        return self._make_id(canonical)

    # -------------------------
    # analytics (optional)
    # -------------------------

    def get_stats(self) -> dict[str, EntityStats]:
        return self._stats

    def resolve_id(self, entity: str) -> str:
        canonical = self.resolve(entity)
        return self._make_id(canonical)
