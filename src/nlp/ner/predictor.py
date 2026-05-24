from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from domain.entity import Entity, EntityType


def _token_span_to_char_span(tokens: list[str], start_tok: int, end_tok: int) -> tuple[int, int]:
    """Match SciERC / training: character slice ``sentence[start:end]`` for joined sentence."""
    start = sum(len(t) + 1 for t in tokens[:start_tok])
    end = sum(len(t) + 1 for t in tokens[: end_tok + 1]) - 1
    return start, end


def _id_to_label(model: torch.nn.Module, tid: int) -> str:
    m = getattr(model.config, "id2label", None)
    if not m:
        return "O"
    if tid in m:
        return str(m[tid])
    ts = str(tid)
    if ts in m:
        return str(m[ts])
    return "O"


def _labels_word_level(
    model: torch.nn.Module,
    pred_ids: list[int],
    word_ids: list[int | None],
    num_words: int,
) -> list[str]:
    word_to_first: dict[int, int] = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in word_to_first:
            word_to_first[wid] = i

    out: list[str] = []
    for w in range(num_words):
        if w not in word_to_first:
            out.append("O")
            continue
        tid = pred_ids[word_to_first[w]]
        out.append(_id_to_label(model, tid))
    return out


def _bio_word_labels_to_spans(word_labels: list[str]) -> list[tuple[int, int, str]]:
    """Word-inclusive spans (start_tok, end_tok, type_suffix e.g. TASK)."""
    n = len(word_labels)
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        lab = word_labels[i]
        if lab == "O" or lab is None:
            i += 1
            continue
        if lab.startswith("B-"):
            typ = lab[2:]
            start = i
            i += 1
            while i < n and word_labels[i] == f"I-{typ}":
                i += 1
            spans.append((start, i - 1, typ))
        elif lab.startswith("I-"):
            typ = lab[2:]
            start = i
            i += 1
            while i < n and word_labels[i] == f"I-{typ}":
                i += 1
            spans.append((start, i - 1, typ))
        else:
            i += 1
    return spans


class SciBERTNER:
    """
    NER inference aligned with training: word pieces from ``tokenizer(tokens, is_split_into_words=True)``,
    word labels from first subword of each token, then BIO -> spans. Avoids HF Pipeline aggregation drift.
    """

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        local_model_dir: str = "artifacts/ner",
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._device: torch.device | None = None

    def _model_source(self) -> str:
        return (
            self.local_model_dir
            if Path(self.local_model_dir).exists()
            else self.model_name
        )

    def was_truncated(self, tokens: list[str]) -> bool:
        """True if ``len(tokens)`` exceeds words covered after ``max_length`` subword truncation."""
        if not tokens:
            return False
        self._lazy_init()
        assert self._tokenizer is not None
        enc = self._tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        wids = enc.word_ids(batch_index=0)
        covered = max((w for w in wids if w is not None), default=-1) + 1
        return covered < len(tokens)

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        src = self._model_source()
        self._tokenizer = AutoTokenizer.from_pretrained(src)
        self._model = AutoModelForTokenClassification.from_pretrained(src)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def _map_type_suffix(self, suffix: str) -> EntityType:
        s = suffix.strip().upper().replace("_", "-")
        aliases = {"OTHERSCIENTIFICTERM": "OTHER-SCIENTIFIC-TERM"}
        s = aliases.get(s, s)
        mapping = {
            "TASK": EntityType.TASK,
            "METHOD": EntityType.METHOD,
            "MATERIAL": EntityType.MATERIAL,
            "METRIC": EntityType.METRIC,
            "OTHER-SCIENTIFIC-TERM": EntityType.OTHER,
            "GENERIC": EntityType.GENERIC,
        }
        return mapping.get(s, EntityType.GENERIC)

    @torch.inference_mode()
    def predict_from_tokens(self, tokens: list[str], sentence_id: int = 0) -> list[Entity]:
        if not tokens:
            return []

        self._lazy_init()
        assert self._tokenizer is not None and self._model is not None and self._device is not None

        enc = self._tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        word_ids = enc.word_ids(batch_index=0)
        model_inputs = {k: v.to(self._device) for k, v in enc.items() if torch.is_tensor(v)}
        logits = self._model(**model_inputs).logits[0]
        pred_ids = logits.argmax(-1).tolist()
        word_labels = _labels_word_level(self._model, pred_ids, word_ids, len(tokens))
        spans = _bio_word_labels_to_spans(word_labels)

        sentence = " ".join(tokens)
        entities: list[Entity] = []
        for st, en, typ in spans:
            cs, ce = _token_span_to_char_span(tokens, st, en)
            entities.append(
                Entity(
                    text=sentence[cs:ce],
                    label=self._map_type_suffix(typ),
                    start=cs,
                    end=ce,
                    sentence_id=sentence_id,
                    start_tok=st,
                    end_tok=en,
                )
            )
        return entities

    def predict(self, sentence: str, sentence_id: int = 0) -> list[Entity]:
        """Best-effort for arbitrary text: whitespace tokenization (not identical to SciERC tokens)."""
        if not sentence.strip():
            return []
        return self.predict_from_tokens(sentence.split(), sentence_id=sentence_id)
