import re


class TextCleaner:
    def clear(self, text: str) -> str:
        text = self._remove_spaced_letter_sequences(text)
        text = self._remove_broken_table_lines(text)
        text = self._remove_citations(text)
        text = self._remove_equation_refs(text)
        text = self._remove_figure_refs(text)
        text = self._remove_table_noise(text)
        text = self._remove_misc_refs(text)
        text = self._remove_numeric_dense_lines(text)
        text = self._normalize_punctuation(text)
        text = self._normalize_whitespace(text)

        return text.strip()

    def _remove_spaced_letter_sequences(self, text: str) -> str:
        # удаляет последовательности коротких токенов (1–2 символа), типичных для OCR таблиц
        return re.sub(
            r"(?:\b[a-zA-Z]{1,2}\b[\s\-/]*){6,}",
            " ",
            text,
        )

    def _remove_broken_table_lines(self, text: str) -> str:
        lines = text.split("\n")
        clean_lines = []

        for line in lines:
            tokens = line.split()

            if not tokens:
                continue

            # доля односимвольных токенов
            single_char_ratio = sum(len(t) == 1 for t in tokens) / len(tokens)

            # доля токенов с цифрами
            digit_ratio = sum(any(c.isdigit() for c in t) for t in tokens) / len(tokens)

            # если строка похожа на таблицу — пропускаем
            if single_char_ratio > 0.4:
                continue

            if digit_ratio > 0.4 and len(tokens) > 8:
                continue

            clean_lines.append(line)

        return "\n".join(clean_lines)

    def _remove_citations(self, text: str) -> str:
        """remove bibliography references like [1], [2,3]"""
        return re.sub(r"\[[0-9,\s]+\]", "", text)

    def _remove_equation_refs(self, text: str) -> str:
        """remove references to equations"""
        return re.sub(
            r"\b(Eq|Eqs|Equation)\.?\s*\d+(\s*(and|–|-)\s*\d+)?",
            " ",
            text,
        )

    def _remove_figure_refs(self, text: str) -> str:
        """remove figure references and caption markers"""
        # Fig. 1 | Fig 2 | Figure 3
        text = re.sub(r"\b(?:Fig|Figure)\.?\s*\d+\b", "", text)

        # Fig. | Fig, | Fig.. | Fig ) | Fig : | Fig
        text = re.sub(r"\bFig(?:ure)?\.?(?=[\s,.:;\)])", "", text)

        # Fig: caption start
        text = re.sub(r"\bFig(?:ure)?\s*:\s*", "", text)

        return text

    def _remove_table_noise(self, text: str) -> str:
        """remove table references and dense numeric sequences"""
        text = re.sub(r"\bTable\s*\d+\b", "", text)
        text = re.sub(r"(?:\b\d+/\d+\b\s*){3,}", " ", text)
        text = re.sub(r"(?:\b\d+\.\d+\b\s*){4,}", " ", text)
        return text

    def _remove_misc_refs(self, text: str) -> str:
        """remove section references and standalone numeric markers"""
        text = re.sub(r"\(\d+\)", "", text)
        text = re.sub(r"\bSec\.?\s*\d+\b", "", text)
        text = re.sub(r"\bSection\s*\d+\b", "", text)
        return text

    def _remove_numeric_dense_lines(self, text: str) -> str:
        """remove fragments that likely originate from flattened tables"""
        lines = text.split(". ")
        cleaned = []

        for line in lines:
            digits = len(re.findall(r"\d", line))
            words = len(line.split())

            if words > 6 and digits / max(words, 1) > 0.5:
                continue

            cleaned.append(line)

        return ". ".join(cleaned)

    def _normalize_punctuation(self, text: str) -> str:
        """fix spacing before punctuation"""

        return re.sub(r"\s+([.,;:])", r"\1", text)

    def _normalize_whitespace(self, text: str) -> str:
        """collapse whitespace"""

        return re.sub(r"\s+", " ", text)
