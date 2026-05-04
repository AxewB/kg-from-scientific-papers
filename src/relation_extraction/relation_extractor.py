import logging
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from domain.ir import BlockIR
from domain.relation import RelationTriple
from domain.sentence import Sentence

lg = logging.getLogger(__name__)


class RelationExtractor:
    """
    Transformer-based relation extraction using REBEL model.
    """

    BAD_ENTITIES = {
        "extract triplets",
        "extract triplet",
        "triplet",
        "relation",
        "entity",
    }

    def __init__(self, ner_extractor, model_name="Babelscape/rebel-large"):
        self.ner = ner_extractor
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # --------------------
    # inference
    # --------------------

    def _extract_triples(self, text: str) -> list[RelationTriple]:
        # ВАЖНО: убран instruction prompt
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=3,
            length_penalty=0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=False
        )

        return self._parse_rebel_output(decoded, text)

    # --------------------
    # parsing
    # --------------------

    def _is_valid(self, x: str | None) -> bool:
        if not x:
            return False

        x = x.strip().lower()

        if x in self.BAD_ENTITIES:
            return False

        if len(x) < 2:
            return False

        return True

    def _parse_rebel_output(self, text: str, original: str):
        triples = []

        chunks = text.split("<triplet>")

        for chunk in chunks:
            chunk = chunk.strip()

            if "<subj>" not in chunk or "<obj>" not in chunk:
                continue

            try:
                subj = chunk.split("<subj>")[0].strip()
                rest = chunk.split("<subj>")[1]

                obj = rest.split("<obj>")[0].strip()
                rel = rest.split("<obj>")[1].replace("</s>", "").strip()

                # -----------------------
                # CLEANING / FILTERING
                # -----------------------
                if not self._is_valid(subj) or not self._is_valid(obj):
                    continue

                # # NER FILTER (обязательно)
                # if not self.ner.is_valid_entity(subj):
                #     continue

                # if not self.ner.is_valid_entity(obj):
                #     continue

                if not rel:
                    continue

                triples.append(
                    RelationTriple(
                        subject=subj,
                        subject_label=None,
                        target=obj,
                        target_label=None,
                        relation=rel,
                        sentence=original,
                    )
                )

            except Exception:
                continue

        return triples

    # --------------------
    # block interface
    # --------------------

    def extract_blocks(self, blocks: list[BlockIR]) -> list[Sentence]:
        results: list[Sentence] = []

        for block in tqdm(blocks, desc="REBEL processing", unit="block"):
            if block.type != "text" or not block.text:
                continue

            doc = self.ner.nlp(block.text)

            for sent in doc.sents:
                triples = self._extract_triples(sent.text)

                # защита от пустых предложений
                if not triples:
                    continue

                results.append(
                    Sentence(
                        text=sent.text,
                        relations=triples,
                    )
                )

        return results
