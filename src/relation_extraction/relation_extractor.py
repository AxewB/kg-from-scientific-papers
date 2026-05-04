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

    def __init__(self, ner_extractor, model_name="Babelscape/rebel-large"):
        self.ner = ner_extractor
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    # --- core inference

    def _extract_triples(self, text: str) -> list[RelationTriple]:
        input_text = f"extract triplets: {text}"

        inputs = self.tokenizer(
            input_text,
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

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        #print("\n=== REBEL RAW OUTPUT ===")
        #print(decoded)
        #print("========================\n")


        triples = self._parse_rebel_output(decoded, text)

        #print("PARSED:", triples)

        return triples

    # --- REBEL output parsing

    def _parse_rebel_output(self, text: str, original: str):
        triples = []

        #print("\n[DEBUG RAW TEXT]")
        #print(text)

        chunks = text.split("<triplet>")

        #print("\n[DEBUG CHUNKS]")
        # for i, c in enumerate(chunks):
        #     print(f"CHUNK {i}: {repr(c)}")

        for chunk in chunks:
            chunk = chunk.strip()

            #print("\n[DEBUG PROCESS CHUNK]")
            #print("RAW:", repr(chunk))

            if "<subj>" not in chunk or "<obj>" not in chunk:
                #print("SKIP: missing tags")
                continue

            try:
                subj = chunk.split("<subj>")[0].strip()
                rest = chunk.split("<subj>")[1]

                obj = rest.split("<obj>")[0].strip()
                rel = rest.split("<obj>")[1].replace("</s>", "").strip()

                #print("SUBJ:", subj)
                #print("OBJ:", obj)
                #print("REL:", rel)

                if not subj or not obj:
                    #print("SKIP: empty subj/obj")
                    continue

                triples.append(
                    RelationTriple(
                        subject=subj,
                        subject_label=None,
                        target=obj,
                        target_label=None,
                        relation=rel if rel else None,
                        sentence=original,
                    )
                )

            except Exception as e:
                # print("ERROR:", e)
                continue

        #print("\nFINAL TRIPLES:", triples)

        return triples

    # --- block interface

    def extract_blocks(self, blocks: list[BlockIR]) -> list[Sentence]:
        results: list[Sentence] = []

        for block in tqdm(blocks, desc="REBEL processing", unit="block"):
            if block.type != "text" or not block.text:
                continue

            doc = self.ner.nlp(block.text)

            for sent in doc.sents:
                triples = self._extract_triples(sent.text)

                results.append(
                    Sentence(
                        text=sent.text,
                        relations=triples,
                    )
                )

        return results
