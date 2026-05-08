from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


@dataclass
class NERTrainingConfig:
    model_name: str = "allenai/scibert_scivocab_uncased"
    output_dir: str = "artifacts/ner"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 256


def _scierc_to_bio(entities: list[tuple[int, int, str]], tokens: list[str]) -> list[str]:
    tags = ["O"] * len(tokens)
    for start, end, label in sorted(entities, key=lambda x: (x[0], x[1])):
        if start < 0 or end >= len(tokens) or start > end:
            continue
        tags[start] = f"B-{label.upper()}"
        for i in range(start + 1, end + 1):
            tags[i] = f"I-{label.upper()}"
    return tags


class NERDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        label2id: dict[str, int],
        max_length: int,
    ) -> None:
        self.samples: list[dict[str, list[int]]] = []
        for row in rows:
            for sentence_tokens, sentence_ner in zip(row["sentences"], row["ner"]):
                tags = _scierc_to_bio(sentence_ner, sentence_tokens)
                enc = tokenizer(
                    sentence_tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=max_length,
                )

                word_ids = enc.word_ids()
                label_ids: list[int] = []
                prev_word_id = None
                for word_id in word_ids:
                    if word_id is None:
                        label_ids.append(-100)
                        continue
                    token_tag = tags[word_id]
                    if word_id == prev_word_id and token_tag.startswith("B-"):
                        token_tag = "I-" + token_tag[2:]
                    label_ids.append(label2id.get(token_tag, label2id["O"]))
                    prev_word_id = word_id

                self.samples.append(
                    {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "labels": label_ids,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]


class SciBERTNERTrainer:
    def __init__(self, config: NERTrainingConfig) -> None:
        self.config = config

    def train(self, train_path: str | Path, dev_path: str | Path) -> None:
        train_json = [line for line in Path(train_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        dev_json = [line for line in Path(dev_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        import json

        train_records = [json.loads(line) for line in train_json]
        dev_records = [json.loads(line) for line in dev_json]

        label_set = {"O"}
        for row in train_records + dev_records:
            for sent_ner in row["ner"]:
                for _, _, label in sent_ner:
                    label_set.add(f"B-{label.upper()}")
                    label_set.add(f"I-{label.upper()}")

        id2label = {i: label for i, label in enumerate(sorted(label_set))}
        label2id = {label: idx for idx, label in id2label.items()}

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        train_dataset = NERDataset(
            train_records, tokenizer, label2id, self.config.max_length
        )
        dev_dataset = NERDataset(
            dev_records, tokenizer, label2id, self.config.max_length
        )

        model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = torch.tensor(logits).argmax(dim=-1).tolist()
            labels = labels.tolist()

            tp = fp = fn = 0
            for pred_row, label_row in zip(preds, labels):
                for p, l in zip(pred_row, label_row):
                    if l == -100:
                        continue
                    pred_lab = id2label[p]
                    gold_lab = id2label[l]
                    if gold_lab == "O" and pred_lab == "O":
                        continue
                    if gold_lab == pred_lab and gold_lab != "O":
                        tp += 1
                    elif gold_lab == "O" and pred_lab != "O":
                        fp += 1
                    elif gold_lab != "O" and pred_lab == "O":
                        fn += 1
                    else:
                        fp += 1
                        fn += 1

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            return {"precision": precision, "recall": recall, "f1": f1}

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

    def train_from_default_scierc(self) -> None:
        train_path = Path("datasets/scierc/train.json")
        dev_path = Path("datasets/scierc/dev.json")
        self.train(train_path, dev_path)
