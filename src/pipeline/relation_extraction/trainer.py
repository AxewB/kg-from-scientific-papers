from __future__ import annotations

import logging
import inspect
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

from datasets.relation_dataset_builder import build_relation_dataset

lg = logging.getLogger(__name__)


@dataclass
class RETrainingConfig:
    model_name: str = "allenai/scibert_scivocab_uncased"
    output_dir: str = "artifacts/re"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 256


class REDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, label2id: dict[str, int], max_length: int) -> None:
        self.samples: list[dict[str, list[int] | int]] = []
        for row in tqdm(rows, desc="Building RE samples", unit="pair"):
            enc = tokenizer(
                row["text"],
                truncation=True,
                max_length=max_length,
            )
            self.samples.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": label2id[row["label"]],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int] | int]:
        return self.samples[idx]


class SciBERTRETrainer:
    def __init__(self, config: RETrainingConfig) -> None:
        self.config = config

    def train(self, train_path: str | Path, dev_path: str | Path) -> None:
        lg.info("Loading SciERC RE train/dev splits and building pair dataset...")
        train_rows = build_relation_dataset(train_path, include_none=True)
        dev_rows = build_relation_dataset(dev_path, include_none=True)
        lg.info("Built RE pairs: train=%d dev=%d", len(train_rows), len(dev_rows))

        labels = sorted({row["label"] for row in train_rows + dev_rows})
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: idx for idx, label in id2label.items()}

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        lg.info("Tokenizing RE datasets...")
        train_dataset = REDataset(train_rows, tokenizer, label2id, self.config.max_length)
        dev_dataset = REDataset(dev_rows, tokenizer, label2id, self.config.max_length)
        lg.info("Prepared RE samples: train=%d dev=%d", len(train_dataset), len(dev_dataset))

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def compute_metrics(eval_pred):
            logits, labels_true = eval_pred
            preds = torch.tensor(logits).argmax(dim=-1).tolist()
            labels_true = labels_true.tolist()

            tp = fp = fn = 0
            for pred_id, gold_id in zip(preds, labels_true):
                pred = id2label[pred_id]
                gold = id2label[gold_id]
                if pred == "NONE" and gold == "NONE":
                    continue
                if pred == gold and gold != "NONE":
                    tp += 1
                elif pred != "NONE" and gold == "NONE":
                    fp += 1
                elif pred == "NONE" and gold != "NONE":
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

        args_dict = {
            "output_dir": self.config.output_dir,
            "learning_rate": self.config.learning_rate,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "num_train_epochs": self.config.epochs,
            "save_strategy": "epoch",
            "logging_steps": 10,
            "disable_tqdm": False,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "report_to": [],
        }

        training_sig = inspect.signature(TrainingArguments.__init__)
        if "evaluation_strategy" in training_sig.parameters:
            args_dict["evaluation_strategy"] = "epoch"
        elif "eval_strategy" in training_sig.parameters:
            args_dict["eval_strategy"] = "epoch"

        if "logging_strategy" in training_sig.parameters:
            args_dict["logging_strategy"] = "steps"

        training_args = TrainingArguments(**args_dict)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        lg.info("Starting RE training...")
        trainer.train()
        lg.info("Saving RE checkpoint to %s", self.config.output_dir)
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

    def train_from_default_scierc(self) -> None:
        train_path = Path("datasets/scierc/train.json")
        dev_path = Path("datasets/scierc/dev.json")
        self.train(train_path, dev_path)
