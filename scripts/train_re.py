import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python train_re.py` without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nlp.re.trainer import RETrainingConfig, SciBERTRETrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SciBERT RE on SciERC.")
    parser.add_argument("--train", default="datasets/scierc/train.json", help="Path to SciERC train jsonl.")
    parser.add_argument("--dev", default="datasets/scierc/dev.json", help="Path to SciERC dev jsonl.")
    parser.add_argument("--model-name", default="allenai/scibert_scivocab_uncased", help="Base model name/path.")
    parser.add_argument("--output-dir", default="artifacts/re", help="Directory for trained checkpoint.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    logging.info("Starting RE training CLI...")
    logging.info("Train: %s | Dev: %s | Output: %s", args.train, args.dev, args.output_dir)
    config = RETrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    trainer = SciBERTRETrainer(config)
    trainer.train(args.train, args.dev)
    logging.info("RE training finished.")


if __name__ == "__main__":
    main()
