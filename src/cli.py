"""Command-line interface for the masters-diploma application."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

DEFAULT_CATEGORIES = ("math.SG", "math.SP")


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--train", default="datasets/scierc/train.json", help="Путь к SciERC train (json/jsonl).")
    _ = parser.add_argument("--dev", default="datasets/scierc/dev.json", help="Путь к SciERC dev (json/jsonl).")
    _ = parser.add_argument(
        "--model-name", default="allenai/scibert_scivocab_uncased", help="Базовая Hugging Face модель."
    )
    _ = parser.add_argument("--epochs", type=int, default=3, help="Число эпох.")
    _ = parser.add_argument("--batch-size", type=int, default=8, help="Размер батча.")
    _ = parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    _ = parser.add_argument(
        "--max-length", type=int, default=256, help="Максимальная длина последовательности (токены)."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py", description="NLP-пайплайн для статей arXiv: GROBID → NER/RE → Neo4j."
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    run_p = sub.add_parser("run", help="Скачать/обработать статьи и записать граф в Neo4j (по умолчанию).")
    _ = run_p.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        metavar="CAT",
        help=f"Категории arXiv (по умолчанию: {' '.join(DEFAULT_CATEGORIES)}).",
    )
    _ = run_p.add_argument(
        "--num-each", type=int, default=20, help="Сколько статей скачивать на категорию (--download)."
    )
    _ = run_p.add_argument(
        "--download",
        action="store_true",
        help="Сначала скачать PDF с arXiv; иначе — только локальные из .cache/papers.",
    )
    _ = run_p.add_argument("--papers-dir", type=Path, default=None, help="Каталог с PDF (по умолчанию .cache/papers).")
    _ = run_p.add_argument("--grobid-url", default="http://localhost:8070", help="URL сервиса GROBID.")
    _ = run_p.add_argument("--neo4j-uri", default="neo4j://localhost:7687", help="URI Neo4j.")
    _ = run_p.add_argument("--neo4j-user", default="neo4j", help="Пользователь Neo4j.")
    _ = run_p.add_argument(
        "--neo4j-password", default="", help="Пароль Neo4j (пустой при NEO4J_AUTH=none в docker-compose)."
    )
    _ = run_p.add_argument("--ner-model", default="artifacts/ner", help="Каталог чекпоинта NER.")
    _ = run_p.add_argument("--re-model", default="artifacts/re", help="Каталог чекпоинта RE.")

    ner_p = sub.add_parser("train-ner", aliases=["train_ner"], help="Обучить SciBERT NER на SciERC.")
    _ = ner_p.add_argument("--output-dir", default="artifacts/ner", help="Куда сохранить чекпоинт.")
    _add_training_args(ner_p)

    re_p = sub.add_parser("train-re", aliases=["train_re"], help="Обучить SciBERT RE на SciERC.")
    _ = re_p.add_argument("--output-dir", default="artifacts/re", help="Куда сохранить чекпоинт.")
    _add_training_args(re_p)

    learn_p = sub.add_parser(
        "ner-re-learn", aliases=["ner_re_learn"], help="Последовательно обучить NER и RE на SciERC."
    )
    _ = learn_p.add_argument("--ner-output-dir", default="artifacts/ner", help="Каталог чекпоинта NER.")
    _ = learn_p.add_argument("--re-output-dir", default="artifacts/re", help="Каталог чекпоинта RE.")
    _add_training_args(learn_p)

    metrics_p = sub.add_parser(
        "rebuild-metrics", aliases=["rebuild_metrics"], help="Пересобрать summary.csv и figures/ из metrics.jsonl."
    )
    _ = metrics_p.add_argument("run_dir", type=Path, help="Каталог прогона (напр. .cache/metrics/2026-05-09_14-30-00).")

    return parser


def cmd_run(args: argparse.Namespace) -> None:
    from data_extraction.grobid import GrobidClient
    from db.neo4j_writer import Neo4jGraphWriter
    from downloader.arxiv_downloader import ArxivDownloader
    from helpers import logger
    from helpers.paths import paths
    from nlp.ner.predictor import SciBERTNER
    from nlp.pipeline import NLPPipeline
    from nlp.re.predictor import SciBERTRE
    from pipeline.graph_sink import Neo4jSink
    from pipeline.workflow import Workflow

    logger.init_logger(paths.log_file())
    lg = logging.getLogger(__name__)

    papers_dir = args.papers_dir or paths.papers

    lg.info("Initializing downloader (categories=%s)...", args.categories)
    downloader = ArxivDownloader(categories=args.categories, num_each=args.num_each, download_dir=papers_dir)

    if args.download:
        lg.info("Downloading papers from arXiv...")
        downloader.download()

    lg.info("Loading SciBERT NER (%s) and RE (%s)...", args.ner_model, args.re_model)
    ner = SciBERTNER(local_model_dir=args.ner_model)
    re = SciBERTRE(local_model_dir=args.re_model)

    pipeline = NLPPipeline(ner, re)

    lg.info("Connecting to Neo4j at %s...", args.neo4j_uri)
    db = Neo4jGraphWriter(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
    sink = Neo4jSink(db)

    grobid = GrobidClient(base_url=args.grobid_url)
    workflow = Workflow(downloader=downloader, grobid=grobid, pipeline=pipeline, sink=sink)

    try:
        lg.info("Running workflow...")
        workflow.run()
    finally:
        lg.info("Closing Neo4j connection...")
        db.close()


def cmd_train_ner(args: argparse.Namespace) -> None:
    from nlp.ner.trainer import NERTrainingConfig, SciBERTNERTrainer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    lg = logging.getLogger(__name__)
    lg.info("NER training: train=%s dev=%s → %s", args.train, args.dev, args.output_dir)
    config = NERTrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    SciBERTNERTrainer(config).train(args.train, args.dev)
    lg.info("NER training finished.")


def cmd_train_re(args: argparse.Namespace) -> None:
    from nlp.re.trainer import RETrainingConfig, SciBERTRETrainer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    lg = logging.getLogger(__name__)
    lg.info("RE training: train=%s dev=%s → %s", args.train, args.dev, args.output_dir)
    config = RETrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    SciBERTRETrainer(config).train(args.train, args.dev)
    lg.info("RE training finished.")


def cmd_ner_re_learn(args: argparse.Namespace) -> None:
    ner_args = argparse.Namespace(
        train=args.train,
        dev=args.dev,
        model_name=args.model_name,
        output_dir=args.ner_output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    re_args = argparse.Namespace(
        train=args.train,
        dev=args.dev,
        model_name=args.model_name,
        output_dir=args.re_output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )

    cmd_train_ner(ner_args)
    cmd_train_re(re_args)


def cmd_rebuild_metrics(args: argparse.Namespace) -> None:
    from pipeline.metrics_collector import MetricsCollector

    run_dir = args.run_dir.resolve()
    raw = run_dir / "metrics.jsonl"
    if not raw.is_file():
        print(f"Нет файла: {raw}", file=sys.stderr)
        sys.exit(1)

    m = MetricsCollector.from_existing_run(run_dir)
    m.standard_analysis()
    m.plot_all_metric_figures()
    print(f"Готово: {run_dir / 'summary.csv'}, {run_dir / 'figures'}")


_COMMANDS = frozenset(
    {
        "run",
        "train-ner",
        "train_ner",
        "train-re",
        "train_re",
        "ner-re-learn",
        "ner_re_learn",
        "rebuild-metrics",
        "rebuild_metrics",
    }
)


def _normalize_argv(argv: list[str]) -> list[str]:
    """Без подкоманды считаем, что вызван `run` (python main.py --download …)."""
    if not argv:
        return ["run"]
    if argv in (["-h"], ["--help"]):
        return argv
    if argv[0] in _COMMANDS:
        return argv
    if argv[0].startswith("-"):
        return ["run", *argv]
    return argv


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(_normalize_argv(raw))

    command = args.command or "run"
    if command == "run":
        cmd_run(args)
    elif command in ("train-ner", "train_ner"):
        cmd_train_ner(args)
    elif command in ("train-re", "train_re"):
        cmd_train_re(args)
    elif command in ("ner-re-learn", "ner_re_learn"):
        cmd_ner_re_learn(args)
    elif command in ("rebuild-metrics", "rebuild_metrics"):
        cmd_rebuild_metrics(args)
    else:
        parser.error(f"Неизвестная команда: {command}")


if __name__ == "__main__":
    main()
