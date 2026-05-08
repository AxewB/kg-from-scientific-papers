import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from helpers.paths import paths
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class PaperMetrics:
    paper_id: str

    stage_time: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)

    sentences: int = 0
    entities: int = 0
    relations: int = 0


class MetricsCollector:
    def __init__(self):
        self.papers: list[PaperMetrics] = []

        self.run_dir = paths.run_dir
        self.raw_path = paths.run_metrics

        paths.figures.mkdir(parents=True, exist_ok=True)

    # lifecycle
    def start_paper(self, paper_id: str) -> PaperMetrics:
        m = PaperMetrics(paper_id=paper_id)
        self.papers.append(m)
        return m

    # timing
    def time_stage(self, paper: PaperMetrics, stage: str) -> Callable[[], None]:
        start = time.perf_counter()

        def stop():
            paper.stage_time[stage] = time.perf_counter() - start

        return stop

    # persistence
    def save_raw(self) -> None:
        with open(self.raw_path, "w", encoding="utf-8") as f:
            for p in self.papers:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    # dataframe
    def _load_df(self) -> pd.DataFrame:
        df = pd.read_json(self.raw_path, lines=True)

        stage_df = pd.json_normalize(df["stage_time"]).add_prefix("time_")
        extra_df = pd.json_normalize(df["extra"]).add_prefix("extra_")

        df = df.drop(columns=["stage_time", "extra"])

        return pd.concat([df, stage_df, extra_df], axis=1)

    # analysis
    def standard_analysis(self) -> dict:
        df = self._load_df()

        summary = df.describe().to_dict()

        # сохранить CSV
        df.to_csv(paths.run_summary, index=False)

        return summary

    # plotting
    def plot_stage_times(self):
        df = self._load_df()

        stage_cols = [c for c in df.columns if c.startswith("time_")]

        df[stage_cols].mean().sort_values().plot(kind="bar")

        plt.title("Average stage time")
        plt.ylabel("seconds")
        plt.tight_layout()
        plt.savefig(paths.figures / "stage_times.png")
        plt.close()

    def plot_distributions(self):
        df = self._load_df()

        df["relations"].plot(kind="hist", bins=30)
        plt.title("Relations distribution")
        plt.savefig(paths.figures / "relations_hist.png")
        plt.close()

        df["entities"].plot(kind="hist", bins=30)
        plt.title("Entities distribution")
        plt.savefig(paths.figures / "entities_hist.png")
        plt.close()

    def plot_complexity(self):
        df = self._load_df()

        plt.scatter(df["sentences"], df["relations"])

        plt.title("Sentences vs Relations")
        plt.xlabel("sentences")
        plt.ylabel("relations")

        plt.savefig(paths.figures / "complexity.png")
        plt.close()

    # evaluation
    def evaluate_entities(self, gold: set[str], pred: set[str]) -> dict[str, float]:
        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)
        return self._prf1(tp, fp, fn)

    def evaluate_relations(self, gold: set[tuple[str, str, str]], pred: set[tuple[str, str, str]]) -> dict[str, float]:
        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)
        return self._prf1(tp, fp, fn)

    def _prf1(self, tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
