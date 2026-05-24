from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from helpers.paths import paths

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency fallback
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency fallback
    torch = None


lg = logging.getLogger(__name__)

_PLOT_FONT_SIZE = 32
_PLOT_TITLE_SIZE = 36
_PLOT_TICK_SIZE = 20
_PLOT_LEGEND_SIZE = 24
_FIG_SCALE = 1.5
_PLOT_DPI = 200


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": _PLOT_FONT_SIZE,
            "axes.titlesize": _PLOT_TITLE_SIZE,
            "axes.labelsize": _PLOT_FONT_SIZE,
            "xtick.labelsize": _PLOT_TICK_SIZE,
            "ytick.labelsize": _PLOT_TICK_SIZE,
            "legend.fontsize": _PLOT_LEGEND_SIZE,
            "figure.titlesize": _PLOT_TITLE_SIZE,
        }
    )


def _figsize(width: float, height: float) -> tuple[float, float]:
    return width * _FIG_SCALE, height * _FIG_SCALE


def _plain_value_axis(ax, axis: str = "y") -> None:
    """Show full numeric tick labels (no +1e3 offset when values cluster)."""
    axis_obj = getattr(ax, f"{axis}axis")
    fmt = ticker.ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    axis_obj.set_major_formatter(fmt)


def _normalize_cpu_pct(raw_pct: float) -> float:
    """
    Map psutil raw CPU (sum over cores, can be >100%) to share of the host.

    ``Process.cpu_percent`` reports utilization as if each logical CPU were 100%;
    dividing by ``cpu_count(logical=True)`` yields 0–100% of total machine capacity.
    """
    if psutil is None:
        return min(100.0, max(0.0, raw_pct))
    n_cpus = psutil.cpu_count(logical=True) or 1
    return min(100.0, max(0.0, raw_pct / n_cpus))


def _cpu_series_host_pct(series: pd.Series) -> pd.Series:
    """
    CPU for plots as 0–100% of the machine.

    Older metrics.jsonl stores per-core sums (>100); newer runs are already normalized.
    """
    s = series.astype(float)
    if float(s.max()) > 100.0:
        return s.map(_normalize_cpu_pct)
    return s


_nvml_ready = False
_nvml_handle = None


def _sample_gpu(device_index: int = 0) -> dict[str, float] | None:
    """
    GPU utilization (%) and total VRAM used on the device (MiB, all processes).

    Uses NVML (package ``nvidia-ml-py`` / ``pynvml``) when available, else ``nvidia-smi``.
    """
    global _nvml_ready, _nvml_handle

    try:
        import pynvml

        if not _nvml_ready:
            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            _nvml_ready = True
        util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        return {
            "gpu_util_pct": float(util.gpu),
            "gpu_mem_used_mb": mem.used / (1024 * 1024),
            "gpu_mem_total_mb": mem.total / (1024 * 1024),
        }
    except Exception:
        pass

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        parts = [p.strip() for p in proc.stdout.strip().split(",")]
        if len(parts) < 3:
            return None
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_mem_used_mb": float(parts[1]),
            "gpu_mem_total_mb": float(parts[2]),
        }
    except Exception:
        return None


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
        _apply_plot_style()
        self.papers: list[PaperMetrics] = []

        self.run_dir: Path = paths.run_dir
        self.raw_path: Path = paths.run_metrics
        self.run_summary: Path = paths.run_summary
        self.figures_dir: Path = paths.figures
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_existing_run(cls, run_dir: str | Path) -> MetricsCollector:
        """Rebuild summary CSV and figures from a saved `metrics.jsonl` folder."""
        run = Path(run_dir).resolve()
        fig = run / "figures"
        fig.mkdir(parents=True, exist_ok=True)
        _apply_plot_style()
        inst = cls.__new__(cls)
        inst.papers = []
        inst.run_dir = run
        inst.raw_path = run / "metrics.jsonl"
        inst.run_summary = run / "summary.csv"
        inst.figures_dir = fig
        return inst

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

    def monitor_stage_resources(self, paper: PaperMetrics, stage: str, interval_sec: float = 0.2) -> Callable[[], None]:
        """
        Track resource usage for a stage into `paper.extra`.

        Local process (default):
        - RAM: RSS of the Python workflow process (psutil), not system-wide.
        - CPU: ``psutil.Process.cpu_percent`` normalized by logical CPU count
          (0–100% of host; idle I/O stages stay near 0%).
        - VRAM (``vram_mb_*``): PyTorch CUDA allocator on the default GPU only.
        - GPU (``gpu_util_pct_*``, ``gpu_mem_mb_*``): device-wide via NVML /
          ``nvidia-smi`` (utilization % and memory used by all processes).

        """

        if psutil is None:
            # psutil isn't available; keep API stable with no-op.
            return lambda: None

        process = psutil.Process()
        stop_event = threading.Event()
        cpu_samples: list[float] = []
        gpu_util_samples: list[float] = []
        gpu_mem_samples: list[float] = []

        ram_start_mb = process.memory_info().rss / (1024 * 1024)
        ram_peak_mb = ram_start_mb

        cuda_available = bool(torch is not None and torch.cuda.is_available())
        gpu_available = _sample_gpu() is not None
        vram_start_mb = 0.0
        vram_peak_mb = 0.0

        if cuda_available:
            vram_start_mb = float(torch.cuda.memory_allocated() / (1024 * 1024))
            torch.cuda.reset_peak_memory_stats()

        # First call initializes psutil internal counter baseline.
        process.cpu_percent(None)

        def _sample() -> None:
            nonlocal ram_peak_mb, vram_peak_mb
            while not stop_event.is_set():
                try:
                    cpu_samples.append(_normalize_cpu_pct(process.cpu_percent(None)))
                    ram_now_mb = process.memory_info().rss / (1024 * 1024)
                    ram_peak_mb = max(ram_peak_mb, ram_now_mb)

                    if cuda_available:
                        vram_peak_mb = max(vram_peak_mb, float(torch.cuda.max_memory_allocated() / (1024 * 1024)))

                    if gpu_available:
                        snap = _sample_gpu()
                        if snap:
                            gpu_util_samples.append(snap["gpu_util_pct"])
                            gpu_mem_samples.append(snap["gpu_mem_used_mb"])
                except Exception:
                    # Keep monitoring best-effort and never break workflow.
                    pass

                stop_event.wait(interval_sec)

        thread = threading.Thread(target=_sample, daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            thread.join(timeout=max(1.0, interval_sec * 5))

            ram_end_mb = process.memory_info().rss / (1024 * 1024)
            ram_peak = max(ram_peak_mb, ram_end_mb)

            cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
            cpu_peak = max(cpu_samples) if cpu_samples else 0.0

            paper.extra[f"resource_{stage}_cpu_pct_avg"] = cpu_avg
            paper.extra[f"resource_{stage}_cpu_pct_peak"] = cpu_peak
            paper.extra[f"resource_{stage}_ram_mb_start"] = ram_start_mb
            paper.extra[f"resource_{stage}_ram_mb_end"] = ram_end_mb
            paper.extra[f"resource_{stage}_ram_mb_peak"] = ram_peak

            if cuda_available:
                vram_end_mb = float(torch.cuda.memory_allocated() / (1024 * 1024))
                vram_peak = max(vram_peak_mb, vram_end_mb)
                paper.extra[f"resource_{stage}_vram_mb_start"] = vram_start_mb
                paper.extra[f"resource_{stage}_vram_mb_end"] = vram_end_mb
                paper.extra[f"resource_{stage}_vram_mb_peak"] = vram_peak

            if gpu_available and gpu_util_samples:
                gpu_util_avg = sum(gpu_util_samples) / len(gpu_util_samples)
                paper.extra[f"resource_{stage}_gpu_util_pct_avg"] = gpu_util_avg
                paper.extra[f"resource_{stage}_gpu_util_pct_peak"] = max(gpu_util_samples)
                paper.extra[f"resource_{stage}_gpu_mem_mb_peak"] = max(gpu_mem_samples) if gpu_mem_samples else 0.0

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
        df.to_csv(self.run_summary, index=False)

        return summary

    # plotting
    def _save_figure(self, path: Path) -> None:
        plt.savefig(path, dpi=_PLOT_DPI, bbox_inches="tight")

    def plot_stage_times(self):
        df = self._load_df()

        stage_cols = [c for c in df.columns if c.startswith("time_")]

        _stage_ru = {"time_grobid": "GROBID", "time_nlp": "NLP", "time_nlp_wall": "NLP (стена)", "time_neo4j": "Neo4j"}
        means = df[stage_cols].mean().sort_values()
        means.index = [_stage_ru.get(str(i), str(i)) for i in means.index]
        plt.figure(figsize=_figsize(8, 5.5))
        means.plot(kind="bar")

        plt.title("Среднее время этапов")
        plt.ylabel("Время, с")
        plt.xlabel("Этап")
        plt.tight_layout()
        self._save_figure(self.figures_dir / "stage_times.png")
        plt.close()

    def plot_distributions(self):
        df = self._load_df()

        plt.figure(figsize=_figsize(8, 5.5))
        df["relations"].plot(kind="hist", bins=30)
        plt.title("Распределение числа связей")
        plt.xlabel("Число связей")
        plt.ylabel("Частота")
        self._save_figure(self.figures_dir / "relations_hist.png")
        plt.close()

        plt.figure(figsize=_figsize(8, 5.5))
        df["entities"].plot(kind="hist", bins=30)
        plt.title("Распределение числа сущностей")
        plt.xlabel("Число сущностей")
        plt.ylabel("Частота")
        self._save_figure(self.figures_dir / "entities_hist.png")
        plt.close()

    def plot_nlp_vs_text_size(self) -> None:
        """NLP duration vs estimated token count."""
        df = self._load_df()
        if df.empty or "time_nlp" not in df.columns:
            return
        tok = "extra_text_length_tokens_est"
        if tok not in df.columns:
            return

        plt.figure(figsize=_figsize(7, 5))
        plt.scatter(df[tok], df["time_nlp"], s=80, alpha=0.85)
        plt.xlabel("Длина текста")
        plt.ylabel("Время NLP (с)")
        plt.title("Время NLP и размер документа")
        plt.tight_layout()
        self._save_figure(self.figures_dir / "nlp_time_vs_text.png")
        plt.close()

    def plot_resources_vs_paper_index(self) -> None:
        """
        Single CPU / RAM / VRAM curve vs article index (processing order).

        For each paper we take the **maximum** over stages (grobid, nlp_wall,
        neo4j) so one point per paper approximates worst observed load while
        handling that article.
        """
        df = self._load_df()
        if df.empty or "paper_id" not in df.columns:
            return

        df = df.reset_index(drop=True)
        n = len(df)
        x = np.arange(1, n + 1)
        paper_labels = df["paper_id"].astype(str).tolist()

        ram_keys = [
            "extra_resource_grobid_ram_mb_peak",
            "extra_resource_nlp_wall_ram_mb_peak",
            "extra_resource_neo4j_ram_mb_peak",
        ]
        cpu_keys = [
            "extra_resource_grobid_cpu_pct_avg",
            "extra_resource_nlp_wall_cpu_pct_avg",
            "extra_resource_neo4j_cpu_pct_avg",
        ]
        gpu_util_keys = [
            "extra_resource_grobid_gpu_util_pct_avg",
            "extra_resource_nlp_wall_gpu_util_pct_avg",
            "extra_resource_neo4j_gpu_util_pct_avg",
        ]
        gpu_mem_keys = [
            "extra_resource_grobid_gpu_mem_mb_peak",
            "extra_resource_nlp_wall_gpu_mem_mb_peak",
            "extra_resource_neo4j_gpu_mem_mb_peak",
        ]

        def _row_max(keys: list[str]) -> pd.Series | None:
            present = [k for k in keys if k in df.columns]
            if not present:
                return None
            return df[present].astype(float).max(axis=1)

        ram_y = _row_max(ram_keys)
        cpu_y = _row_max(cpu_keys)
        if cpu_y is not None:
            cpu_y = _cpu_series_host_pct(cpu_y)
        gpu_util_y = _row_max(gpu_util_keys)
        gpu_mem_y = _row_max(gpu_mem_keys)

        panels: list[tuple[pd.Series, str, str]] = []
        if ram_y is not None:
            panels.append((ram_y, "Потребление, МБ", "ОЗУ"))
        if cpu_y is not None:
            panels.append((cpu_y, "CPU, % от всех ядер", "CPU"))
        if gpu_util_y is not None:
            panels.append((gpu_util_y, "GPU, %", "Загрузка GPU"))
        if gpu_mem_y is not None:
            panels.append((gpu_mem_y, "Память GPU, МБ", "Память GPU (драйвер)"))

        if not panels:
            return

        fig, axes = plt.subplots(len(panels), 1, figsize=_figsize(6, 2.5 * len(panels)), sharex=True)
        if len(panels) == 1:
            axes = np.array([axes])

        colors = ("C0", "C1", "C2")
        for ax, (y, ylabel, title), color in zip(axes, panels, colors):
            ax.plot(x, y, marker="o", markersize=7, color=color)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.35)
            _plain_value_axis(ax, "y")

            y_min, y_max = float(y.min()), float(y.max())
            spread = y_max - y_min
            pad = max(spread * 0.15, y_max * 0.002, 1.0) if spread > 0 else max(y_max * 0.01, 5.0)
            ax.set_ylim(y_min - pad, y_max + pad)

        plt.tight_layout()
        self._save_figure(self.figures_dir / "resources_vs_paper_index.png")
        plt.close()

    def plot_complexity(self):
        df = self._load_df()

        plt.figure(figsize=_figsize(8, 5.5))
        plt.scatter(df["sentences"], df["relations"], s=80, alpha=0.85)

        plt.title("Предложения и связи")
        plt.xlabel("Число предложений")
        plt.ylabel("Число связей")

        self._save_figure(self.figures_dir / "complexity.png")
        plt.close()

    def plot_all_metric_figures(self) -> None:
        """Render stage/KGE/resource plots for the current run (metrics.jsonl)."""
        self.plot_stage_times()
        self.plot_complexity()
        self.plot_distributions()
        self.plot_resources_vs_paper_index()
        self.plot_nlp_vs_text_size()

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
        return {"precision": precision, "recall": recall, "f1": f1}
