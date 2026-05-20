from __future__ import annotations

import json
import logging
import re
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
    Map psutil/docker raw CPU (sum over cores, can be >100%) to share of the host.

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


def _parse_docker_mem_usage_mib(mem_usage_field: str) -> float:
    """
    Parse one memory field from Docker stats, e.g. \"123.4MiB\" or \"4.187GiB\" -> MiB.
    NET/BLOCK sometimes use decimal MB; treat MB/GB/TB as SI bytes -> MiB.
    """
    s = mem_usage_field.strip()
    m = re.match(
        r"([\d.]+)\s*(KiB|MiB|GiB|TiB|KB|MB|GB|TB|B)\b",
        s,
        re.I,
    )
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2).upper()
    # Binary units (Docker MEM column)
    to_mib_bin = {
        "B": 1.0 / (1024 * 1024),
        "KIB": 1.0 / 1024,
        "MIB": 1.0,
        "GIB": 1024.0,
        "TIB": 1024.0 * 1024.0,
        "KB": 1.0 / 1024,  # rare for MEM
        "MB": 1000 * 1000 / (1024 * 1024),
        "GB": 1000 * 1000 * 1000 / (1024 * 1024),
        "TB": 1000 ** 4 / (1024 * 1024),
    }
    return val * to_mib_bin.get(unit, 0.0)


def _parse_docker_stats_human_table(stdout: str) -> dict[str, dict[str, float]]:
    """
    Parse default `docker stats` table (same layout as interactive docker stats).

    Columns: CONTAINER ID, NAME, CPU %, MEM USAGE / LIMIT, MEM %, NET I/O, BLOCK I/O, PIDS
    Rows are split on 2+ spaces so \"867.2MiB / 31.02GiB\" stays one column.
    """
    by_key: dict[str, dict[str, float]] = {}

    for raw in stdout.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line or line.upper().startswith("CONTAINER ID"):
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) < 8:
            continue

        cid = parts[0].strip()
        name = parts[1].strip()
        try:
            cpu_pct = float(parts[2].replace("%", "").strip())
            mem_ul = parts[3].strip()
            mem_pct = float(parts[4].replace("%", "").strip())
            pids = float(parts[7].strip())
        except ValueError:
            continue

        ul_split = [x.strip() for x in mem_ul.split("/", 1)]
        mem_usage_mib = _parse_docker_mem_usage_mib(ul_split[0]) if ul_split else 0.0
        mem_limit_mib = (
            _parse_docker_mem_usage_mib(ul_split[1]) if len(ul_split) > 1 else 0.0
        )

        metrics = {
            "cpu_pct": cpu_pct,
            "mem_pct": mem_pct,
            "mem_usage_mib": mem_usage_mib,
            "mem_limit_mib": mem_limit_mib,
            "pids": pids,
        }

        by_key[name] = metrics
        by_key[cid] = metrics
        if len(cid) >= 12:
            by_key[cid[:12]] = metrics

    return by_key


def _docker_stats_snapshot_human(container: str, stdout: str) -> dict[str, float] | None:
    table = _parse_docker_stats_human_table(stdout)
    c = container.strip()
    if c in table:
        return table[c]
    c_low = c.lower()
    for k, v in table.items():
        if k.lower() == c_low:
            return v
    return None


def _docker_stats_snapshot(container: str) -> dict[str, float] | None:
    """
    One-shot stats for a Docker container (name or id).

    Prefers parsing the default human table from `docker stats --no-stream <ref>`
    (same output as manual `docker stats`). Falls back to `--format` if parsing fails.
    """
    try:
        proc = subprocess.run(
            ["docker", "stats", "--no-stream", container],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            snap = _docker_stats_snapshot_human(container, proc.stdout)
            if snap is not None:
                return snap

        proc_fmt = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.CPUPerc}}\t{{.MemPerc}}\t{{.MemUsage}}",
                container,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc_fmt.returncode != 0:
            return None
        lines = (proc_fmt.stdout or "").strip().splitlines()
        if not lines:
            return None
        parts = lines[0].split("\t")
        if len(parts) < 3:
            return None
        cpu_pct = float(parts[0].replace("%", "").strip())
        mem_pct = float(parts[1].replace("%", "").strip())
        mem_field = parts[2].strip()
        ul_split = [x.strip() for x in mem_field.split("/", 1)]
        mem_usage_mib = _parse_docker_mem_usage_mib(ul_split[0]) if ul_split else 0.0
        mem_limit_mib = (
            _parse_docker_mem_usage_mib(ul_split[1]) if len(ul_split) > 1 else 0.0
        )
        return {
            "cpu_pct": cpu_pct,
            "mem_pct": mem_pct,
            "mem_usage_mib": mem_usage_mib,
            "mem_limit_mib": mem_limit_mib,
            "pids": 0.0,
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

        self.run_dir = paths.run_dir
        self.raw_path = paths.run_metrics
        self.run_summary = paths.run_summary
        self.figures_dir = paths.figures
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

    def monitor_stage_resources(
        self,
        paper: PaperMetrics,
        stage: str,
        interval_sec: float = 0.2,
        docker_container: str | None = None,
    ) -> Callable[[], None]:
        """
        Track resource usage for a stage into `paper.extra`.

        Local process (default):
        - RAM: RSS of the Python workflow process (psutil), not system-wide.
        - CPU: ``psutil.Process.cpu_percent`` normalized by logical CPU count
          (0–100% of host; idle I/O stages stay near 0%).
        - VRAM (``vram_mb_*``): PyTorch CUDA allocator on the default GPU only.
        - GPU (``gpu_util_pct_*``, ``gpu_mem_mb_*``): device-wide via NVML /
          ``nvidia-smi`` (utilization % and memory used by all processes).

        Docker (``docker_container`` set):
        - CPU/RAM of the container via ``docker stats`` (GROBID/Neo4j server load).
        - Set METRICS_DOCKER_GROBID / METRICS_DOCKER_NEO4J in Workflow.

        Container names are typically passed from env in ``Workflow``.
        """
        docker_container = (docker_container or "").strip() or None
        if docker_container:
            return self._monitor_docker_container(paper, stage, interval_sec, docker_container)

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
                    cpu_samples.append(
                        _normalize_cpu_pct(process.cpu_percent(None))
                    )
                    ram_now_mb = process.memory_info().rss / (1024 * 1024)
                    ram_peak_mb = max(ram_peak_mb, ram_now_mb)

                    if cuda_available:
                        vram_peak_mb = max(
                            vram_peak_mb,
                            float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
                        )

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
                paper.extra[f"resource_{stage}_gpu_mem_mb_peak"] = (
                    max(gpu_mem_samples) if gpu_mem_samples else 0.0
                )

        return stop

    def _monitor_docker_container(
        self,
        paper: PaperMetrics,
        stage: str,
        interval_sec: float,
        container: str,
    ) -> Callable[[], None]:
        """CPU/memory for a Docker container via `docker stats` (not the Python client)."""
        stop_event = threading.Event()
        cpu_samples: list[float] = []
        mem_pct_samples: list[float] = []
        mem_mib_samples: list[float] = []
        mem_limit_mib_samples: list[float] = []
        pids_samples: list[float] = []
        warned_missing = False

        def _sample() -> None:
            nonlocal warned_missing
            while not stop_event.is_set():
                snap = _docker_stats_snapshot(container)
                if snap:
                    cpu_samples.append(_normalize_cpu_pct(snap["cpu_pct"]))
                    mem_pct_samples.append(snap["mem_pct"])
                    mem_mib_samples.append(snap["mem_usage_mib"])
                    mem_limit_mib_samples.append(snap.get("mem_limit_mib", 0.0))
                    pids_samples.append(snap.get("pids", 0.0))
                elif not warned_missing:
                    warned_missing = True
                    lg.warning(
                        "Docker stats unavailable for container %r (stage %s). "
                        "Check `docker ps`, name, and DOCKER_HOST.",
                        container,
                        stage,
                    )
                stop_event.wait(interval_sec)

        thread = threading.Thread(target=_sample, daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            thread.join(timeout=max(1.0, interval_sec * 5))

            def _avg(xs: list[float]) -> float:
                return sum(xs) / len(xs) if xs else 0.0

            prefix = f"resource_{stage}_docker"
            paper.extra[f"{prefix}_cpu_pct_avg"] = _avg(cpu_samples)
            paper.extra[f"{prefix}_cpu_pct_peak"] = max(cpu_samples) if cpu_samples else 0.0
            paper.extra[f"{prefix}_mem_pct_avg"] = _avg(mem_pct_samples)
            paper.extra[f"{prefix}_mem_pct_peak"] = max(mem_pct_samples) if mem_pct_samples else 0.0
            paper.extra[f"{prefix}_mem_usage_mib_avg"] = _avg(mem_mib_samples)
            paper.extra[f"{prefix}_mem_usage_mib_peak"] = (
                max(mem_mib_samples) if mem_mib_samples else 0.0
            )
            paper.extra[f"{prefix}_mem_limit_mib_avg"] = _avg(mem_limit_mib_samples)
            paper.extra[f"{prefix}_mem_limit_mib_peak"] = (
                max(mem_limit_mib_samples) if mem_limit_mib_samples else 0.0
            )
            paper.extra[f"{prefix}_pids_avg"] = _avg(pids_samples)
            paper.extra[f"{prefix}_pids_peak"] = (
                max(pids_samples) if pids_samples else 0.0
            )

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

        _stage_ru = {
            "time_grobid": "GROBID",
            "time_nlp": "NLP",
            "time_nlp_wall": "NLP (стена)",
            "time_neo4j": "Neo4j",
        }
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

    def plot_complexity(self):
        df = self._load_df()

        plt.figure(figsize=_figsize(8, 5.5))
        plt.scatter(df["sentences"], df["relations"], s=80, alpha=0.85)

        plt.title("Предложения и связи")
        plt.xlabel("Число предложений")
        plt.ylabel("Число связей")

        self._save_figure(self.figures_dir / "complexity.png")
        plt.close()

    def plot_stage_times_per_paper(self) -> None:
        """Grouped bars: wall time per stage for each paper_id."""
        df = self._load_df()
        if df.empty:
            return
        cols = [c for c in ("time_grobid", "time_nlp", "time_neo4j") if c in df.columns]
        if not cols:
            return
        rename = {
            "time_grobid": "GROBID",
            "time_nlp": "NLP",
            "time_neo4j": "Neo4j",
        }
        plot_df = df.set_index("paper_id")[cols].rename(columns=rename)
        w, h = _figsize(max(8.0, 0.45 * len(df)), 5)
        ax = plot_df.plot(
            kind="bar",
            figsize=(w, h),
            rot=25,
        )
        ax.set_title("Время этапов по статьям")
        ax.set_ylabel("с")
        ax.set_xlabel("")
        ax.legend(title="Этап")
        plt.tight_layout()
        self._save_figure(self.figures_dir / "stage_times_per_paper.png")
        plt.close()

    def plot_local_resources_per_paper(self) -> None:
        """RSS / CPU / GPU util / GPU & PyTorch memory per paper."""
        df = self._load_df()
        if df.empty:
            return

        ram_cols = [
            c
            for c in (
                "extra_resource_grobid_ram_mb_peak",
                "extra_resource_nlp_wall_ram_mb_peak",
                "extra_resource_neo4j_ram_mb_peak",
            )
            if c in df.columns
        ]
        cpu_cols = [
            c
            for c in (
                "extra_resource_grobid_cpu_pct_avg",
                "extra_resource_nlp_wall_cpu_pct_avg",
                "extra_resource_neo4j_cpu_pct_avg",
            )
            if c in df.columns
        ]
        gpu_util_cols = [
            c
            for c in (
                "extra_resource_grobid_gpu_util_pct_avg",
                "extra_resource_nlp_wall_gpu_util_pct_avg",
                "extra_resource_neo4j_gpu_util_pct_avg",
            )
            if c in df.columns
        ]
        gpu_mem_cols = [
            c
            for c in (
                "extra_resource_grobid_gpu_mem_mb_peak",
                "extra_resource_nlp_wall_gpu_mem_mb_peak",
                "extra_resource_neo4j_gpu_mem_mb_peak",
            )
            if c in df.columns
        ]
        vram_col = "extra_resource_nlp_wall_vram_mb_peak"

        n_plots = sum(
            [
                bool(ram_cols),
                bool(cpu_cols),
                bool(gpu_util_cols),
                bool(gpu_mem_cols),
                vram_col in df.columns,
            ]
        )
        if n_plots == 0:
            return

        fig, axes = plt.subplots(1, n_plots, figsize=_figsize(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        i = 0
        stage_labels = {
            "grobid": "grobid",
            "nlp_wall": "nlp_wall",
            "neo4j": "neo4j",
        }

        def _legend_labels(cols: list[str]) -> list[str]:
            labels = []
            for c in cols:
                for key, name in stage_labels.items():
                    if key in c:
                        labels.append(name)
                        break
                else:
                    labels.append(c.replace("extra_resource_", "")[:24])
            return labels

        if ram_cols:
            df.set_index("paper_id")[ram_cols].plot(kind="bar", ax=axes[i], rot=25)
            axes[i].set_title("Пик RSS процесса (МБ)")
            axes[i].set_ylabel("МБ")
            axes[i].legend(labels=_legend_labels(ram_cols))
            i += 1

        if cpu_cols:
            df.set_index("paper_id")[cpu_cols].plot(kind="bar", ax=axes[i], rot=25)
            axes[i].set_title("CPU, % от всех ядер (среднее)")
            axes[i].set_ylabel("%")
            axes[i].legend(labels=_legend_labels(cpu_cols))
            i += 1

        if gpu_util_cols:
            df.set_index("paper_id")[gpu_util_cols].plot(kind="bar", ax=axes[i], rot=25)
            axes[i].set_title("Загрузка GPU (среднее)")
            axes[i].set_ylabel("%")
            axes[i].legend(labels=_legend_labels(gpu_util_cols))
            i += 1

        if gpu_mem_cols:
            df.set_index("paper_id")[gpu_mem_cols].plot(kind="bar", ax=axes[i], rot=25)
            axes[i].set_title("Память GPU, пик (драйвер, МБ)")
            axes[i].set_ylabel("МБ")
            axes[i].legend(labels=_legend_labels(gpu_mem_cols))
            i += 1

        if vram_col in df.columns:
            df.set_index("paper_id")[[vram_col]].plot(kind="bar", ax=axes[i], rot=25, legend=False)
            axes[i].set_title("Память PyTorch CUDA, пик на NLP (МБ)")
            axes[i].set_ylabel("МБ")
            i += 1

        plt.tight_layout()
        self._save_figure(self.figures_dir / "local_resources_per_paper.png")
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

    def plot_kg_derived_metrics(self) -> None:
        """Histograms / bars for analyzer-derived KG statistics."""
        df = self._load_df()
        if df.empty:
            return

        needed = (
            "extra_relation_entropy",
            "extra_relation_coverage",
            "extra_avg_entity_mentions_per_sentence",
            "extra_avg_relations_per_sentence",
        )
        if not any(c in df.columns for c in needed):
            return

        fig, axes = plt.subplots(2, 2, figsize=_figsize(10, 8))

        used = [[False, False], [False, False]]

        if "extra_relation_entropy" in df.columns:
            df["extra_relation_entropy"].plot(
                kind="hist",
                bins=min(20, max(3, len(df))),
                ax=axes[0, 0],
            )
            axes[0, 0].set_title("Энтропия типов связей")
            axes[0, 0].set_xlabel("энтропия, биты")
            axes[0, 0].set_ylabel("частота")
            used[0][0] = True

        if "extra_relation_coverage" in df.columns:
            df.set_index("paper_id")["extra_relation_coverage"].plot(
                kind="bar", ax=axes[0, 1], rot=25
            )
            axes[0, 1].set_title("Покрытие связями")
            axes[0, 1].set_ylabel("доля")
            axes[0, 1].set_xlabel("")
            used[0][1] = True

        if "extra_avg_entity_mentions_per_sentence" in df.columns:
            df.set_index("paper_id")["extra_avg_entity_mentions_per_sentence"].plot(
                kind="bar", ax=axes[1, 0], rot=25
            )
            axes[1, 0].set_title("Среднее число упоминаний сущностей на предложение")
            axes[1, 0].set_ylabel("значение")
            axes[1, 0].set_xlabel("")
            used[1][0] = True

        if "extra_avg_relations_per_sentence" in df.columns:
            df.set_index("paper_id")["extra_avg_relations_per_sentence"].plot(
                kind="bar", ax=axes[1, 1], rot=25
            )
            axes[1, 1].set_title("Среднее число связей на предложение")
            axes[1, 1].set_ylabel("значение")
            axes[1, 1].set_xlabel("")
            used[1][1] = True

        for r in range(2):
            for c in range(2):
                if not used[r][c]:
                    axes[r, c].set_visible(False)

        if not any(u for row in used for u in row):
            plt.close()
            return

        plt.tight_layout()
        self._save_figure(self.figures_dir / "kg_derived_metrics.png")
        plt.close()

    def plot_docker_resources_per_paper(self) -> None:
        """Container CPU/mem from docker stats (only if METRICS_DOCKER_* was set)."""
        df = self._load_df()
        if df.empty:
            return

        cpu_pat = "_docker_cpu_pct_avg"
        mem_pat = "_docker_mem_usage_mib_peak"
        cpu_cols = sorted(c for c in df.columns if cpu_pat in c)
        mem_cols = sorted(c for c in df.columns if mem_pat in c)
        if not cpu_cols and not mem_cols:
            return

        n = sum([bool(cpu_cols), bool(mem_cols)])
        fig, axes = plt.subplots(1, n, figsize=_figsize(6 * n, 4.5))
        if n == 1:
            axes = [axes]

        def _short_label(col: str) -> str:
            c = col.lower()
            stage = ""
            if "grobid" in c:
                stage = "GROBID"
            elif "neo4j" in c:
                stage = "Neo4j"
            elif "nlp" in c:
                stage = "NLP"
            if "cpu" in c:
                return f"{stage}: CPU % (ср.)" if stage else col[:40]
            if "mem" in c or "mib" in c:
                return f"{stage}: ОЗУ, пик (МБ)" if stage else col[:40]
            return col.replace("extra_resource_", "")[:40]

        ax_i = 0
        if cpu_cols:
            df.set_index("paper_id")[cpu_cols].plot(kind="bar", ax=axes[ax_i], rot=25)
            axes[ax_i].set_title("Контейнер Docker: CPU, % от всех ядер (среднее)")
            axes[ax_i].set_ylabel("%")
            axes[ax_i].legend(
                [_short_label(c) for c in cpu_cols],
                loc="upper left",
            )
            ax_i += 1

        if mem_cols:
            df.set_index("paper_id")[mem_cols].plot(kind="bar", ax=axes[ax_i], rot=25)
            axes[ax_i].set_title("Контейнер Docker: использование памяти, пик (МБ)")
            axes[ax_i].set_ylabel("МБ")
            axes[ax_i].legend(
                [_short_label(c) for c in mem_cols],
                loc="upper left",
            )

        plt.tight_layout()
        self._save_figure(self.figures_dir / "docker_resources_per_paper.png")
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
            panels.append(
                (
                    ram_y,
                    "Потребление, МБ",
                    "ОЗУ",
                )
            )
        if cpu_y is not None:
            panels.append(
                (
                    cpu_y,
                    "CPU, % от всех ядер",
                    "CPU",
                )
            )
        if gpu_util_y is not None:
            panels.append(
                (
                    gpu_util_y,
                    "GPU, %",
                    "Загрузка GPU",
                )
            )
        if gpu_mem_y is not None:
            panels.append(
                (
                    gpu_mem_y,
                    "Память GPU, МБ",
                    "Память GPU (драйвер)",
                )
            )

        if not panels:
            return

        fig, axes = plt.subplots(
            len(panels), 1, figsize=_figsize(6, 2.5 * len(panels)), sharex=True
        )
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

        # axes[-1].set_xlabel("Индекс статьи (порядок в этом прогоне)")
        # if n <= 24:
        #     axes[-1].set_xticks(x)
        #     axes[-1].set_xticklabels(paper_labels, rotation=40, ha="right")
        # else:
        #     axes[-1].set_xticks(x)

        plt.tight_layout()
        self._save_figure(self.figures_dir / "resources_vs_paper_index.png")
        plt.close()

    def plot_all_metric_figures(self) -> None:
        """Render stage/KGE/resource plots for the current run (metrics.jsonl)."""
        self.plot_stage_times()
        self.plot_complexity()
        self.plot_distributions()
        self.plot_stage_times_per_paper()
        self.plot_local_resources_per_paper()
        self.plot_resources_vs_paper_index()
        self.plot_nlp_vs_text_size()
        self.plot_kg_derived_metrics()
        self.plot_docker_resources_per_paper()

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
