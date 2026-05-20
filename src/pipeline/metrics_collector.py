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

        - Default: local Python process via psutil (+ PyTorch VRAM when CUDA is used).
        - If `docker_container` is set (e.g. GROBID / Neo4j in Docker): poll that
          container with `docker stats` instead — the client process is not the server.

        Container names are typically passed from env in `Workflow` (see
        METRICS_DOCKER_GROBID / METRICS_DOCKER_NEO4J).
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

        ram_start_mb = process.memory_info().rss / (1024 * 1024)
        ram_peak_mb = ram_start_mb

        cuda_available = bool(torch is not None and torch.cuda.is_available())
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
                    cpu_samples.append(process.cpu_percent(None))
                    ram_now_mb = process.memory_info().rss / (1024 * 1024)
                    ram_peak_mb = max(ram_peak_mb, ram_now_mb)

                    if cuda_available:
                        vram_peak_mb = max(
                            vram_peak_mb,
                            float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
                        )
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
                    cpu_samples.append(snap["cpu_pct"])
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
        means.plot(kind="bar")

        plt.title("Среднее время этапов")
        plt.ylabel("с")
        plt.xlabel("Этап")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "stage_times.png")
        plt.close()

    def plot_distributions(self):
        df = self._load_df()

        df["relations"].plot(kind="hist", bins=30)
        plt.title("Распределение числа связей")
        plt.xlabel("Число связей")
        plt.ylabel("Частота")
        plt.savefig(self.figures_dir / "relations_hist.png")
        plt.close()

        df["entities"].plot(kind="hist", bins=30)
        plt.title("Распределение числа сущностей")
        plt.xlabel("Число сущностей")
        plt.ylabel("Частота")
        plt.savefig(self.figures_dir / "entities_hist.png")
        plt.close()

    def plot_complexity(self):
        df = self._load_df()

        plt.scatter(df["sentences"], df["relations"])

        plt.title("Предложения и связи")
        plt.xlabel("Число предложений")
        plt.ylabel("Число связей")

        plt.savefig(self.figures_dir / "complexity.png")
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
        ax = plot_df.plot(
            kind="bar",
            figsize=(max(8.0, 0.45 * len(df)), 5),
            rot=25,
        )
        ax.set_title("Время этапов по статьям")
        ax.set_ylabel("с")
        ax.set_xlabel("")
        ax.legend(title="Этап")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "stage_times_per_paper.png")
        plt.close()

    def plot_local_resources_per_paper(self) -> None:
        """RSS / VRAM / CPU (local Python process) peaks & NLP VRAM per paper."""
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
        vram_col = "extra_resource_nlp_wall_vram_mb_peak"
        cpu_cols = [
            c
            for c in (
                "extra_resource_grobid_cpu_pct_avg",
                "extra_resource_nlp_wall_cpu_pct_avg",
                "extra_resource_neo4j_cpu_pct_avg",
            )
            if c in df.columns
        ]

        n_plots = sum([bool(ram_cols), vram_col in df.columns, bool(cpu_cols)])
        if n_plots == 0:
            return

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
        if n_plots == 1:
            axes = [axes]

        i = 0
        ram_labels = {
            "extra_resource_grobid_ram_mb_peak": "grobid",
            "extra_resource_nlp_wall_ram_mb_peak": "nlp_wall",
            "extra_resource_neo4j_ram_mb_peak": "neo4j",
        }
        if ram_cols:
            df.set_index("paper_id")[ram_cols].plot(kind="bar", ax=axes[i], rot=25)
            axes[i].set_title("Пик RSS процесса (МиБ)")
            axes[i].set_ylabel("МиБ")
            axes[i].legend(labels=[ram_labels[c] for c in ram_cols], fontsize=8)
            i += 1

        if vram_col in df.columns:
            df.set_index("paper_id")[[vram_col]].plot(kind="bar", ax=axes[i], rot=25, legend=False)
            axes[i].set_title("Пик VRAM PyTorch на этапе NLP (МиБ)")
            axes[i].set_ylabel("МиБ")
            i += 1

        cpu_labels = {
            "extra_resource_grobid_cpu_pct_avg": "grobid",
            "extra_resource_nlp_wall_cpu_pct_avg": "nlp_wall",
            "extra_resource_neo4j_cpu_pct_avg": "neo4j",
        }
        if cpu_cols:
            df.set_index("paper_id")[cpu_cols].plot(kind="bar", ax=axes[i], rot=25)
            axes[i].set_title(
                "Средняя загрузка CPU, % (на многоядерных системах может быть >100)"
            )
            axes[i].set_ylabel("%")
            axes[i].legend(labels=[cpu_labels[c] for c in cpu_cols], fontsize=8)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "local_resources_per_paper.png")
        plt.close()

    def plot_nlp_vs_text_size(self) -> None:
        """NLP duration vs estimated token count."""
        df = self._load_df()
        if df.empty or "time_nlp" not in df.columns:
            return
        tok = "extra_text_length_tokens_est"
        if tok not in df.columns:
            return

        plt.figure(figsize=(7, 5))
        plt.scatter(df[tok], df["time_nlp"], s=60, alpha=0.85)
        plt.xlabel("Длина текста")
        plt.ylabel("Время NLP (с)")
        plt.title("Время NLP и размер документа")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "nlp_time_vs_text.png")
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

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

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
        plt.savefig(self.figures_dir / "kg_derived_metrics.png")
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
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
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
                return f"{stage}: CPU % (средн.)" if stage else col[:40]
            if "mem" in c or "mib" in c:
                return f"{stage}: ОЗУ, пик (МиБ)" if stage else col[:40]
            return col.replace("extra_resource_", "")[:40]

        ax_i = 0
        if cpu_cols:
            df.set_index("paper_id")[cpu_cols].plot(kind="bar", ax=axes[ax_i], rot=25)
            axes[ax_i].set_title("Контейнер Docker: CPU, % (среднее за этап)")
            axes[ax_i].set_ylabel("%")
            axes[ax_i].legend(
                [_short_label(c) for c in cpu_cols],
                fontsize=7,
                loc="upper left",
            )
            ax_i += 1

        if mem_cols:
            df.set_index("paper_id")[mem_cols].plot(kind="bar", ax=axes[ax_i], rot=25)
            axes[ax_i].set_title("Контейнер Docker: использование памяти, пик (МиБ)")
            axes[ax_i].set_ylabel("МиБ")
            axes[ax_i].legend(
                [_short_label(c) for c in mem_cols],
                fontsize=7,
                loc="upper left",
            )

        plt.tight_layout()
        plt.savefig(self.figures_dir / "docker_resources_per_paper.png")
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
        vram_keys = [
            "extra_resource_grobid_vram_mb_peak",
            "extra_resource_nlp_wall_vram_mb_peak",
            "extra_resource_neo4j_vram_mb_peak",
        ]

        def _row_max(keys: list[str]) -> pd.Series | None:
            present = [k for k in keys if k in df.columns]
            if not present:
                return None
            return df[present].astype(float).max(axis=1)

        ram_y = _row_max(ram_keys)
        cpu_y = _row_max(cpu_keys)
        vram_y = _row_max(vram_keys)

        panels: list[tuple[pd.Series, str, str]] = []
        if ram_y is not None:
            panels.append(
                (
                    ram_y,
                    "Пик RSS, МиБ",
                    "ОЗУ — максимум по этапам (GROBID, NLP, Neo4j) на статью",
                )
            )
        if cpu_y is not None:
            panels.append(
                (
                    cpu_y,
                    "CPU, % (макс. средних по этапам)",
                    "CPU — максимум средней загрузки по этапам на статью",
                )
            )
        if vram_y is not None:
            panels.append(
                (
                    vram_y,
                    "Пик VRAM, МиБ",
                    "VRAM GPU — максимум PyTorch по этапам на статью",
                )
            )

        if not panels:
            return

        fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.2 * len(panels)), sharex=True)
        if len(panels) == 1:
            axes = np.array([axes])

        colors = ("C0", "C1", "C2")
        for ax, (y, ylabel, title), color in zip(axes, panels, colors):
            ax.plot(x, y, marker="o", markersize=5, color=color)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.35)

        axes[-1].set_xlabel("Индекс статьи (порядок в этом прогоне)")
        if n <= 24:
            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels(paper_labels, rotation=40, ha="right", fontsize=7)
        else:
            axes[-1].set_xticks(x)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "resources_vs_paper_index.png")
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
