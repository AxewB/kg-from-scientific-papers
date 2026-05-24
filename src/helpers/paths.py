import datetime
from pathlib import Path


class Paths:
    def __init__(self) -> None:
        self.root: Path = self._find_project_root()

        self.output: Path = self.root / "output"

        self.log: Path = self.root / self.output / "log"
        self.cache: Path = self.root / self.output /  "cache"

        self.papers: Path = self.cache / "papers"
        self.metrics_root: Path = self.cache / "metrics"

        self.ensure_dirs()

        self.run_dir: Path = self._create_run_dir()

        self.run_metrics: Path = self.run_dir / "metrics.jsonl"
        self.run_summary: Path = self.run_dir / "summary.csv"
        self.run_report: Path = self.run_dir / "report.md"
        self.figures: Path = self.run_dir / "figures"

        self.figures.mkdir(parents=True, exist_ok=True)

    def _create_run_dir(self) -> Path:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.metrics_root / ts
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _find_project_root(self) -> Path:
        current = Path(__file__).resolve().parent

        for _ in range(10):
            if any(
                (current / m).exists() for m in ["pyproject.toml", ".git", "README.md"]
            ):
                return current
            current = current.parent

        return Path(__file__).resolve().parent

    def ensure_dirs(self) -> None:
        for p in [self.log, self.cache, self.papers, self.metrics_root]:
            p.mkdir(parents=True, exist_ok=True)

    def file(self, *parts: str, ext: str | None = None) -> Path:
        path = self.root.joinpath(*parts)
        if ext and not path.suffix:
            path = path.with_suffix(ext)
        return path

    def log_file(self, name: str | None = None) -> Path:
        if name is None:
            name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        return self.log / f"{name}.log"


paths = Paths()
