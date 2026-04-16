import datetime
from pathlib import Path


class Paths:
  def __init__(self) -> None:
    self.root = self._find_project_root()

    self.log = self.root / ".log"
    self.cache = self.root / ".cache"
    self.papers = self.cache / "papers"

    self.ensure_dirs()

  def _find_project_root(self) -> Path:
    current = Path(__file__).resolve().parent

    for _ in range(10):
      if any(
        (current / marker).exists()
        for marker in ["pyproject.toml", ".git", "README.md"]
      ):
        return current
      current = current.parent

    return Path(__file__).resolve().parent

  def ensure_dirs(self) -> None:
    for p in [self.log, self.cache, self.papers]:
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
