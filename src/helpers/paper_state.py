from pathlib import Path


class PaperState:
    def __init__(self, paper_dir: Path):
        self.dir: Path = paper_dir
        self.dir.mkdir(parents=True, exist_ok=True)

        self.pdf: Path = self.dir / f"{self.dir.name}.pdf"
        self.tei: Path = self.dir / "tei.xml"
        self.nlp: Path = self.dir / "nlp.json"

    def is_valid_tei(self) -> bool:
        if not self.tei.exists():
            return False

        try:
            content = self.tei.read_text(encoding="utf-8")
            return bool(content.strip() and content.lstrip().startswith("<?xml"))
        except Exception:
            return False
