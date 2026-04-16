from pathlib import Path


class PaperState:
    def __init__(self, paper_dir: Path):
        self.dir = paper_dir

    def pdf(self) -> Path:
        return self.dir / f"{self.dir.name}.pdf"

    def tei(self) -> Path:
        return self.dir / "tei.xml"

    def nlp(self) -> Path:
        return self.dir / "nlp.json"
