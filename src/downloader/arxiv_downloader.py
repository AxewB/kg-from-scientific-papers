# docs:
# https://lukasschwab.me/arxiv.py/arxiv.html

from pathlib import Path
from time import sleep

from arxiv import Client, Search, SortCriterion

from downloader.downloader_base import DownloaderBase
from helpers.paths import paths

CATEGORIES = [
    # Computer Science
    "cs.AI",
    "cs.AR",
    "cs.CC",
    "cs.CE",
    "cs.CG",
    "cs.CL",
    "cs.CR",
    "cs.CV",
    "cs.CY",
    "cs.DB",
    "cs.DC",
    "cs.DL",
    "cs.DM",
    "cs.DS",
    "cs.ET",
    "cs.FL",
    "cs.GL",
    "cs.GR",
    "cs.GT",
    "cs.HC",
    "cs.IR",
    "cs.IT",
    "cs.LG",
    "cs.LO",
    "cs.MA",
    "cs.MM",
    "cs.MS",
    "cs.NA",
    "cs.NE",
    "cs.NI",
    "cs.OH",
    "cs.OS",
    "cs.PF",
    "cs.PL",
    "cs.RO",
    "cs.SC",
    "cs.SD",
    "cs.SE",
    "cs.SI",
    "cs.SY",
    # Economics
    "econ.EM",
    "econ.GN",
    "econ.TH",
    "eess.AS",
    "eess.IV",
    "eess.SP",
    "eess.SY",
    # Mathematics
    "math.AC",
    "math.AG",
    "math.AP",
    "math.AT",
    "math.CA",
    "math.CO",
    "math.CT",
    "math.CV",
    "math.DG",
    "math.DS",
    "math.FA",
    "math.GM",
    "math.GN",
    "math.GR",
    "math.GT",
    "math.HO",
    "math.IT",
    "math.KT",
    "math.IT",
    "math.LO",
    "math.MG",
    "math.MP",
    "math.MP",
    "math.NA",
    "math.NT",
    "math.OA",
    "math.OC",
    "math.PR",
    "math.QA",
    "math.RA",
    "math.RT",
    "math.SG",
    "math.SP",
    "math.ST",
    # Physics
    ## Astrophysics
    "astro-ph.CO",
    "astro-ph.EP",
    "astro-ph.GA",
    "astro-ph.HE",
    "astro-ph.IM",
    "astro-ph.SR",
    ## Condensed Matter
    "cond-mat.dis-nn",
    "cond-mat.mes-hall",
    "cond-mat.mtrl-sci",
    "cond-mat.other",
    "cond-mat.quant-gas",
    "cond-mat.soft",
    "cond-mat.stat-mech",
    "cond-mat.str-el",
    "cond-mat.supr-con",
    ## General Relativity and Quantum Cosmology
    "gr-qc",
    ## High Energy Physics - Experiment
    "hep-ex",
    ## High Energy Physics - Lattice
    "hep-lat",
    ## High Energy Physics - Phenomenology
    "hep-ph",
    ## High Energy Physics - Theory
    "hep-th",
    ## Mathematical Physics
    "math-ph",
    "nlin.AO",
    "nlin.CD",
    "nlin.CG",
    "nlin.PS",
    "nlin.SI",
    ## Nuclear Experiment
    "nucl-ex",
    ## Nuclear Theory
    "nucl-th",
    # Physics
    "physics.acc-ph",
    "physics.ao-ph",
    "physics.app-ph",
    "physics.atm-clus",
    "physics.atom-ph",
    "physics.bio-ph",
    "physics.chem-ph",
    "physics.class-ph",
    "physics.comp-ph",
    "physics.data-an",
    "physics.ed-ph",
    "physics.flu-dyn",
    "physics.gen-ph",
    "physics.geo-ph",
    "physics.optics",
    "physics.pop-ph",
    "physics.plasm-ph",
    "physics.med-ph",
    "physics.hist-ph",
    "physics.ins-det",
    "physics.soc-ph",
    "physics.space-ph",
    "quant-ph",
    # Quantitative Biology
    "q-bio.BM",
    "q-bio.CB",
    "q-bio.GN",
    "q-bio.MN",
    "q-bio.NC",
    "q-bio.OT",
    "q-bio.PE",
    "q-bio.QM",
    "q-bio.SC",
    "q-bio.TO",
    # Quantitative Finance
    "q-fin.CP",
    "q-fin.EC",
    "q-fin.GN",
    "q-fin.EC",
    "q-fin.MF",
    "q-fin.PM",
    "q-fin.PR",
    "q-fin.RM",
    "q-fin.ST",
    "q-fin.TR",
    # Statistics
    "stat.AP",
    "stat.CO",
    "stat.ME",
    "stat.ML",
    "stat.OT",
    "stat.TH",
    "stat.TH",
]


class ArxivDownloader(DownloaderBase):
    def __init__(
        self,
        categories: list[str] = CATEGORIES,
        num_each: int = 3,
        download_dir: Path = paths.papers,
    ) -> None:
        """
        categories - categories dictionary (https://arxiv.org/category_taxonomy)
        num_each - how much papers of each category should be downloaded
        root_dir - parent dir where all papers should be saved
        """
        super().__init__(root_dir=download_dir)

        self.categories = categories
        self.num_each = num_each

    def download(self):
        client = Client()
        papers = []

        for category in self.categories:
            print(f'Category "{category}"')

            search = Search(
                query=category,
                max_results=self.num_each,
                sort_by=SortCriterion.SubmittedDate,
            )

            for r in client.results(search):
                paper_id = r.entry_id.split("/")[-1]

                paper_dir = self.root_dir / paper_id
                paper_dir.mkdir(parents=True, exist_ok=True)

                paper_name = f"{paper_id}.pdf"
                pdf_path = paper_dir / paper_name

                # download only if missing
                if not pdf_path.exists():
                    r.download_pdf(
                        dirpath=str(paper_dir),
                        filename=paper_name,
                    )
                    print(f"Downloaded: {pdf_path}")
                else:
                    print(f"Exists: {paper_id}")

                # IMPORTANT: always return path
                papers.append(pdf_path)

            sleep(3)

        return papers

    def _safe_name(self, name: str) -> str:
        return "".join(c for c in name if c.isalnum() or c in "._- ").strip()
