from dataclasses import dataclass, field

from domain.paper_domain import PaperDomain, infer_domain


@dataclass
class Category:
    """For now uses arXiv categories specification"""

    code: str  # "cs.LG"
    name: str  # "Machine Learning"
    description: str | None = None
    keywords: list[str] = field(default_factory=list)

    @property
    def parent(self) -> PaperDomain:
        return infer_domain(self.code)


CATEGORY_REGISTRY = {
    # Computer Science
    "cs.AI": Category(code="cs.AI", name="Artificial Intelligence"),
    "cs.AR": Category(code="cs.AR", name="Hardware Architecture"),
    "cs.CC": Category(code="cs.CC", name="Computational Complexity"),
    "cs.CE": Category(
        code="cs.CE",
        name="Computational Engineering, Finance, and Science",
    ),
    "cs.CG": Category(code="cs.CG", name="Computational Geometry"),
    "cs.CL": Category(code="cs.CL", name="Computation and Language"),
    "cs.CR": Category(code="cs.CR", name="Cryptography and Security"),
    "cs.CV": Category(
        code="cs.CV",
        name="Computer Vision and Pattern Recognition",
    ),
    "cs.CY": Category(code="cs.CY", name="Computers and Society"),
    "cs.DB": Category(code="cs.DB", name="Databases"),
    "cs.DC": Category(
        code="cs.DC",
        name="Distributed, Parallel, and Cluster Computing",
    ),
    "cs.DL": Category(code="cs.DL", name="Digital Libraries"),
    "cs.DM": Category(code="cs.DM", name="Discrete Mathematics"),
    "cs.DS": Category(code="cs.DS", name="Data Structures and Algorithms"),
    "cs.ET": Category(code="cs.ET", name="Emerging Technologies"),
    "cs.FL": Category(code="cs.FL", name="Formal Languages and Automata Theory"),
    "cs.GL": Category(code="cs.GL", name="General Literature"),
    "cs.GR": Category(code="cs.GR", name="Graphics"),
    "cs.GT": Category(code="cs.GT", name="Computer Science and Game Theory"),
    "cs.HC": Category(code="cs.HC", name="Human-Computer Interaction"),
    "cs.IR": Category(code="cs.IR", name="Information Retrieval"),
    "cs.IT": Category(code="cs.IT", name="Information Theory"),
    "cs.LG": Category(code="cs.LG", name="Machine Learning"),
    "cs.LO": Category(code="cs.LO", name="Logic in Computer Science"),
    "cs.MA": Category(code="cs.MA", name="Multiagent Systems"),
    "cs.MM": Category(code="cs.MM", name="Multimedia"),
    "cs.MS": Category(code="cs.MS", name="Mathematical Software"),
    "cs.NA": Category(code="cs.NA", name="Numerical Analysis"),
    "cs.NE": Category(code="cs.NE", name="Neural and Evolutionary Computing"),
    "cs.NI": Category(code="cs.NI", name="Networking and Internet Architecture"),
    "cs.OH": Category(code="cs.OH", name="Other Computer Science"),
    "cs.OS": Category(code="cs.OS", name="Operating Systems"),
    "cs.PF": Category(code="cs.PF", name="Performance"),
    "cs.PL": Category(code="cs.PL", name="Programming Languages"),
    "cs.RO": Category(code="cs.RO", name="Robotics"),
    "cs.SC": Category(code="cs.SC", name="Symbolic Computation"),
    "cs.SD": Category(code="cs.SD", name="Sound"),
    "cs.SE": Category(code="cs.SE", name="Software Engineering"),
    "cs.SI": Category(code="cs.SI", name="Social and Information Networks"),
    "cs.SY": Category(code="cs.SY", name="Systems and Control"),
    # Economics
    "econ.EM": Category(code="econ.EM", name="Econometrics"),
    "econ.GN": Category(code="econ.GN", name="General Economics"),
    "econ.TH": Category(code="econ.TH", name="Theoretical Economics"),
    # Electrical Engineering and Systems Science
    "eess.AS": Category(code="eess.AS", name="Audio and Speech Processing"),
    "eess.IV": Category(code="eess.IV", name="Image and Video Processing"),
    "eess.SP": Category(code="eess.SP", name="Signal Processing"),
    "eess.SY": Category(code="eess.SY", name="Systems and Control"),
    # Mathematics
    "math.AC": Category(code="math.AC", name="Commutative Algebra"),
    "math.AG": Category(code="math.AG", name="Algebraic Geometry"),
    "math.AP": Category(code="math.AP", name="Analysis of PDEs"),
    "math.AT": Category(code="math.AT", name="Algebraic Topology"),
    "math.CA": Category(code="math.CA", name="Classical Analysis and ODEs"),
    "math.CO": Category(code="math.CO", name="Combinatorics"),
    "math.CT": Category(code="math.CT", name="Category Theory"),
    "math.CV": Category(code="math.CV", name="Complex Variables"),
    "math.DG": Category(code="math.DG", name="Differential Geometry"),
    "math.DS": Category(code="math.DS", name="Dynamical Systems"),
    "math.FA": Category(code="math.FA", name="Functional Analysis"),
    "math.GM": Category(code="math.GM", name="General Mathematics"),
    "math.GN": Category(code="math.GN", name="General Topology"),
    "math.GR": Category(code="math.GR", name="Group Theory"),
    "math.GT": Category(code="math.GT", name="Geometric Topology"),
    "math.HO": Category(code="math.HO", name="History and Overview"),
    "math.KT": Category(code="math.KT", name="K-Theory and Homology"),
    "math.IT": Category(code="math.IT", name="Information Theory"),
    "math.LO": Category(code="math.LO", name="Logic"),
    "math.MG": Category(code="math.MG", name="Metric Geometry"),
    "math.MP": Category(code="math.MP", name="Mathematical Physics"),
    "math.NA": Category(code="math.NA", name="Numerical Analysis"),
    "math.NT": Category(code="math.NT", name="Number Theory"),
    "math.OA": Category(code="math.OA", name="Operator Algebras"),
    "math.OC": Category(code="math.OC", name="Optimization and Control"),
    "math.PR": Category(code="math.PR", name="Probability"),
    "math.QA": Category(code="math.QA", name="Quantum Algebra"),
    "math.RA": Category(code="math.RA", name="Rings and Algebras"),
    "math.RT": Category(code="math.RT", name="Representation Theory"),
    "math.SG": Category(code="math.SG", name="Symplectic Geometry"),
    "math.SP": Category(code="math.SP", name="Spectral Theory"),
    "math.ST": Category(code="math.ST", name="Statistics Theory"),
    # Physics (including Astrophysics, Condensed Matter, etc.)
    "astro-ph.CO": Category(
        code="astro-ph.CO",
        name="Cosmology and Nongalactic Astrophysics",
    ),
    "astro-ph.EP": Category(
        code="astro-ph.EP",
        name="Earth and Planetary Astrophysics",
    ),
    "astro-ph.GA": Category(code="astro-ph.GA", name="Astrophysics of Galaxies"),
    "astro-ph.HE": Category(
        code="astro-ph.HE",
        name="High Energy Astrophysical Phenomena",
    ),
    "astro-ph.IM": Category(
        code="astro-ph.IM",
        name="Instrumentation and Methods for Astrophysics",
    ),
    "astro-ph.SR": Category(
        code="astro-ph.SR",
        name="Solar and Stellar Astrophysics",
    ),
    "cond-mat.dis-nn": Category(
        code="cond-mat.dis-nn",
        name="Disordered Systems and Neural Networks",
    ),
    "cond-mat.mes-hall": Category(
        code="cond-mat.mes-hall",
        name="Mesoscale and Nanoscale Physics",
    ),
    "cond-mat.mtrl-sci": Category(code="cond-mat.mtrl-sci", name="Materials Science"),
    "cond-mat.other": Category(code="cond-mat.other", name="Other Condensed Matter"),
    "cond-mat.quant-gas": Category(code="cond-mat.quant-gas", name="Quantum Gases"),
    "cond-mat.soft": Category(code="cond-mat.soft", name="Soft Condensed Matter"),
    "cond-mat.stat-mech": Category(
        code="cond-mat.stat-mech",
        name="Statistical Mechanics",
    ),
    "cond-mat.str-el": Category(
        code="cond-mat.str-el",
        name="Strongly Correlated Electrons",
    ),
    "cond-mat.supr-con": Category(code="cond-mat.supr-con", name="Superconductivity"),
    "gr-qc": Category(
        code="gr-qc",
        name="General Relativity and Quantum Cosmology",
    ),
    "hep-ex": Category(
        code="hep-ex",
        name="High Energy Physics - Experiment",
    ),
    "hep-lat": Category(code="hep-lat", name="High Energy Physics - Lattice"),
    "hep-ph": Category(
        code="hep-ph",
        name="High Energy Physics - Phenomenology",
    ),
    "hep-th": Category(code="hep-th", name="High Energy Physics - Theory"),
    "math-ph": Category(code="math-ph", name="Mathematical Physics"),
    "nlin.AO": Category(
        code="nlin.AO",
        name="Adaptation and Self-Organizing Systems",
    ),
    "nlin.CD": Category(code="nlin.CD", name="Chaotic Dynamics"),
    "nlin.CG": Category(
        code="nlin.CG",
        name="Cellular Automata and Lattice Gases",
    ),
    "nlin.PS": Category(
        code="nlin.PS",
        name="Pattern Formation and Solitons",
    ),
    "nlin.SI": Category(
        code="nlin.SI",
        name="Exactly Solvable and Integrable Systems",
    ),
    "nucl-ex": Category(code="nucl-ex", name="Nuclear Experiment"),
    "nucl-th": Category(code="nucl-th", name="Nuclear Theory"),
    "physics.acc-ph": Category(code="physics.acc-ph", name="Accelerator Physics"),
    "physics.ao-ph": Category(
        code="physics.ao-ph",
        name="Atmospheric and Oceanic Physics",
    ),
    "physics.app-ph": Category(code="physics.app-ph", name="Applied Physics"),
    "physics.atm-clus": Category(
        code="physics.atm-clus",
        name="Atomic and Molecular Clusters",
    ),
    "physics.atom-ph": Category(code="physics.atom-ph", name="Atomic Physics"),
    "physics.bio-ph": Category(code="physics.bio-ph", name="Biological Physics"),
    "physics.chem-ph": Category(code="physics.chem-ph", name="Chemical Physics"),
    "physics.class-ph": Category(code="physics.class-ph", name="Classical Physics"),
    "physics.comp-ph": Category(code="physics.comp-ph", name="Computational Physics"),
    "physics.data-an": Category(
        code="physics.data-an",
        name="Data Analysis, Statistics and Probability",
    ),
    "physics.ed-ph": Category(code="physics.ed-ph", name="Physics Education"),
    "physics.flu-dyn": Category(code="physics.flu-dyn", name="Fluid Dynamics"),
    "physics.gen-ph": Category(code="physics.gen-ph", name="General Physics"),
    "physics.geo-ph": Category(code="physics.geo-ph", name="Geophysics"),
    "physics.hist-ph": Category(
        code="physics.hist-ph",
        name="History and Philosophy of Physics",
    ),
    "physics.ins-det": Category(
        code="physics.ins-det",
        name="Instrumentation and Detectors",
    ),
    "physics.med-ph": Category(code="physics.med-ph", name="Medical Physics"),
    "physics.optics": Category(code="physics.optics", name="Optics"),
    "physics.plasm-ph": Category(code="physics.plasm-ph", name="Plasma Physics"),
    "physics.pop-ph": Category(code="physics.pop-ph", name="Popular Physics"),
    "physics.soc-ph": Category(code="physics.soc-ph", name="Physics and Society"),
    "physics.space-ph": Category(code="physics.space-ph", name="Space Physics"),
    "quant-ph": Category(code="quant-ph", name="Quantum Physics"),
    # Quantitative Biology
    "q-bio.BM": Category(code="q-bio.BM", name="Biomolecules"),
    "q-bio.CB": Category(code="q-bio.CB", name="Cell Behavior"),
    "q-bio.GN": Category(code="q-bio.GN", name="Genomics"),
    "q-bio.MN": Category(code="q-bio.MN", name="Molecular Networks"),
    "q-bio.NC": Category(code="q-bio.NC", name="Neurons and Cognition"),
    "q-bio.OT": Category(code="q-bio.OT", name="Other Quantitative Biology"),
    "q-bio.PE": Category(code="q-bio.PE", name="Populations and Evolution"),
    "q-bio.QM": Category(code="q-bio.QM", name="Quantitative Methods"),
    "q-bio.SC": Category(code="q-bio.SC", name="Subcellular Processes"),
    "q-bio.TO": Category(code="q-bio.TO", name="Tissues and Organs"),
    # Quantitative Finance
    "q-fin.CP": Category(code="q-fin.CP", name="Computational Finance"),
    "q-fin.EC": Category(code="q-fin.EC", name="Economics"),
    "q-fin.GN": Category(code="q-fin.GN", name="General Finance"),
    "q-fin.MF": Category(code="q-fin.MF", name="Mathematical Finance"),
    "q-fin.PM": Category(code="q-fin.PM", name="Portfolio Management"),
    "q-fin.PR": Category(code="q-fin.PR", name="Pricing of Securities"),
    "q-fin.RM": Category(code="q-fin.RM", name="Risk Management"),
    "q-fin.ST": Category(code="q-fin.ST", name="Statistical Finance"),
    "q-fin.TR": Category(
        code="q-fin.TR",
        name="Trading and Market Microstructure",
    ),
    # Statistics
    "stat.AP": Category(code="stat.AP", name="Applications"),
    "stat.CO": Category(code="stat.CO", name="Computation"),
    "stat.ME": Category(code="stat.ME", name="Methodology"),
    "stat.ML": Category(code="stat.ML", name="Machine Learning"),
    "stat.OT": Category(code="stat.OT", name="Other Statistics"),
    "stat.TH": Category(code="stat.TH", name="Statistics Theory"),
}
