from enum import Enum


class PaperDomain(Enum):
    CS = "Computer Science"
    ECON = "Economics"
    MATH = "Mathematics"
    PHYSICS = "Physics"
    Q_BIO = "Quantitative Biology"
    Q_FIN = "Quantitative Finance"
    STAT = "Statistics"
    EESS = "Electrical Engineering"

    UNKNOWN = "Unknown"

    @property
    def label(self):
        return self.value


PHYSICS_PREFIXES = (
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math-ph",
    "nlin",
    "nucl-ex",
    "nucl-th",
    "physics",
    "quant-ph",
)


def infer_domain(code: str) -> PaperDomain:
    prefix = code.split(".")[0]

    if prefix == "cs":
        return PaperDomain.CS
    if prefix == "math":
        return PaperDomain.MATH
    if prefix == "econ":
        return PaperDomain.ECON
    if prefix == "q-bio":
        return PaperDomain.Q_BIO
    if prefix == "q-fin":
        return PaperDomain.Q_FIN
    if prefix == "stat":
        return PaperDomain.STAT
    if prefix == "eess":
        return PaperDomain.EESS

    if any(code.startswith(p) for p in PHYSICS_PREFIXES):
        return PaperDomain.PHYSICS

    return PaperDomain.UNKNOWN
