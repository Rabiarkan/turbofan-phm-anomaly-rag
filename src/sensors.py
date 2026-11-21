"""C-MAPSS sensors used by the detector: description, unit, engine part and handbook wording.

- description, unit: Saxena et al. (2008), Table 2. The column order in the data files is assumed
  to follow that table (src/config.py). Ps30 (about 47.5 psia) is far below P30 (about 553 psia)
  at the same station; the labels are kept as given.
- component: the engine part named in the description; None if none is named.
- terms: wording the FAA handbook (FAA-H-8083-32B) uses for the same quantity or part, each with
  the page it appears on. Terms add the handbook's vocabulary to retrieval queries. They carry no
  fault knowledge, and no term is added without a page in the indexed chapters.
"""

from typing import NamedTuple


class Sensor(NamedTuple):
    description: str  # Table 2 wording, abbreviations expanded
    unit: str  # Table 2 unit
    component: str | None
    terms: tuple[str, ...] = ()  # handbook wording not already in the description


SENSORS: dict[str, Sensor] = {
    "T24": Sensor(
        "total temperature at LPC (low-pressure compressor) outlet", "°R", "low-pressure compressor"
    ),
    "T30": Sensor(
        "total temperature at HPC (high-pressure compressor) outlet",
        "°R",
        "high-pressure compressor",
        ("compressor discharge",),  # 2-33, 2-34 (file:04_amtp_ch2.pdf, pages:33, 34)
    ),
    "T50": Sensor(
        "total temperature at LPT (low-pressure turbine) outlet",
        "°R",
        "low-pressure turbine",
        ("turbine outlet temperature",),  # 10-55
    ),
    "P30": Sensor(
        "total pressure at HPC outlet",
        "psia",
        "high-pressure compressor",
        ("compressor discharge pressure",),  # 2-33, 2-34
    ),
    "Nf": Sensor("physical fan speed", "rpm", "fan", ("low-pressure spool",)),  # 1-36
    "Nc": Sensor("physical core speed", "rpm", "core", ("high-pressure spool",)),  # 1-36
    "Ps30": Sensor(
        "static pressure at HPC outlet",
        "psia",
        "high-pressure compressor",
        ("compressor discharge pressure",),  # 2-33, 2-34
    ),
    "phi": Sensor("ratio of fuel flow to Ps30", "pps/psi", None, ("Wf/P3 ratio",)),  # 2-34
    "NRf": Sensor("corrected fan speed", "rpm", "fan", ("low-pressure spool",)),  # 1-36
    "NRc": Sensor("corrected core speed", "rpm", "core", ("high-pressure spool",)),  # 1-36
    "BPR": Sensor("bypass ratio", "--", None),
    "htBleed": Sensor("bleed enthalpy", "--", None, ("bleed air",)),  # 1-38
    "W31": Sensor(
        "HPT (high-pressure turbine) coolant bleed",
        "lbm/s",
        "high-pressure turbine",
        ("bleed cooling air",),  # 1-48
    ),
    "W32": Sensor(
        "LPT (low-pressure turbine) coolant bleed",
        "lbm/s",
        "low-pressure turbine",
        ("bleed cooling air",),  # 1-48
    ),
}


def phrase(name: str, shift_sigma: float) -> str:
    """One sensor line of a deviation report, e.g. 'T50 (total temperature ...) ↑ 3.4σ'."""
    arrow = "↑" if shift_sigma > 0 else "↓" if shift_sigma < 0 else "→"
    return f"{name} ({SENSORS[name].description}) {arrow} {abs(shift_sigma):.1f}σ"


def query_text(name: str) -> str:
    """Words for a retrieval query about one sensor: Table 2 description plus handbook terms."""
    s = SENSORS[name]
    return " ".join((s.description, *s.terms))
