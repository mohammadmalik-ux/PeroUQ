"""Literature parameter database: every value with its source.

Files
-----
parameters.csv : material, param, value, unit, source, note
sources.csv    : id, kind, citation, doi, url

The spread of values between sources is the uncertainty range used by the
Monte Carlo and Sobol modules.  Densities, mobilities, defect densities and
capture cross-sections vary over orders of magnitude and are sampled
log-uniformly; energies and permittivities uniformly.
"""
import csv
import os
from collections import defaultdict

import numpy as np

from ..device import Material

_HERE = os.path.dirname(__file__)
LOG_PARAMS = {"Nc", "Nv", "mu_n", "mu_p", "ND", "NA", "Nt", "sigma", "S"}
ENERGY_PARAMS = {"Eg", "chi", "W"}
NOT_SAMPLED = {"vth"}          # thermal velocity: a physical constant, not a material uncertainty
DEFAULT_MIN_SPREAD = {"log": 3.0, "energy": 0.1, "rel": 0.1}
MATERIAL_FIELDS = ("Eg", "chi", "eps_r", "Nc", "Nv", "mu_n", "mu_p", "ND", "NA")
# used when a layer has a defect density but no capture cross-section / thermal velocity
DEFAULT_SIGMA = 2e-14   # cm^2  (Subudhi2023, applied to all layers in that work)
DEFAULT_VTH = 1e7       # cm/s  (Subudhi2023, Son2024)


def load_decisions(path=None):
    """Author's decisions on which literature values enter the study (decisions.csv)."""
    path = path or os.path.join(_HERE, "decisions.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def apply_decisions(rows, decisions):
    """exclude_source: drop that source for (material, param);
    only_sources: keep only the listed sources (separated by ';')."""
    out = []
    for r in rows:
        keep = True
        for d in decisions:
            if d["material"] != r["material"] or d["param"] != r["param"]:
                continue
            srcs = {s.strip() for s in d["arg"].split(";")}
            if d["action"] == "exclude_source" and r["source"] in srcs:
                keep = False
            elif d["action"] == "only_sources" and r["source"] not in srcs:
                keep = False
        if keep:
            out.append(r)
    return out


def load_rows(path=None, decisions=True):
    """All literature rows; by default with the author's decisions applied.
    decisions=False returns the raw literature data."""
    path = path or os.path.join(_HERE, "parameters.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["value"] = float(r["value"])
    if decisions:
        rows = apply_decisions(rows, load_decisions())
    return rows


def load_sources(path=None):
    path = path or os.path.join(_HERE, "sources.csv")
    with open(path, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def table(material, rows=None):
    """param -> dict(values, sources, unit, min, max, median, n) for one material."""
    rows = rows if rows is not None else load_rows()
    d = defaultdict(lambda: dict(values=[], sources=[], unit=None))
    for r in rows:
        if r["material"] != material:
            continue
        e = d[r["param"]]
        e["values"].append(r["value"])
        e["sources"].append(r["source"])
        e["unit"] = r["unit"]
    out = {}
    for p, e in d.items():
        v = np.array(e["values"])
        out[p] = dict(values=v, sources=e["sources"], unit=e["unit"],
                      min=v.min(), max=v.max(), median=float(np.median(v)),
                      n=len(v), n_sources=len(set(e["sources"])))
    return out


def _tau(Nt, sigma, vth):
    return 1.0 / (sigma * vth * Nt)


def material(name, values=None, alpha=None, rows=None):
    """Build a Material from the database.

    values: optional dict param -> value overriding the median (used by the
    samplers).  Lifetimes follow from Nt, sigma and vth: tau = 1/(sigma vth Nt).
    """
    t = table(name, rows)
    if not t:
        raise KeyError(f"no parameters for material '{name}'")
    vals = {p: e["median"] for p, e in t.items()}
    if values:
        vals.update(values)
    kw = {k: vals.get(k, 0.0) for k in MATERIAL_FIELDS}
    sigma = vals.get("sigma", DEFAULT_SIGMA)
    vth = vals.get("vth", DEFAULT_VTH)
    tau = _tau(vals["Nt"], sigma, vth) if "Nt" in vals else 1e-6
    return Material(name, tau_n=tau, tau_p=tau, alpha=alpha, **kw)


def ranges(name, rows=None, min_sources=1, min_spread=None):
    """param -> (low, high, log) for parameters reported by >= min_sources sources.

    min_spread: optional minimum uncertainty, so that a parameter reported by
    one source (or identically by all) is not treated as exactly known while
    other materials carry the literature spread.  Dict with
      'log'    multiplicative half-width for log parameters (3 -> [v/3, 3v]),
      'energy' half-width in eV for Eg, chi, W,
      'rel'    relative half-width for the rest (eps_r).
    A range already wider than the minimum is left unchanged.
    """
    out = {}
    for p, e in table(name, rows).items():
        if e["n_sources"] < min_sources or p in NOT_SAMPLED:
            continue
        lo, hi, lg = e["min"], e["max"], p in LOG_PARAMS
        if min_spread:
            # widen only a range that is narrower than the minimum; a reported spread that is
            # already wider (e.g. MAPbI3 eps_r 6.5-10, the author's bounds) is kept as it is
            mid = e["median"]
            if lg:
                f = min_spread.get("log", 1.0)
                if np.log10(hi) - np.log10(lo) < 2 * np.log10(f):
                    lo, hi = min(lo, mid / f), max(hi, mid * f)
            elif p in ENERGY_PARAMS:
                de = min_spread.get("energy", 0.0)
                if hi - lo < 2 * de:
                    lo, hi = min(lo, mid - de), max(hi, mid + de)
            else:
                r = min_spread.get("rel", 0.0)
                if hi - lo < 2 * r * abs(mid):
                    lo, hi = min(lo, mid * (1 - r)), max(hi, mid * (1 + r))
        out[p] = (lo, hi, lg)
    return out


def sample_values(name, u, rows=None, min_spread=None):
    """Map a vector u in [0,1]^k (one entry per varied parameter, sorted by name)
    to parameter values.  Returns (dict, list of varied parameter names)."""
    rg = {p: r for p, r in ranges(name, rows, min_spread=min_spread).items() if r[0] != r[1]}
    names = sorted(rg)
    vals = {}
    for p, ui in zip(names, u):
        lo, hi, lg = rg[p]
        vals[p] = float(10 ** (np.log10(lo) + ui * (np.log10(hi) - np.log10(lo)))) if lg \
            else float(lo + ui * (hi - lo))
    return vals, names


def summary_markdown(materials=("MAPbI3", "TiO2", "SnO2", "ZnO", "CdS", "FTO", "ITO", "carbon",
                                "ETL/MAPbI3")):
    """Markdown table of the ranges, for reports."""
    lines = ["| Material | Parameter | Unit | Min | Median | Max | Sources |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in materials:
        for p, e in sorted(table(m).items()):
            lines.append(f"| {m} | {p} | {e['unit']} | {e['min']:.3g} | {e['median']:.3g} | "
                         f"{e['max']:.3g} | {', '.join(sorted(set(e['sources'])))} |")
    return "\n".join(lines)
