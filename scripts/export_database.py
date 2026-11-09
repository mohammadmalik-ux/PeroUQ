"""Export the parameter database as a reusable data package.

    python scripts/export_database.py --out database

Writes
  database/parameters.csv, sources.csv, decisions.csv, calibration_devices.csv, nk_sources.csv
      (copies of perouq/params, the files the code reads)
  database/parameter_summary.csv   one row per material and parameter: number of values and
      sources, min / median / max, unit, sampling scale and the range actually sampled in the
      study (with the minimum-uncertainty rule)
  database/PeroUQ_database.xlsx    all of the above as sheets, plus a data dictionary
  database/README.md               description of every file and column; licence CC BY 4.0
"""
import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perouq import params                                     # noqa: E402
from perouq.params import DEFAULT_MIN_SPREAD, LOG_PARAMS, NOT_SAMPLED   # noqa: E402

SRC = os.path.join(os.path.dirname(params.__file__))
FILES = ["parameters.csv", "sources.csv", "decisions.csv", "calibration_devices.csv", "nk_sources.csv"]
DICT = [
    ("parameters.csv", "material", "Layer or interface: MAPbI3, TiO2, SnO2, ZnO, CdS, ETL/MAPbI3 (interface), carbon"),
    ("parameters.csv", "param", "Eg band gap; chi electron affinity; eps_r relative permittivity; Nc, Nv effective "
     "densities of states; mu_n, mu_p mobilities; ND, NA doping; Nt bulk defect density; sigma capture "
     "cross-section; vth thermal velocity; S interface recombination velocity; W work function"),
    ("parameters.csv", "value", "Numerical value as reported by the source"),
    ("parameters.csv", "unit", "Unit of value"),
    ("parameters.csv", "source", "Key into sources.csv"),
    ("parameters.csv", "note", "Remarks (e.g. measured vs simulation input, page or table)"),
    ("sources.csv", "id", "Source key"),
    ("sources.csv", "kind", "measurement or simulation-input"),
    ("sources.csv", "citation / doi / url", "Bibliographic reference"),
    ("decisions.csv", "material, param", "What the decision applies to"),
    ("decisions.csv", "action, arg", "exclude_source / only_sources and the source keys concerned"),
    ("decisions.csv", "reason, date", "Why, and when it was recorded (author's decisions)"),
    ("calibration_devices.csv", "*", "Published devices used for calibration or validation and their metrics"),
    ("nk_sources.csv", "*", "Candidate sources of measured optical constants and their status"),
    ("parameter_summary.csv", "n_values, n_sources", "Number of values and of distinct sources after the decisions"),
    ("parameter_summary.csv", "min, median, max", "Spread of the retained values"),
    ("parameter_summary.csv", "scale", "log (sampled log-uniformly) or linear (uniform)"),
    ("parameter_summary.csv", "sampled_low, sampled_high", "Range sampled in the study (minimum-uncertainty rule "
     "x/3 for log parameters, +/-0.1 eV for energies, +/-10 % for eps_r); empty = held fixed"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="database")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in FILES:
        shutil.copy(os.path.join(SRC, f), os.path.join(a.out, f))
    mats = []
    with open(os.path.join(SRC, "parameters.csv")) as f:
        for r in csv.DictReader(f):
            if r["material"] not in mats:
                mats.append(r["material"])
    rows = []
    for m in mats:
        t = params.table(m)
        rng = params.ranges(m, min_spread=DEFAULT_MIN_SPREAD)
        for p, e in t.items():
            lo, hi = (rng[p][0], rng[p][1]) if p in rng and rng[p][0] != rng[p][1] else ("", "")
            rows.append(dict(material=m, param=p, unit=e["unit"], n_values=e["n"], n_sources=e["n_sources"],
                             min=e["min"], median=e["median"], max=e["max"],
                             scale="log" if p in LOG_PARAMS else "linear",
                             sampled_low=lo, sampled_high=hi,
                             sources=";".join(sorted(set(e["sources"]))),
                             note="physical constant, not sampled" if p in NOT_SAMPLED else ""))
    with open(os.path.join(a.out, "parameter_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import openpyxl
    from openpyxl.styles import Font
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "README"
    ws.append(["PeroUQ parameter database", "CC BY 4.0", "M. M. Abood, University of Alkafeel"])
    ws.append([])
    ws.append(["file / sheet", "column", "meaning"])
    for r in DICT:
        ws.append(list(r))
    for c in ws[1] + ws[3]:
        c.font = Font(bold=True)
    for name in ["parameter_summary.csv"] + FILES:
        sh = wb.create_sheet(name.replace(".csv", "")[:31])
        with open(os.path.join(a.out, name)) as f:
            for i, r in enumerate(csv.reader(f)):
                sh.append([float(x) if _num(x) else x for x in r])
        for c in sh[1]:
            c.font = Font(bold=True)
        sh.freeze_panes = "A2"
    wb.save(os.path.join(a.out, "PeroUQ_database.xlsx"))
    n_val = sum(1 for _ in open(os.path.join(SRC, "parameters.csv"))) - 1
    n_src = sum(1 for _ in open(os.path.join(SRC, "sources.csv"))) - 1
    md = ["# PeroUQ parameter database", "",
          f"{n_val} parameter values from {n_src} sources for MAPbI3, TiO2, SnO2, ZnO, CdS, the ETL/MAPbI3 interface "
          "and the carbon electrode, as used in the PeroUQ study. Licence: CC BY 4.0 (data); the code is MIT.", "",
          "| File | Content |", "|---|---|",
          "| parameters.csv | every value with unit, source key and note (one row per value) |",
          "| sources.csv | the sources, with kind (measurement or simulation input), citation, DOI and URL |",
          "| decisions.csv | the author's decisions applied before sampling (excluded or restricted sources) |",
          "| calibration_devices.csv | published devices and their metrics |",
          "| nk_sources.csv | candidate sources of measured n,k data and their status |",
          "| parameter_summary.csv | one row per material and parameter: spread, scale and sampled range |",
          "| PeroUQ_database.xlsx | the same tables as sheets, with a data dictionary |", "",
          "## Columns", "", "| File | Column | Meaning |", "|---|---|---|"]
    md += [f"| {a_} | {b} | {c} |" for a_, b, c in DICT]
    md += ["", "The code reads the copies in `perouq/params/`; `python scripts/export_database.py` regenerates "
           "this folder from them."]
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"{len(rows)} parameter rows, {n_val} values, {n_src} sources -> {a.out}")


def _num(x):
    try:
        float(x)
        return x.strip() != ""
    except ValueError:
        return False


if __name__ == "__main__":
    main()
