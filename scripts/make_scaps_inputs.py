"""Input tables for a SCAPS-1D cross-check, and the matching PeroUQ results.

    python scripts/make_scaps_inputs.py --spectrum data/ASTMG173.csv --out scaps

For each ETL the median device of the database (50 nm ETL, 500 nm MAPbI3,
S = median, W = 5.15 eV) is written as a SCAPS layer table, and PeroUQ is run on
exactly that device with SCAPS-like settings: Beer-Lambert absorption with
alpha = A sqrt(hv - Eg) (A = 1e5 cm^-1 eV^-1/2), no reflection, eta_opt = 1,
no radiative recombination, no series or shunt resistance.
Then fill scaps/scaps_results.csv with the SCAPS output and run
    python scripts/make_scaps_inputs.py --compare scaps/scaps_results.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum      # noqa: E402
from perouq import params, optics                        # noqa: E402
from perouq.params import DEFAULT_SIGMA, DEFAULT_VTH     # noqa: E402

ETLS = ["TiO2", "SnO2", "ZnO", "CdS"]
W_CARBON = 5.15
FIELDS = ["Eg", "chi", "eps_r", "Nc", "Nv", "mu_n", "mu_p", "ND", "NA", "Nt"]
UNITS = dict(Eg="eV", chi="eV", eps_r="-", Nc="cm-3", Nv="cm-3", mu_n="cm2/Vs", mu_p="cm2/Vs",
             ND="cm-3", NA="cm-3", Nt="cm-3")


def compare(path, ref_path):
    ref = {r["ETL"]: r for r in csv.DictReader(open(ref_path))}
    rows = list(csv.DictReader(open(path)))
    print("ETL   quantity   PeroUQ     SCAPS     difference")
    for r in rows:
        for k in ("Jsc", "Voc", "FF", "PCE"):
            if r.get(k):
                a, b = float(ref[r["ETL"]][k]), float(r[k])
                print(f"{r['ETL']:5s} {k:8s} {a:9.4f} {b:9.4f} {100 * (a - b) / b:+8.2f} %")


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--out", default="scaps")
    ap.add_argument("--compare")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare, os.path.join(os.path.dirname(a.compare), "peroUQ_reference.csv"))
        return
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    S = params.table("ETL/MAPbI3")["S"]["median"]
    sigma, vth = DEFAULT_SIGMA, DEFAULT_VTH
    md = ["# SCAPS-1D input for the PeroUQ cross-check", "",
          "Structure (illuminated from the left): FTO (front contact, flat band, S = 1e7 cm/s) / ETL 50 nm / "
          f"MAPbI3 500 nm / carbon (back contact, work function {W_CARBON} eV, S = 1e7 cm/s).", "",
          "Settings: AM1.5G 1000 W/m2, 300 K, no reflection, no series or shunt resistance; radiative "
          "recombination off; absorption model sqrt(hv - Eg) with A = 1e5 cm^-1 eV^-1/2 for every layer; "
          f"one neutral midgap bulk defect per layer with sigma_n = sigma_p = {sigma:g} cm2 and "
          f"vth = {vth:g} cm/s; interface ETL/MAPbI3: neutral midgap defect with "
          f"N_if = S / (sigma vth) = {S / (sigma * vth):.3g} cm-2 (S = {S:g} cm/s).", ""]
    rows_csv = []
    ref = []
    for etl in ETLS:
        tp, te = params.table("MAPbI3"), params.table(etl)
        md += [f"## {etl}", "", "| Parameter | Unit | ETL (" + etl + ") | MAPbI3 |", "|---|---|---|---|"]
        for k in FIELDS:
            ve = te.get(k, {}).get("median", 0.0)
            vp = tp.get(k, {}).get("median", 0.0)
            md.append(f"| {k} | {UNITS[k]} | {ve:.4g} | {vp:.4g} |")
            rows_csv.append([etl, k, UNITS[k], ve, vp])
        md.append("")
        # the median device: every uncertain parameter at its database median
        from perouq.device import Layer, Contact, Interface, Device
        from perouq import jv
        pero = params.material("MAPbI3", alpha=optics.alpha_sqrt_model(tp["Eg"]["median"], Eu=1e-3))
        etlm = params.material(etl, alpha=optics.alpha_sqrt_model(te["Eg"]["median"], A=1e5, Eu=1e-3))
        dev = Device([Layer(etlm, 50.0, 50), Layer(pero, 500.0, 120)], left=Contact(None),
                     right=Contact(W_CARBON), interfaces=[Interface(0, S, S)])
        jr = jv.light_jv(dev, lam, E)
        m = jv.metrics(jr["V"], jr["J"], 1000.0)
        ref.append(dict(ETL=etl, Jsc=m["Jsc"] / 10, Voc=m["Voc"], FF=m["FF"], PCE=100 * m["PCE"]))
    with open(os.path.join(a.out, "scaps_inputs.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(a.out, "scaps_inputs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ETL", "parameter", "unit", "ETL_value", "MAPbI3_value"])
        w.writerows(rows_csv)
    with open(os.path.join(a.out, "peroUQ_reference.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ETL", "Jsc", "Voc", "FF", "PCE"])
        w.writeheader()
        w.writerows(ref)
    with open(os.path.join(a.out, "scaps_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ETL", "Jsc", "Voc", "FF", "PCE"])
        for etl in ETLS:
            w.writerow([etl, "", "", "", ""])
    print(json.dumps(ref, indent=1))


if __name__ == "__main__":
    main()
