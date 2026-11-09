"""Evaluate extra designs on the same 500 common samples as run_design_check.py and add them.

    python scripts/run_design_eval.py --spectrum data/ASTMG173.csv --W 4.95 5.35 --min-spread \
        --optics tmm --designs results/design_v3 --label bo_tmm --check results/design_check

For each ETL the BO optimum stored in DESIGNS/design_<ETL>.json is evaluated (unless it
coincides with a design already in the check) and added to design_check.json under LABEL,
with its F, bootstrap interval and the paired difference from every other design.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum      # noqa: E402
from perouq import calib, uq                             # noqa: E402
from perouq.params import DEFAULT_MIN_SPREAD             # noqa: E402
from perouq.study import default_extra, make_cell        # noqa: E402
from run_design_check import F                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--W", type=float, nargs=2)
    ap.add_argument("--optics", choices=["bl", "tmm"], default="tmm")
    ap.add_argument("--min-spread", action="store_true")
    ap.add_argument("--designs", default="results/design_v3")
    ap.add_argument("--label", default="bo_tmm")
    ap.add_argument("--check", default="results/design_check")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    extra = default_extra()
    if a.W:
        extra = [p for p in extra if p.name != "W"] + [calib.Param("W", a.W[0], a.W[1])]
    ms = DEFAULT_MIN_SPREAD if a.min_spread else None
    jpath = os.path.join(a.check, "design_check.json")
    chk = json.load(open(jpath))
    rng = np.random.default_rng(1)
    for etl in a.etl:
        e = chk["etl"][etl]
        if a.label in e["stats"]:
            continue
        best = json.load(open(os.path.join(a.designs, f"design_{etl}.json")))["best"]
        de, dp = best["etl_nm"], best["pero_nm"]
        same = [k for k, (x, y) in e["designs"].items() if abs(x - de) < 1.0 and abs(y - dp) < 5.0]
        e["designs"][a.label] = (de, dp)
        cache = os.path.join(a.check, f"pce_{etl}_{a.label}.npy")
        if same:
            e["stats"][a.label] = dict(e["stats"][same[0]], same_as=same[0])
            print(etl, "same as", same[0])
        else:
            if os.path.exists(cache):
                x = np.load(cache)
            else:
                cell = make_cell(a.optics, etl, lam, E, etl_nm=de, pero_nm=dp, extra=extra, min_spread=ms)
                U = qmc.LatinHypercube(cell.d, seed=0).random(a.n)
                x = uq._column(uq.evaluate(cell, U, a.jobs), "PCE")
                np.save(cache, x)
            xs = x[np.isfinite(x)]
            boot = [F(xs[rng.integers(0, len(xs), len(xs))], a.lam) for _ in range(2000)]
            e["stats"][a.label] = dict(mean=float(xs.mean()), std=float(xs.std(ddof=1)), P5=float(np.percentile(xs, 5)),
                                       F=F(xs, a.lam), F_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
            e["convergence"][a.label] = {str(n): F(x[:n][np.isfinite(x[:n])], a.lam) for n in (24, 50, 100, 200, a.n)}
            # paired differences need the other designs' per-sample values: recompute ref from the main run
            ref = np.array([float(r["PCE"]) if r["PCE"] else np.nan
                            for r in csv.DictReader(open(f"results/uq_tmm/mc_{etl}.csv"))])[:a.n]
            ok = np.isfinite(x) & np.isfinite(ref)
            d = [F(x[ok][i], a.lam) - F(ref[ok][i], a.lam) for i in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(2000))]
            e.setdefault("paired_vs_ref", {})[a.label] = dict(dF=F(x[ok], a.lam) - F(ref[ok], a.lam),
                                                              ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        print(etl, json.dumps(e["stats"][a.label]), flush=True)
        with open(jpath, "w") as f:
            json.dump(chk, f, indent=1)


if __name__ == "__main__":
    main()
