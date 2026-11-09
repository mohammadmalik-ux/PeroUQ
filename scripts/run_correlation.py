"""Effect of correlations between input parameters (Gaussian copula).

    python scripts/run_correlation.py --spectrum data/ASTMG173.csv --W 4.95 5.35 --min-spread \
        --optics tmm --n 500 --out results/correlation

The main study samples all parameters independently.  Physically motivated
correlations are imposed here through a Gaussian copula on the unit cube (the
marginal distributions stay the same):
  MAPbI3 chi  - MAPbI3 Eg   negative  (sources agree better on the ionisation energy
                                       chi + Eg than on chi and Eg separately)
  MAPbI3 Nt   - MAPbI3 mu_n, mu_p  negative (defective films have lower mobility)
  MAPbI3 Nt   - S           positive  (bulk and interface defects share the processing)
  mu_n        - mu_p        positive
with |rho| = 0.5 ('moderate') and 0.8 ('strong').  The same Latin-hypercube points
(seed 0) are used with and without correlation, so differences are not sampling noise.
Output: OUT/correlation.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import norm, qmc

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum      # noqa: E402
from perouq import calib, uq                             # noqa: E402
from perouq.params import DEFAULT_MIN_SPREAD             # noqa: E402
from perouq.study import default_extra, make_cell        # noqa: E402

PAIRS = [("MAPbI3.chi", "MAPbI3.Eg", -1), ("MAPbI3.Nt", "MAPbI3.mu_n", -1), ("MAPbI3.Nt", "MAPbI3.mu_p", -1),
         ("MAPbI3.Nt", "S", +1), ("MAPbI3.mu_n", "MAPbI3.mu_p", +1)]


def corr_matrix(names, rho):
    R = np.eye(len(names))
    for a, b, s in PAIRS:
        if a in names and b in names:
            i, j = names.index(a), names.index(b)
            R[i, j] = R[j, i] = s * rho
    w, V = np.linalg.eigh(R)
    if w.min() <= 0:                      # keep it a valid correlation matrix
        w = np.clip(w, 1e-3, None)
        R = V @ np.diag(w) @ V.T
        dd = np.sqrt(np.diag(R))
        R = R / np.outer(dd, dd)
    return R


def correlate(U, R):
    Z = norm.ppf(np.clip(U, 1e-9, 1 - 1e-9))
    Zc = Z @ np.linalg.cholesky(R).T
    return norm.cdf(Zc)


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--W", type=float, nargs=2)
    ap.add_argument("--eta-opt", type=float)
    ap.add_argument("--optics", choices=["bl", "tmm"], default="tmm")
    ap.add_argument("--min-spread", action="store_true")
    ap.add_argument("--main", help="directory with mc_<ETL>.csv of the uncorrelated run (same seed, n)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/correlation")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    extra = default_extra()
    if a.W:
        extra = [p for p in extra if p.name != "W"] + [calib.Param("W", a.W[0], a.W[1])]
    circuit = {} if a.eta_opt is None else {"eta_opt": a.eta_opt}
    ms = DEFAULT_MIN_SPREAD if a.min_spread else None
    out = dict(warning=tag, optics=a.optics, pairs=PAIRS, n=a.n, etl={})
    jpath = os.path.join(a.out, "correlation.json")
    if os.path.exists(jpath):
        out["etl"] = json.load(open(jpath))["etl"]
    for etl in a.etl:
        if etl in out["etl"]:
            continue
        cell = make_cell(a.optics, etl, lam, E, extra=extra, min_spread=ms, **circuit)
        U = qmc.LatinHypercube(cell.d, seed=0).random(a.n)
        res = {}
        if a.main and os.path.exists(os.path.join(a.main, f"mc_{etl}.csv")):
            import csv
            with open(os.path.join(a.main, f"mc_{etl}.csv")) as f:
                res["independent"] = np.array([float(r["PCE"]) if r["PCE"] else np.nan for r in csv.DictReader(f)])
        else:
            res["independent"] = uq._column(uq.evaluate(cell, U, a.jobs), "PCE")
        for label, rho in (("moderate", 0.5), ("strong", 0.8)):
            R = corr_matrix(cell.names, rho)
            res[label] = uq._column(uq.evaluate(cell, correlate(U, R), a.jobs), "PCE")
        e = {}
        for k, v in res.items():
            v = v[np.isfinite(v)]
            e[k] = dict(n=len(v), P5=float(np.percentile(v, 5)), median=float(np.median(v)),
                        P95=float(np.percentile(v, 95)), P15=float(np.mean(v > 15)), P20=float(np.mean(v > 20)))
        out["etl"][etl] = e
        print(etl, json.dumps(e), flush=True)
        with open(jpath, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
