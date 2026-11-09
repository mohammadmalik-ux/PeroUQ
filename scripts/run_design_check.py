"""Re-evaluate the robust designs with a large common sample.

    python scripts/run_design_check.py --spectrum data/ASTMG173.csv --W 4.95 5.35 --min-spread \
        --optics tmm --n 500 --designs results/design_v2 --out results/design_check

The Bayesian-optimisation search (run_design.py) uses 24 common parameter samples per
design.  Here, for each ETL, four designs are evaluated on the same n = 500
Latin-hypercube samples (common random numbers):
  opt   - the BO optimum (ETL and absorber thickness),
  thin  - the optimum ETL thickness with a 450 nm absorber,
  thick - the optimum ETL thickness with an 800 nm absorber,
  ref   - the reference design (50 nm ETL, 500 nm absorber).
F = mean - lambda * std is reported with a bootstrap interval, together with its
running value for n = 24, 50, 100, 200, 500 (convergence) and the paired
bootstrap interval of F(opt) - F(other).
Output: OUT/design_check.json
"""
import argparse
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


def F(x, lam):
    return float(np.mean(x) - lam * np.std(x, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--W", type=float, nargs=2)
    ap.add_argument("--eta-opt", type=float)
    ap.add_argument("--optics", choices=["bl", "tmm"], default="tmm")
    ap.add_argument("--min-spread", action="store_true")
    ap.add_argument("--designs", default="results/design_v2")
    ap.add_argument("--main", help="mc_<ETL>.csv of the main run (50/500 design, same seed and n), reused for 'ref'")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/design_check")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    extra = default_extra()
    if a.W:
        extra = [p for p in extra if p.name != "W"] + [calib.Param("W", a.W[0], a.W[1])]
    circuit = {} if a.eta_opt is None else {"eta_opt": a.eta_opt}
    ms = DEFAULT_MIN_SPREAD if a.min_spread else None
    jpath = os.path.join(a.out, "design_check.json")
    out = json.load(open(jpath)) if os.path.exists(jpath) else dict(warning=tag, optics=a.optics, n=a.n, etl={})
    rng = np.random.default_rng(0)
    for etl in a.etl:
        if etl in out["etl"]:
            continue
        best = json.load(open(os.path.join(a.designs, f"design_{etl}.json")))["best"]
        designs = {"opt": (best["etl_nm"], best["pero_nm"]), "thin": (best["etl_nm"], 450.0),
                   "thick": (best["etl_nm"], 800.0), "ref": (50.0, 500.0)}
        res = {}
        for k, (de, dp) in designs.items():
            mpath = os.path.join(a.main, f"mc_{etl}.csv") if a.main else ""
            if k == "ref" and os.path.exists(mpath):
                import csv
                with open(mpath) as f:
                    res[k] = np.array([float(r["PCE"]) if r["PCE"] else np.nan for r in csv.DictReader(f)])[:a.n]
                continue
            cell = make_cell(a.optics, etl, lam, E, etl_nm=de, pero_nm=dp, extra=extra, min_spread=ms, **circuit)
            U = qmc.LatinHypercube(cell.d, seed=0).random(a.n)
            res[k] = uq._column(uq.evaluate(cell, U, a.jobs), "PCE")
        ok = np.all([np.isfinite(v) for v in res.values()], axis=0)
        e = dict(designs=designs, n_ok=int(ok.sum()), stats={}, convergence={}, paired_vs_opt={})
        for k, v in res.items():
            x = v[ok]
            boot = [F(x[rng.integers(0, len(x), len(x))], a.lam) for _ in range(2000)]
            e["stats"][k] = dict(mean=float(x.mean()), std=float(x.std(ddof=1)), P5=float(np.percentile(x, 5)),
                                 F=F(x, a.lam), F_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
            e["convergence"][k] = {str(n): F(v[:n][np.isfinite(v[:n])], a.lam) for n in (24, 50, 100, 200, a.n)}
        xo = res["opt"][ok]
        for k in ("thin", "thick", "ref"):
            xk = res[k][ok]
            dif = []
            for _ in range(2000):
                i = rng.integers(0, len(xo), len(xo))
                dif.append(F(xo[i], a.lam) - F(xk[i], a.lam))
            e["paired_vs_opt"][k] = dict(dF=F(xo, a.lam) - F(xk, a.lam),
                                         ci=[float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))])
        out["etl"][etl] = e
        print(etl, json.dumps(e), flush=True)
        with open(jpath, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
