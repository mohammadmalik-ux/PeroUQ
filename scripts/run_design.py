"""Robust thickness design for each ETL.

    python scripts/run_design.py --spectrum data/ASTMG173.csv --W 4.95 5.35 --eta-opt 0.95 --min-spread

For each ETL, Bayesian optimisation over ETL thickness and absorber thickness
maximises mean(PCE) - lambda * std(PCE) over the literature uncertainty of the
material parameters (common random numbers for all designs).
Writes results/design/design_<ETL>.json and design_figure.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum        # noqa: E402
from perouq import calib                                  # noqa: E402
from perouq.design import RobustDesigner                  # noqa: E402
from perouq.params import DEFAULT_MIN_SPREAD              # noqa: E402
from perouq.study import default_extra, make_cell           # noqa: E402


class Factory:
    def __init__(self, etl, lam_nm, E, extra, eta, ms, optics="bl"):
        self.etl, self.lam, self.E, self.extra, self.eta, self.ms = etl, lam_nm, E, extra, eta, ms
        self.optics = optics

    def __call__(self, d):
        return make_cell(self.optics, self.etl, self.lam, self.E, etl_nm=d["etl_nm"], pero_nm=d["pero_nm"],
                         extra=self.extra, eta_opt=self.eta, min_spread=self.ms)


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--W", type=float, nargs=2)
    ap.add_argument("--eta-opt", type=float, default=1.0)
    ap.add_argument("--min-spread", action="store_true")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n-inner", type=int, default=24)
    ap.add_argument("--n-init", type=int, default=6)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--optics", choices=["bl", "tmm"], default="bl")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/design_v3")
    ap.add_argument("--resume", action="store_true", help="skip ETLs with an existing design_<ETL>.json")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    extra = default_extra()
    if a.W:
        extra = [p for p in extra if p.name != "W"] + [calib.Param("W", a.W[0], a.W[1])]
    ms = DEFAULT_MIN_SPREAD if a.min_spread else None
    dps = [calib.Param("etl_nm", 20.0, 100.0), calib.Param("pero_nm", 300.0, 900.0)]
    allres = {}
    for etl in a.etl:
        jpath = os.path.join(a.out, f"design_{etl}.json")
        if a.resume and os.path.exists(jpath):
            with open(jpath) as f:
                allres[etl] = json.load(f)
            print(f"{etl}: already done, skipped (--resume)")
            continue
        t0 = time.time()
        fac = Factory(etl, lam, E, extra, a.eta_opt, ms, a.optics)
        d_unc = fac({"etl_nm": 50.0, "pero_nm": 500.0}).d
        rd = RobustDesigner(dps, fac, d_uncertain=d_unc, n_inner=a.n_inner, lam=a.lam, n_jobs=a.jobs, seed=0)
        r = rd.run(n_init=a.n_init, n_iter=a.n_iter, verbose=True)
        out = dict(etl=etl, warning=tag, lam=a.lam, best=r["best"], best_F=r["best_F"], best_stats=r["best_stats"],
                   designs=r["designs"], F=list(map(float, r["F"])), stats=r["stats"],
                   surface=None if r["surface"] is None else r["surface"]["F"].tolist(),
                   seconds=round(time.time() - t0))
        with open(os.path.join(a.out, f"design_{etl}.json"), "w") as f:
            json.dump(out, f, indent=1)
        allres[etl] = out
        print(etl, "best", r["best"], "mean", r["best_stats"]["mean"], "std", r["best_stats"]["std"])
    figure(allres, dps, a.out, tag)


def redraw(out):
    """Redraw the figure from saved design_<ETL>.json files (python -c 'import run_design; run_design.redraw(DIR)')."""
    allres = {}
    for etl in ("TiO2", "SnO2", "ZnO", "CdS"):
        f = os.path.join(out, f"design_{etl}.json")
        if os.path.exists(f):
            allres[etl] = json.load(open(f))
    figure(allres, [calib.Param("etl_nm", 20.0, 100.0), calib.Param("pero_nm", 300.0, 900.0)], out, "")


def figure(allres, dps, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(allres)
    ncol = 2 if n == 4 else n
    nrow = (n + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow), squeeze=False)
    sub = {"TiO2": "TiO$_2$", "SnO2": "SnO$_2$", "ZnO": "ZnO", "CdS": "CdS"}
    for k, (etl, r) in enumerate(allres.items()):
        a = ax[k // ncol, k % ncol]
        if r["surface"] is not None:
            F = np.array(r["surface"])
            x = np.linspace(dps[0].low, dps[0].high, F.shape[1])
            y = np.linspace(dps[1].low, dps[1].high, F.shape[0])
            cs = a.contourf(x, y, F, levels=15, cmap="viridis")
            fig.colorbar(cs, ax=a, label=r"F = mean $-$ $\lambda\,$std of PCE (%)")
        xs = [d["etl_nm"] for d in r["designs"]]
        ys = [d["pero_nm"] for d in r["designs"]]
        a.plot(xs, ys, "w.", ms=5)
        a.plot(r["best"]["etl_nm"], r["best"]["pero_nm"], "r*", ms=14)
        s = r["best_stats"]
        a.set_title(f"({'abcdefgh'[k]}) {sub.get(etl, etl)}: optimum {r['best']['etl_nm']:.0f} / {r['best']['pero_nm']:.0f} nm, "
                    f"PCE {s['mean']:.1f} $\\pm$ {s['std']:.1f} %", fontsize=10)
        a.set_xlabel("ETL thickness (nm)"); a.set_ylabel("MAPbI$_3$ thickness (nm)")
    if tag:
        fig.suptitle(tag, color="red")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "design_figure.png"), dpi=300)


if __name__ == "__main__":
    main()
