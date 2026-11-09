"""Temperature dependence of the efficiency for each ETL.

    python scripts/run_temperature.py --spectrum data/ASTMG173.csv --W 4.95 5.35 --min-spread \
        --optics tmm --n 100 --out results/temperature

The first n Latin-hypercube samples of the main study are simulated at
T = 275, 300, 325 and 350 K (MAPbI3 stays tetragonal over this range) in two models:
  kT    - only the thermal voltage changes (as in the solver's T argument);
  full  - in addition Nc, Nv of both layers scale as (T/300)^1.5, the MAPbI3 carrier
          mobilities as (T/300)^-1.5 (band-like transport, Milot et al. 2015) and the
          MAPbI3 band gap shifts by dEg/dT = +0.3 meV/K (blue shift on heating in the
          tetragonal phase; the magnitude is an assumption, see the paper).
Output: OUT/temperature.json and OUT/temperature_<ETL>.csv
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
from perouq import calib, jv, params, uq                 # noqa: E402
from perouq.params import DEFAULT_MIN_SPREAD             # noqa: E402
from perouq.study import default_extra, make_cell, TMMCarbonCell   # noqa: E402

TEMPS = [275.0, 300.0, 325.0, 350.0]
DEG_DT = 0.3e-3     # eV/K


class TCell:
    """Wraps a study cell: evaluates it at temperature T with the chosen model."""
    def __init__(self, cell, T, model):
        self.cell, self.T, self.model = cell, T, model
        self.d = cell.d

    def __call__(self, u):
        c, T = self.cell, self.T
        base_theta = c.theta

        def theta(uu):
            vp, ve, vx = base_theta(uu)
            vp, ve = dict(vp), dict(ve)
            if self.model == "full":
                f = (T / 300.0)
                tp, te = params.table("MAPbI3"), params.table(c.etl)
                for v, tab in ((vp, tp), (ve, te)):
                    for k in ("Nc", "Nv"):
                        v[k] = v.get(k, tab[k]["median"]) * f ** 1.5
                for k in ("mu_n", "mu_p"):
                    vp[k] = vp.get(k, tp[k]["median"]) * f ** -1.5
                vp["Eg"] = vp.get("Eg", tp["Eg"]["median"]) + DEG_DT * (T - 300.0)
            return vp, ve, vx
        c.theta = theta
        try:
            if isinstance(c, TMMCarbonCell):
                vp, ve, _ = theta(u)
                from perouq.study import default_stack
                c.stack = default_stack(c.etl, vp["Eg"] if "Eg" in vp else params.table("MAPbI3")["Eg"]["median"],
                                        ve.get("Eg", params.table(c.etl)["Eg"]["median"]), c.scenario, c.nk)
            r = jv.light_jv(c.device(u), c.lam, c.E, V_max=c.V_max, dV=c.dV, T=T, stack=c.stack,
                            gen_scale=c.eta_opt)
            m = jv.metrics(r["V"], r["J"], c.P_in)
            return dict(PCE=100 * m["PCE"], Voc=m["Voc"], Jsc=m["Jsc"] / 10, FF=m["FF"])
        finally:
            c.theta = base_theta


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--W", type=float, nargs=2)
    ap.add_argument("--eta-opt", type=float)
    ap.add_argument("--optics", choices=["bl", "tmm"], default="tmm")
    ap.add_argument("--min-spread", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/temperature")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    extra = default_extra()
    if a.W:
        extra = [p for p in extra if p.name != "W"] + [calib.Param("W", a.W[0], a.W[1])]
    circuit = {} if a.eta_opt is None else {"eta_opt": a.eta_opt}
    ms = DEFAULT_MIN_SPREAD if a.min_spread else None
    out = dict(warning=tag, optics=a.optics, temps=TEMPS, dEg_dT=DEG_DT, n=a.n, etl={})
    for etl in a.etl:
        cell = make_cell(a.optics, etl, lam, E, extra=extra, min_spread=ms, **circuit)
        U = qmc.LatinHypercube(cell.d, seed=0).random(500)[:a.n]
        cols, res = [], {}
        for model in ("kT", "full"):
            for T in TEMPS:
                if model == "full" and T == 300.0:
                    res[(model, T)] = res[("kT", 300.0)]
                    continue
                res[(model, T)] = uq._column(uq.evaluate(TCell(cell, T, model), U, a.jobs), "PCE")
        with open(os.path.join(a.out, f"temperature_{etl}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            keys = list(res)
            w.writerow([f"{m}_{int(T)}" for m, T in keys])
            for i in range(a.n):
                w.writerow([res[k][i] for k in keys])
        e = {}
        for model in ("kT", "full"):
            P = np.array([res[(model, T)] for T in TEMPS])          # (nT, n)
            ok = np.all(np.isfinite(P), axis=0)
            # relative temperature coefficient per sample from a linear fit, %/K of the 300 K value
            slope = np.polyfit(TEMPS, P[:, ok], 1)[0]
            tc = 100 * slope / P[1, ok]
            e[model] = dict(n_ok=int(ok.sum()),
                            median_PCE={str(int(T)): float(np.median(P[j, ok])) for j, T in enumerate(TEMPS)},
                            P15={str(int(T)): float(np.mean(P[j, ok] > 15)) for j, T in enumerate(TEMPS)},
                            tempco_median=float(np.median(tc)),
                            tempco_p5_p95=[float(np.percentile(tc, 5)), float(np.percentile(tc, 95))])
        out["etl"][etl] = e
        print(etl, json.dumps(e), flush=True)
        with open(os.path.join(a.out, "temperature.json"), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
