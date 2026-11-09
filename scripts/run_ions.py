"""Mobile-ion scenarios: how much do ions change the efficiency and the ETL comparison?

    python scripts/run_ions.py --spectrum data/ASTMG173.csv --W 4.95 5.35 --eta-opt 0.95 \
        --min-spread --n 40 --out results/ions

For the first n Latin-hypercube samples of the main study (seed 0) and each ETL,
the J-V curve is computed without ions and with a mobile vacancy density
N0 = 1e17, 1e18 and 1e19 cm^-3 in three limits (perouq.ions):
  steady   - ions in equilibrium at every voltage (stabilised efficiency),
  fwd      - ions frozen at 0 V, fast scan (forward scan from short circuit),
  rev      - ions frozen at 1.0 V under light, fast scan (reverse scan after light soaking).
Hysteresis index HI = (PCE_rev - PCE_fwd) / PCE_rev.
Beer-Lambert generation x eta_opt is used here; the quantities reported are changes
relative to the ion-free device with the same optics.
Output: OUT/ions_<ETL>.csv and OUT/ions_summary.json; --mesh-check adds a refinement test.
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum      # noqa: E402
from perouq import calib, ions                           # noqa: E402
from perouq.params import DEFAULT_MIN_SPREAD             # noqa: E402
from perouq.study import CarbonCell, default_extra       # noqa: E402

N0S = [1e17, 1e18, 1e19]
MODES = [("steady", None), ("fwd", 0.0), ("rev", 1.0)]


def one(args):
    cell, u = args
    out = {}
    try:
        out["none"] = cell(u)["PCE"]
    except Exception:
        return None
    dev = cell.device(u)
    for N0 in N0S:
        for name, Vp in MODES:
            key = f"{name}_{N0:.0e}"
            try:
                r = ions.ion_jv(dev, cell.lam, cell.E, N0, mode="frozen" if Vp is not None else "steady",
                                V_pre=Vp, gen_scale=cell.eta_opt)
                out[key] = 100 * r["metrics"]["PCE"]
            except Exception:
                out[key] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--W", type=float, nargs=2)
    ap.add_argument("--eta-opt", type=float, default=0.95)
    ap.add_argument("--min-spread", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--mesh-check", action="store_true")
    ap.add_argument("--out", default="results/ions")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    extra = default_extra()
    if a.W:
        extra = [p for p in extra if p.name != "W"] + [calib.Param("W", a.W[0], a.W[1])]
    ms = DEFAULT_MIN_SPREAD if a.min_spread else None
    summary = dict(warning=tag, N0_cm3=N0S, n=a.n, etl={})
    for etl in a.etl:
        cell = CarbonCell(etl, lam, E, extra=extra, min_spread=ms, eta_opt=a.eta_opt)
        U = qmc.LatinHypercube(cell.d, seed=0).random(500)[:a.n]
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(one, [(cell, u) for u in U]))
        keys = ["none"] + [f"{m}_{N0:.0e}" for N0 in N0S for m, _ in MODES]
        with open(os.path.join(a.out, f"ions_{etl}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cell.names + keys)
            for u, r in zip(U, res):
                w.writerow(list(u) + ([r[k] for k in keys] if r else [""] * len(keys)))
        R = np.array([[r[k] for k in keys] for r in res if r], float)
        Uok = np.array([u for u, r in zip(U, res) if r])
        e = dict(n_ok=len(R), median_none=float(np.nanmedian(R[:, 0])), P15_none=float(np.mean(R[:, 0] > 15)))
        for N0 in N0S:
            tagN = f"{N0:.0e}"
            st = R[:, keys.index(f"steady_{tagN}")]
            fw = R[:, keys.index(f"fwd_{tagN}")]
            rv = R[:, keys.index(f"rev_{tagN}")]
            ok = np.isfinite(st) & np.isfinite(fw) & np.isfinite(rv)
            d = st - R[:, 0]
            HI = (rv - fw) / rv
            rho = {}
            for j, nm in enumerate(cell.names):
                if np.sum(np.isfinite(d)) > 5:
                    rho[nm] = float(spearmanr(Uok[np.isfinite(d), j], d[np.isfinite(d)]).correlation)
            top = sorted(rho.items(), key=lambda t: -abs(t[1]))[:4]
            e[tagN] = dict(n_converged=int(np.sum(ok)),
                           dPCE_steady_median=float(np.nanmedian(d)),
                           dPCE_steady_p5_p95=[float(np.nanpercentile(d, 5)), float(np.nanpercentile(d, 95))],
                           median_steady=float(np.nanmedian(st)), P15_steady=float(np.nanmean(st > 15)),
                           median_rev=float(np.nanmedian(rv)), median_fwd=float(np.nanmedian(fw)),
                           HI_median=float(np.nanmedian(HI[ok])) if ok.any() else None,
                           drivers_of_dPCE=top)
        summary["etl"][etl] = e
        print(etl, json.dumps(e, default=float), flush=True)
        with open(os.path.join(a.out, "ions_summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
    if a.mesh_check:
        cell = CarbonCell("TiO2", lam, E, extra=extra, min_spread=ms, eta_opt=a.eta_opt)
        U = qmc.LatinHypercube(cell.d, seed=0).random(500)[:3]
        rows = []
        for u in U:
            row = []
            for npero in (120, 300):
                cell.n_pero = npero
                r = ions.ion_jv(cell.device(u), lam, E, 1e19, mode="steady", gen_scale=a.eta_opt)
                row.append(100 * r["metrics"]["PCE"])
            rows.append(row)
        summary["mesh_check_1e19_steady_PCE_120_vs_300_nodes"] = rows
        print("mesh check", rows)
        with open(os.path.join(a.out, "ions_summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
