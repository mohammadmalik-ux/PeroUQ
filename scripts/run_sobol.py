"""Sobol indices of PCE only (no Monte Carlo), for a larger base sample N.

    python scripts/run_sobol.py --spectrum data/ASTMG173.csv --W 4.95 5.35 \
        --min-spread --optics tmm --N 512 --etl TiO2 --out results/sobol512_tmm

Uses the same model settings as run_uq_study.py. Writes sobol_<ETL>.csv and
sobol_<ETL>.json per ETL, so each ETL can be run (and resumed) separately.
Cost: N (d + 2) runs per ETL; d = 22 gives 12 288 runs for N = 512.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum      # noqa: E402
from perouq import uq, calib                             # noqa: E402
from perouq.params import DEFAULT_MIN_SPREAD             # noqa: E402
from perouq.study import default_extra, make_cell        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--W", type=float, nargs=2)
    ap.add_argument("--eta-opt", type=float)
    ap.add_argument("--min-spread", action="store_true")
    ap.add_argument("--optics", choices=["bl", "tmm"], default="bl")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default="results/sobol512_tmm")
    ap.add_argument("--chunk", type=int, default=0,
                    help="evaluate at most this many design rows per call (resumable; 0 = all)")
    ap.add_argument("--upto", type=int, default=0,
                    help="evaluate only base samples j < UPTO (rows are run sample by sample, so any prefix "
                         "of the base sample is a complete Saltelli design) and report the indices for N = UPTO")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    extra = default_extra()
    if a.W:
        extra = [p for p in extra if p.name != "W"] + [calib.Param("W", a.W[0], a.W[1])]
    circuit = {} if a.eta_opt is None else {"eta_opt": a.eta_opt}
    ms = DEFAULT_MIN_SPREAD if a.min_spread else None
    for etl in a.etl:
        Nuse = a.upto if 0 < a.upto < a.N else a.N
        sfx = "" if Nuse == a.N else f"_N{Nuse}"
        jpath = os.path.join(a.out, f"sobol_{etl}{sfx}.json")
        if os.path.exists(jpath):
            print(f"{etl}: done, skipped")
            continue
        t0 = time.time()
        model = make_cell(a.optics, etl, lam, E, extra=extra, min_spread=ms, **circuit)
        print(f"== {etl}: d = {model.d}, {a.N * (model.d + 2)} runs", flush=True)
        # resumable evaluation of the Saltelli design in chunks
        A, B, AB = uq.saltelli_design(model.d, a.N, a.seed)
        U = np.vstack([A, B] + AB)
        ppath = os.path.join(a.out, f"sobol_{etl}_partial.npy")
        y = np.load(ppath) if os.path.exists(ppath) else np.full(len(U), np.nan)
        done = np.load(ppath + ".done.npy") if os.path.exists(ppath + ".done.npy") else np.zeros(len(U), bool)
        j = np.arange(len(U)) % a.N                      # base-sample index of every row
        todo = np.where(~done & (j < Nuse))[0]
        todo = todo[np.argsort(j[todo], kind="stable")]  # sample by sample: prefixes are nested designs
        if a.chunk:
            todo = todo[:a.chunk]
        for k in range(0, len(todo), 256):          # save after every batch (survives interruptions)
            idx = todo[k:k + 256]
            res = uq.evaluate(model, U[idx], a.jobs)
            y[idx] = uq._column(res, "PCE")
            done[idx] = True
            np.save(ppath, y)
            np.save(ppath + ".done.npy", done)
        if not done[j < Nuse].all():
            print(f"{etl}: {int(done[j < Nuse].sum())} of {int((j < Nuse).sum())} rows (N = {Nuse}) done; "
                  f"{int(done.sum())} of {len(U)} in total; run again to continue", flush=True)
            continue
        N = a.N
        fA, fB, fAB = y[:N][:Nuse], y[N:2 * N][:Nuse], y[2 * N:].reshape(model.d, N)[:, :Nuse]
        sob = uq.sobol_from_outputs(fA, fB, fAB, n_boot=a.n_boot, seed=a.seed)
        sob["raw"] = dict(fA=fA, fB=fB, fAB=fAB)
        with open(os.path.join(a.out, f"sobol_{etl}{sfx}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["parameter", "S1", "S1_low", "S1_high", "ST", "ST_low", "ST_high"])
            for i, n in enumerate(model.names):
                w.writerow([n, sob["S1"][i], *sob["S1_ci"][:, i], sob["ST"][i], *sob["ST_ci"][:, i]])
        raw = sob["raw"]
        np.savez_compressed(os.path.join(a.out, f"sobol_{etl}{sfx}_raw.npz"), fA=raw["fA"], fB=raw["fB"], fAB=raw["fAB"])
        # convergence: indices from the first N' rows of the (balanced) Sobol design
        conv = {}
        for Np in sorted({64, 128, 256, 512, Nuse}):
            if Np > Nuse:
                continue
            s2 = uq.sobol_from_outputs(raw["fA"][:Np], raw["fB"][:Np], raw["fAB"][:, :Np], n_boot=200, seed=a.seed)
            conv[str(Np)] = {model.names[i]: [float(s2["ST"][i]), float(s2["S1"][i])] for i in range(model.d)}
        order = np.argsort(sob["ST"])[::-1]
        out = dict(etl=etl, optics=a.optics, N=Nuse, d=model.d, warning=tag, seconds=round(time.time() - t0),
                   top=[(model.names[i], float(sob["ST"][i]), *map(float, sob["ST_ci"][:, i]),
                         float(sob["S1"][i]), *map(float, sob["S1_ci"][:, i])) for i in order[:8]],
                   sum_S1=float(np.sum(sob["S1"])), n_dropped=int(sob.get("n_dropped", 0)),
                   convergence=conv)
        with open(jpath, "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
