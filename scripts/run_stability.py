"""Statistical stability of the Monte Carlo summaries (P5, P50, P95, P(PCE > 15 %), P(PCE > 20 %)).

    python scripts/run_stability.py --spectrum data/ASTMG173.csv --W 4.95 5.35 \
        --min-spread --optics tmm --main results/uq_tmm --seeds 1 2 --out results/stability_tmm

Three checks for each ETL:
  1. bootstrap 95 % intervals (percentile, 2000 resamples) of every summary for the
     main run (results/uq_tmm, n = 500, seed 0), next to the Wilson intervals;
  2. running estimates for n = 50, 100, ..., 500 (convergence with sample size);
  3. independent replicate runs with other Latin-hypercube seeds (same model and n):
     their spread is the direct measure of run-to-run variability.
Also the bootstrap interval of the TiO2 - CdS difference of each summary.
Output: OUT/stability.json and OUT/mc_<ETL>_seed<k>.csv for the replicates.
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

STATS = ("P5", "P50", "P95", "P15", "P20")


def stats(x):
    x = np.asarray(x, float)
    return dict(P5=float(np.percentile(x, 5)), P50=float(np.median(x)), P95=float(np.percentile(x, 95)),
                P15=float(np.mean(x > 15)), P20=float(np.mean(x > 20)))


def bootstrap(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    B = np.array([[stats(x[rng.integers(0, len(x), len(x))])[k] for k in STATS] for _ in range(n_boot)])
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    return {k: [float(lo[i]), float(hi[i])] for i, k in enumerate(STATS)}


def boot_diff(x, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    D = []
    for _ in range(n_boot):
        a = stats(x[rng.integers(0, len(x), len(x))])
        b = stats(y[rng.integers(0, len(y), len(y))])
        D.append([a[k] - b[k] for k in STATS])
    D = np.array(D)
    lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
    sx, sy = stats(x), stats(y)
    return {k: dict(diff=sx[k] - sy[k], ci=[float(lo[i]), float(hi[i])]) for i, k in enumerate(STATS)}


def read_pce(path):
    with open(path) as f:
        return np.array([float(r["PCE"]) for r in csv.DictReader(f) if r["PCE"]])


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--main", default="results/uq_tmm")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--W", type=float, nargs=2)
    ap.add_argument("--eta-opt", type=float)
    ap.add_argument("--min-spread", action="store_true")
    ap.add_argument("--optics", choices=["bl", "tmm"], default="bl")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/stability_tmm")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    extra = default_extra()
    if a.W:
        extra = [p for p in extra if p.name != "W"] + [calib.Param("W", a.W[0], a.W[1])]
    circuit = {} if a.eta_opt is None else {"eta_opt": a.eta_opt}
    ms = DEFAULT_MIN_SPREAD if a.min_spread else None

    # replicate runs (resumable: an existing CSV is reused)
    for etl in a.etl:
        model = None
        for s in a.seeds:
            path = os.path.join(a.out, f"mc_{etl}_seed{s}.csv")
            if os.path.exists(path):
                continue
            model = model or make_cell(a.optics, etl, lam, E, extra=extra, min_spread=ms, **circuit)
            t0 = time.time()
            U, res = uq.propagate(model, model.d, a.n, seed=s, n_jobs=a.jobs)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["PCE", "Voc", "Jsc", "FF"])
                for r in res:
                    w.writerow([r[k] for k in ("PCE", "Voc", "Jsc", "FF")] if r else [""] * 4)
            print(f"{etl} seed {s}: {round(time.time() - t0)} s", flush=True)

    out = dict(warning=tag, optics=a.optics, n=a.n, seeds=[0] + list(a.seeds), etl={})
    main_pce = {}
    for etl in a.etl:
        x = read_pce(os.path.join(a.main, f"mc_{etl}.csv"))
        main_pce[etl] = x
        reps = [x] + [read_pce(os.path.join(a.out, f"mc_{etl}_seed{s}.csv")) for s in a.seeds]
        rs = [stats(r) for r in reps]
        conv = {str(n): stats(x[:n]) for n in range(50, len(x) + 1, 50)}
        e = dict(n_main=len(x), n_failed=[a.n - len(r) for r in reps], main=stats(x), boot=bootstrap(x),
                 replicates=rs,
                 replicate_range={k: [min(r[k] for r in rs), max(r[k] for r in rs)] for k in STATS},
                 pooled=stats(np.concatenate(reps)), pooled_n=int(sum(len(r) for r in reps)),
                 convergence=conv,
                 change_last_200={k: max(abs(conv[str(n)][k] - conv[str(len(x))][k])
                                         for n in range(300, len(x) + 1, 50)) for k in STATS})
        out["etl"][etl] = e
        print(etl, json.dumps({k: e[k] for k in ("main", "boot", "replicate_range", "change_last_200")}), flush=True)
    if "TiO2" in main_pce and "CdS" in main_pce:
        out["TiO2_minus_CdS"] = boot_diff(main_pce["TiO2"], main_pce["CdS"])
        print("TiO2 - CdS", json.dumps(out["TiO2_minus_CdS"]))
    with open(os.path.join(a.out, "stability.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
