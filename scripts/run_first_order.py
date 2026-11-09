"""First-order Sobol indices from Monte Carlo data (given-data estimator) and interaction share.

    python scripts/run_first_order.py --main results/uq_tmm --replicates results/stability_tmm \
        --sobol results/sobol512_tmm --out results/first_order

The Saltelli/Jansen design gives precise total indices but wide intervals for the
first-order indices.  Here S1 is estimated directly from the Monte Carlo samples
(main run + replicate runs, 1500 per ETL) as the correlation ratio
    S1_i = Var(E[Y | X_i]) / Var(Y),
with E[Y | X_i] from M equal-count bins of X_i and the usual bias correction
(Plischke et al. 2013).  95 % intervals by half-sampling without replacement: an ordinary
bootstrap duplicates points, which inflates a binned estimator, whereas the deviation of
a half-sample estimate from the full estimate has about the variance of the full estimate.  The inputs are regenerated from the
Latin-hypercube seeds used by the runs (seed 0 main, seeds 1, 2 replicates).
Interaction share = 1 - sum S1;  for each parameter ST - S1 (ST from the Sobol run, if given).
Output: OUT/first_order.json
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import qmc


def s1_binned(x, y, M=25):
    order = np.argsort(x)
    ys = y[order]
    bins = np.array_split(ys, M)
    n = len(y)
    vy = np.var(y, ddof=1)
    between = sum(len(b) * (b.mean() - y.mean()) ** 2 for b in bins) / (n - 1)
    within = sum(((b - b.mean()) ** 2).sum() for b in bins) / (n - M)
    # bias correction: E[between] = Var(E[Y|X]) + (M - 1) within / (n - 1)
    return float((between - (M - 1) * within / (n - 1)) / vy)


def read(path):
    with open(path) as f:
        return np.array([float(r["PCE"]) if r["PCE"] else np.nan for r in csv.DictReader(f)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--main", default="results/uq_tmm")
    ap.add_argument("--replicates", default="results/stability_tmm")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--sobol", default="", help="directory with sobol_<ETL>.csv (optional)")
    ap.add_argument("--sobol-suffix", default="", help="e.g. _N256")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="results/first_order")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    out = dict(method="given-data correlation ratio, 25 bins, bias-corrected", etl={})
    for etl in a.etl:
        with open(os.path.join(a.main, f"mc_{etl}.csv")) as f:
            header = next(csv.reader(f))
        names = header[:header.index("PCE")]
        ST, S1_saltelli = {}, {}
        spath = os.path.join(a.sobol, f"sobol_{etl}{a.sobol_suffix}.csv") if a.sobol else ""
        if spath and os.path.exists(spath):
            with open(spath) as f:
                rows = list(csv.DictReader(f))
            assert [r["parameter"] for r in rows] == names, "parameter order differs from the Monte Carlo run"
            ST = {r["parameter"]: float(r["ST"]) for r in rows}
            S1_saltelli = {r["parameter"]: (float(r["S1"]), float(r["S1_low"]), float(r["S1_high"])) for r in rows}
        d = len(names)
        Us, Ys = [qmc.LatinHypercube(d, seed=0).random(a.n)], [read(os.path.join(a.main, f"mc_{etl}.csv"))]
        for s in a.seeds:
            Us.append(qmc.LatinHypercube(d, seed=s).random(a.n))
            Ys.append(read(os.path.join(a.replicates, f"mc_{etl}_seed{s}.csv")))
        # the regenerated inputs must be the ones the main run used (rank check against its CSV)
        with open(os.path.join(a.main, f"mc_{etl}.csv")) as f:
            X0 = np.array([[float(r[n]) for n in names] for r in csv.DictReader(f)])
        from scipy.stats import spearmanr
        assert min(spearmanr(Us[0][:, j], X0[:, j])[0] for j in range(len(names))) > 0.999
        U, Y = np.vstack(Us), np.concatenate(Ys)
        ok = np.isfinite(Y)
        U, Y = U[ok], Y[ok]
        s1 = np.array([s1_binned(U[:, j], Y) for j in range(d)])
        D = []
        m = len(Y) // 2
        for _ in range(a.n_boot):
            i = rng.choice(len(Y), m, replace=False)
            D.append([s1_binned(U[i, j], Y[i]) for j in range(d)])
        D = np.array(D) - s1                                  # half-sample deviations
        lo, hi = s1 + np.percentile(D, 2.5, axis=0), s1 + np.percentile(D, 97.5, axis=0)
        Ds = D.sum(axis=1)
        order = np.argsort([-ST.get(n, s1[k]) for k, n in enumerate(names)])
        e = dict(n=int(len(Y)), sum_S1=float(s1.sum()), interaction_share=float(1 - s1.sum()),
                 sum_S1_ci=[float(s1.sum() + np.percentile(Ds, 2.5)), float(s1.sum() + np.percentile(Ds, 97.5))],
                 params={})
        for j in order[:10]:
            n = names[j]
            e["params"][n] = dict(S1=float(s1[j]), S1_ci=[float(lo[j]), float(hi[j])])
            if n in ST:
                e["params"][n].update(ST=ST[n], ST_minus_S1=float(ST[n] - s1[j]), S1_saltelli=S1_saltelli[n])
        out["etl"][etl] = e
        print(etl, json.dumps(e, indent=1), flush=True)
    with open(os.path.join(a.out, "first_order.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
