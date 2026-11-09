"""Bayesian posterior for a calibration device (emulator-based).

    python scripts/run_posterior.py --spectrum data/ASTMG173.csv --device Iqbal2023 --seeds 0
    python scripts/run_posterior.py --spectrum data/ASTMG173.csv --device Yang2014-D --seeds 0 1 2

Runs the Bayesian-optimisation calibration of run_calibration.py for each seed,
pools all simulations, fits the Gaussian-process emulator of the misfit and samples
the posterior with an ensemble MCMC (perouq.posterior).  Then simulates n_check
random posterior draws (posterior-predictive check).
Output: results/posterior/posterior_<device>.json (summary, correlations, 2000 draws).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum              # noqa: E402
from perouq import calib, posterior                             # noqa: E402
from run_calibration import DEVICES, make_builder, make_params   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--device", choices=sorted(DEVICES), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--n-map", type=int, default=60)
    ap.add_argument("--n-check", type=int, default=40)
    ap.add_argument("--out", default="results/posterior")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    cfg = DEVICES[a.device]
    prm = make_params(cfg)
    sim = calib.device_simulator(make_builder(cfg), lam, E, P_in=1000.0)
    t0 = time.time()
    pooled = calib.Calibrator(prm, sim, cfg["target"], seed=99)
    for s in a.seeds:
        cal = calib.Calibrator(prm, sim, cfg["target"], seed=s)
        cal.run(n_iter=a.n_iter, n_map=a.n_map, n_check=0)
        pooled.X += cal.X; pooled.C += cal.C; pooled.M += cal.M; pooled.thetas += cal.thetas
        print(f"seed {s}: {len(cal.C)} runs, best C = {min(cal.C):.3f}", flush=True)
    gp = pooled._gp()
    d = len(prm)
    acc_pts = [x for x, c in zip(pooled.X, pooled.C) if c <= pooled.n_targets()]
    res = {}
    for label, noisy in (("emulator_mean", False),):
        lp = posterior.log_post_factory(gp, d, noisy=noisy, rng=np.random.default_rng(1))
        S, acc = posterior.ensemble_sampler(lp, d, n_walkers=64, n_steps=3000, burn=1000, thin=10,
                                            x0=acc_pts or None, seed=1)
        summ, corr = posterior.summarise(S, prm)
        res[label] = dict(acceptance=float(acc), n_samples=len(S), params=summ,
                          corr=dict(names=[p.name for p in prm], matrix=corr.round(3).tolist()))
        if label == "emulator_mean":
            draws = S[np.random.default_rng(2).choice(len(S), min(2000, len(S)), replace=False)]
    # posterior-predictive check with true simulations
    rng = np.random.default_rng(3)
    pick = draws[rng.choice(len(draws), a.n_check, replace=False)]
    pred, inside = [], []
    for u in pick:
        try:
            m = sim(pooled.theta_of(u))
            pred.append([m[k] for k in cfg["target"]])
            inside.append(pooled.misfit(m) <= pooled.n_targets())
        except Exception:
            pass
    pred = np.array(pred)
    out = dict(device=a.device, seeds=a.seeds, warning=tag, target=cfg["target"],
               tolerances={k: calib.DEFAULT_REL[k] * abs(v) for k, v in cfg["target"].items()},
               n_simulations=len(pooled.C), posterior=res,
               predictive=dict(n=len(pred), fraction_within_tolerance=float(np.mean(inside)),
                               mean={k: float(pred[:, i].mean()) for i, k in enumerate(cfg["target"])},
                               sd={k: float(pred[:, i].std(ddof=1)) for i, k in enumerate(cfg["target"])}),
               draws_unit=draws.round(5).tolist(), param_names=[p.name for p in prm],
               history=dict(X=np.array(pooled.X).round(6).tolist(), C=[float(c) for c in pooled.C]),
               seconds=round(time.time() - t0))
    with open(os.path.join(a.out, f"posterior_{a.device}.json"), "w") as f:
        json.dump(out, f)
    print(json.dumps({k: out[k] for k in ("n_simulations", "predictive", "seconds")}, indent=1))
    for k, v in res.items():
        print(k, "acc", round(v["acceptance"], 3))
        for p, s in v["params"].items():
            print(f"  {p:8s} median {s['median']:.4g}  95% [{s['ci95'][0]:.4g}, {s['ci95'][1]:.4g}]  "
                  f"width/prior {s['width_fraction_of_prior']:.2f}")
    print("corr", res["emulator_mean"]["corr"])


if __name__ == "__main__":
    main()
