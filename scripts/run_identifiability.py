"""Which measurements would pin down the unknown parameters?  A synthetic test.

    python scripts/run_identifiability.py --spectrum data/ASTMG173.csv --truth iqbal --targets A B C D

A 'true' device is simulated (planar ITO/SnO2/MAPbI3/carbon, the Iqbal2023 geometry).
Its synthetic measurements are then used as calibration targets, and the calibration
of run_calibration.py (same six unknowns: S, W, Nt, Rs, Rsh, eta_opt) is repeated with
growing sets of targets:

  A  Jsc, Voc, FF at 1 sun (what papers report; the set used in the study)
  B  A + ideality factor from Voc at 1 and 0.1 sun          (light-intensity J-V)
  C  B + dVoc/dT between 280 and 320 K                      (temperature-dependent J-V)
  D  C + EQE at 400, 550, 700 and 780 nm                     (EQE spectrum)

The width of the acceptable range of each unknown shows what each data set adds.
Truths: 'iqbal' = the best fit to the published device (interface-limited, S ~ 1e4 cm/s);
'bulk' = the same device with S = 10 cm/s and Nt = 8e14 cm^-3 (bulk-limited, inside the prior).
The synthetic data carry no noise; the tolerances stand for measurement uncertainty.
Temperature enters only through kT: band gaps, mobilities and capture cross-sections
are kept at their 300 K values.
Output: results/identifiability/ident_<truth>_<set>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum      # noqa: E402
from perouq import calib, jv, posterior                  # noqa: E402
from perouq.constants import thermal_voltage             # noqa: E402
from run_calibration import DEVICES, make_builder, make_params   # noqa: E402

EQE_WL = [400.0, 550.0, 700.0, 780.0]
SETS = {
    "A": ["Jsc", "Voc", "FF"],
    "B": ["Jsc", "Voc", "FF", "n_id"],
    "C": ["Jsc", "Voc", "FF", "n_id", "dVdT"],
    "D": ["Jsc", "Voc", "FF", "n_id", "dVdT"] + [f"EQE_{int(w)}" for w in EQE_WL],
}
# absolute tolerances for the new quantities (one 'tolerance' = expected measurement uncertainty)
ABS_TOL = {"n_id": 0.1, "dVdT": 0.1, **{f"EQE_{int(w)}": 0.02 for w in EQE_WL}}


def truths():
    best = json.load(open("results/calibration_Iqbal2023.json"))["best"]
    bulk = dict(best, S=10.0, Nt=8e14)
    return {"iqbal": best, "bulk": bulk}


def make_simulator(cfg, lam, E, keys):
    build = make_builder(cfg)
    base = calib.device_simulator(build, lam, E, P_in=1000.0)

    def simulate(th):
        m = base(th)
        eta, Rs, Rsh = th.get("eta_opt", 1.0), th.get("Rs", 0.0) * 1e-4, th.get("Rsh", np.inf) * 1e-4
        dev = None
        if "n_id" in keys:
            dev = dev or build(th)
            v01 = jv.voc_at(dev, lam, E, suns=0.1, Rs=Rs, Rsh=Rsh, gen_scale=eta)
            v1 = jv.voc_at(dev, lam, E, suns=1.0, Rs=Rs, Rsh=Rsh, gen_scale=eta)
            m["n_id"] = (v1 - v01) / (thermal_voltage(300.0) * np.log(10.0))
        if "dVdT" in keys:
            dev = dev or build(th)
            vlo = jv.voc_at(dev, lam, E, T=280.0, Rs=Rs, Rsh=Rsh, gen_scale=eta)
            vhi = jv.voc_at(dev, lam, E, T=320.0, Rs=Rs, Rsh=Rsh, gen_scale=eta)
            m["dVdT"] = 1e3 * (vhi - vlo) / 40.0                      # mV/K
        if any(k.startswith("EQE_") for k in keys):
            dev = dev or build(th)
            Q = jv.eqe(dev, EQE_WL, gen_scale=eta)
            for w, qv in zip(EQE_WL, Q):
                m[f"EQE_{int(w)}"] = float(qv)
        return m
    return simulate


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--truth", choices=["iqbal", "bulk"], default="iqbal")
    ap.add_argument("--targets", nargs="+", default=list(SETS))
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--n-map", type=int, default=40)
    ap.add_argument("--n-check", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/identifiability")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    cfg = DEVICES["Iqbal2023"]
    truth = truths()[a.truth]
    full = make_simulator(cfg, lam, E, SETS["D"])(truth)
    data = {k: float(full[k]) for k in SETS["D"]}
    print("synthetic data:", json.dumps(data), flush=True)
    for s in a.targets:
        path = os.path.join(a.out, f"ident_{a.truth}_{s}.json")
        if os.path.exists(path):
            print(s, "done, skipped"); continue
        keys = SETS[s]
        rel = dict(calib.DEFAULT_REL)
        rel.update({k: ABS_TOL[k] / abs(data[k]) for k in keys if k in ABS_TOL})
        cal = calib.Calibrator(make_params(cfg), make_simulator(cfg, lam, E, keys),
                               {k: data[k] for k in keys}, rel=rel, seed=a.seed)
        t0 = time.time()
        res = cal.run(n_iter=a.n_iter, n_map=a.n_map, n_check=a.n_check)
        gp = cal._gp()
        acc_pts = [x for x, c in zip(cal.X, cal.C) if c <= cal.n_targets()]
        lp = posterior.log_post_factory(gp, len(cal.params))
        S_post, acc = posterior.ensemble_sampler(lp, len(cal.params), n_walkers=64, n_steps=2500, burn=800,
                                                 thin=10, x0=acc_pts or None, seed=1)
        post, corr = posterior.summarise(S_post, cal.params)
        out = dict(truth_name=a.truth, truth=truth, set=s, targets={k: data[k] for k in keys},
                   tolerances={k: rel.get(k, 0.02) * abs(data[k]) for k in keys}, warning=tag,
                   best=res["best"], best_cost=res["best_cost"],
                   ranges={k: [v[0], v[1]] for k, v in res["ranges"].items()},
                   region_fraction=res["region_fraction"], precision=res["precision"],
                   posterior=post, posterior_acceptance=float(acc),
                   history=dict(X=np.array(cal.X).round(6).tolist(), C=[float(c) for c in cal.C]),
                   n_eval=res["n_eval"], failures=res["failures"], seconds=round(time.time() - t0))
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(s, json.dumps({k: out[k] for k in ("ranges", "region_fraction", "precision", "seconds")}),
              flush=True)
        for p, v in post.items():
            print(f"   {p:8s} 95% [{v['ci95'][0]:.4g}, {v['ci95'][1]:.4g}]  width/prior {v['width_fraction_of_prior']:.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
