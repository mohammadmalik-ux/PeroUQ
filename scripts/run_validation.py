"""Independent validation: predict a published device that was not used for calibration.

    python scripts/run_validation.py --spectrum data/ASTMG173.csv --calibrated Yang2014-D \
        --test Nandigana2024 --n 200

The posterior of the calibration device (results/posterior/posterior_<cal>.json) is
used, without any refitting, to predict Jsc, Voc and FF of the test device, which has
the same architecture (FTO / c-TiO2 / mp-TiO2 / MAPbI3 / carbon).  Two predictive
distributions:
  posterior          - posterior draws, material parameters at their database medians;
  posterior+material - posterior draws plus the literature uncertainty of the MAPbI3 and
                       TiO2 parameters (independent Latin-hypercube sample).
For each quantity the percentile of the measured value in the predictive distribution
and whether it lies inside the central 95 % interval are reported.
Output: results/validation/validation_<test>.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum              # noqa: E402
from perouq import calib, optics, params                         # noqa: E402
from perouq.device import Layer, Contact, Interface, Device      # noqa: E402
from perouq.params import DEFAULT_MIN_SPREAD                     # noqa: E402
from run_calibration import DEVICES, make_params                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--calibrated", default="Yang2014-D")
    ap.add_argument("--test", default="Nandigana2024")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/validation")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    post = json.load(open(f"results/posterior/posterior_{a.calibrated}.json"))
    prm = make_params(DEVICES[a.calibrated])
    draws = np.array(post["draws_unit"])
    rng = np.random.default_rng(0)
    pick = draws[rng.choice(len(draws), a.n, replace=len(draws) < a.n)]
    test = DEVICES[a.test]
    cfg_cal = DEVICES[a.calibrated]
    rp = {p: r for p, r in params.ranges("MAPbI3", min_spread=DEFAULT_MIN_SPREAD).items() if r[0] != r[1] and p != "Nt"}
    re = {p: r for p, r in params.ranges(test["etl"], min_spread=DEFAULT_MIN_SPREAD).items() if r[0] != r[1]}
    p_names, e_names = sorted(rp), sorted(re)
    Umat = qmc.LatinHypercube(len(p_names) + len(e_names), seed=5).random(a.n)

    def simulate(th, um=None):
        vp, ve = {"Nt": th["Nt"]}, {}
        if um is not None:
            v1, _ = params.sample_values("MAPbI3", um[:len(p_names)], min_spread=DEFAULT_MIN_SPREAD)
            v2, _ = params.sample_values(test["etl"], um[len(p_names):], min_spread=DEFAULT_MIN_SPREAD)
            vp.update({k: v for k, v in v1.items() if k != "Nt"}); ve.update(v2)
        Eg = vp.get("Eg", params.table("MAPbI3")["Eg"]["median"])
        pero = params.material("MAPbI3", vp, alpha=optics.alpha_sqrt_model(Eg))
        etl = params.material(test["etl"], ve)
        t = th.get("pero_nm", test["pero_nm"] or cfg_cal["pero_nm"])
        dev = Device([Layer(etl, test["etl_nm"], 50), Layer(pero, t, 120)],
                     left=Contact(None), right=Contact(th["W"]), interfaces=[Interface(0, th["S"], th["S"])])
        sim = calib.device_simulator(lambda _: dev, lam, E, P_in=1000.0)
        return sim(th)

    out = dict(calibrated=a.calibrated, test=a.test, measured=test["target"], warning=tag, n=a.n, variants={})
    for label in ("posterior", "posterior+material"):
        M = []
        for i, u in enumerate(pick):
            th = {p.name: p.from_unit(x) for p, x in zip(prm, u)}
            try:
                m = simulate(th, Umat[i] if label != "posterior" else None)
                M.append([m["Jsc"], m["Voc"], m["FF"]])
            except Exception:
                pass
        M = np.array(M)
        res = dict(n_ok=len(M))
        for j, k in enumerate(("Jsc", "Voc", "FF")):
            x, obs = M[:, j], test["target"][k]
            lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
            res[k] = dict(median=float(med), ci95=[float(lo), float(hi)], measured=obs,
                          percentile_of_measured=float(100 * np.mean(x < obs)), inside_95=bool(lo <= obs <= hi))
        out["variants"][label] = res
        print(label, json.dumps(res), flush=True)
    with open(os.path.join(a.out, f"validation_{a.test}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
