"""Verification of the Bayesian calibration: recover a synthetic device.

A device with known interface recombination velocity, carbon work function
and series resistance produces 'published' metrics; the calibration must
reproduce them and its accepted set must contain the true parameters.

Run:  python tests/test_calib.py   (about 1 minute)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perouq import Layer, Contact, Interface, Device, optics, params, calib   # noqa: E402

TRUE = dict(S=300.0, W=5.15, Rs=3.0)
PARAMS = [calib.Param("S", 0.01, 1e5, log=True),
          calib.Param("W", 4.95, 5.40),
          calib.Param("Rs", 0.0, 10.0)]


def build(theta):
    pero = params.material("MAPbI3", alpha=optics.alpha_sqrt_model(1.55))
    etl = params.material("TiO2")
    return Device([Layer(etl, 50.0, 50), Layer(pero, 400.0, 120)],
                  left=Contact(None), right=Contact(theta["W"]),
                  interfaces=[Interface(0, theta["S"], theta["S"])])


def synthetic_recovery(n_iter=30, seed=0):
    lam, E = optics.blackbody_test_spectrum()
    sim = calib.device_simulator(build, lam, E)
    truth = sim(TRUE)
    target = {k: truth[k] for k in ("Jsc", "Voc", "FF", "PCE")}
    cal = calib.Calibrator(PARAMS, sim, target, seed=seed)
    res = cal.run(n_iter=n_iter)
    return truth, target, res


def test_synthetic_recovery():
    truth, target, res = synthetic_recovery()
    assert res["best_cost"] < len(target), res["best_cost"]
    for k, t in target.items():
        assert abs(res["best_metrics"][k] / t - 1) < 0.03, (k, res["best_metrics"][k], t)
    for p in PARAMS:
        lo, hi, _ = res["ranges"][p.name]
        u, ulo, uhi = p.to_unit(TRUE[p.name]), p.to_unit(lo), p.to_unit(hi)
        assert ulo - 0.03 <= u <= uhi + 0.03, (p.name, lo, hi, TRUE[p.name])
    assert res["precision"] is None or res["precision"] >= 0.5, res["precision"]


if __name__ == "__main__":
    import time
    t0 = time.time()
    truth, target, res = synthetic_recovery()
    print(f"{res['n_eval']} evaluations in {time.time()-t0:.0f} s, failures={res['failures']}")
    print("target :", {k: round(v, 4) for k, v in target.items()})
    print("best   :", {k: round(res['best_metrics'][k], 4) for k in target}, f"misfit={res['best_cost']:.3f}")
    print("truth  :", TRUE)
    print("best theta:", {k: round(v, 4) for k, v in res["best"].items()})
    print(f"acceptable region: {100*res['region_fraction']:.1f} % of the prior box; "
          f"checked with true simulations: precision {res['precision']}")
    for k, (lo, hi, span) in res["ranges"].items():
        print(f"  {k:3s}: {lo:.4g} .. {hi:.4g}   (covers {100*span:.0f} % of its prior range)")
