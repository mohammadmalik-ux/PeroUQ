"""Verification of the Monte Carlo / Sobol estimators.

Ishigami function (a = 7, b = 0.1) has analytic indices:
S1 = (0.3139, 0.4424, 0), ST = (0.5576, 0.4424, 0.2437).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perouq import uq   # noqa: E402

S1_TRUE = np.array([0.3139, 0.4424, 0.0])
ST_TRUE = np.array([0.5576, 0.4424, 0.2437])


def ishigami(u, a=7.0, b=0.1):
    x = -np.pi + 2 * np.pi * np.asarray(u)
    return {"y": np.sin(x[0]) + a * np.sin(x[1]) ** 2 + b * x[2] ** 4 * np.sin(x[0])}


def test_ishigami_sobol():
    r = uq.sobol(ishigami, 3, 8192, ["y"], seed=1, n_boot=200)["y"]
    assert np.allclose(r["S1"], S1_TRUE, atol=0.02), r["S1"]
    assert np.allclose(r["ST"], ST_TRUE, atol=0.02), r["ST"]
    # the analytic values must lie inside (slightly widened) bootstrap intervals
    assert np.all(r["ST_ci"][0] - 0.01 <= ST_TRUE) and np.all(ST_TRUE <= r["ST_ci"][1] + 0.01)
    return r


def test_failed_runs_are_dropped():
    def f(u):
        if u[0] < 0.05:
            raise RuntimeError("non-converged")
        return ishigami(u)
    r = uq.sobol(f, 3, 4096, ["y"], seed=2, n_boot=50)["y"]
    assert r["n_dropped"] > 0
    assert np.allclose(r["ST"], ST_TRUE, atol=0.05), r["ST"]


def test_exceedance_wilson():
    v = np.array([1.0] * 30 + [0.0] * 70 + [np.nan] * 5)
    e = uq.exceedance(v, 0.5)
    assert e["n"] == 100 and abs(e["p"] - 0.3) < 1e-12
    assert 0.21 < e["low"] < 0.3 < e["high"] < 0.40


if __name__ == "__main__":
    r = test_ishigami_sobol()
    for i in range(3):
        print(f"x{i+1}: S1={r['S1'][i]:.4f} (true {S1_TRUE[i]:.4f})  ST={r['ST'][i]:.4f} (true {ST_TRUE[i]:.4f})"
              f"  ST 95% CI [{r['ST_ci'][0][i]:.3f}, {r['ST_ci'][1][i]:.3f}]")
    test_failed_runs_are_dropped()
    test_exceedance_wilson()
    print("UQ estimators: OK")
