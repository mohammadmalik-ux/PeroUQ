"""Robust design optimiser on an analytic problem with a known answer.

Toy model: PCE = 20 - 30 (x1 - 0.3)^2 - (5 + 40 (x2 - 0.7)^2) * |z|, z ~ U(-1, 1).
Mean is maximal at x1 = 0.3 for any x2; spread is minimal at x2 = 0.7.
With lambda = 1 the robust optimum is (0.3, 0.7).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perouq.design import RobustDesigner   # noqa: E402
from perouq.calib import Param             # noqa: E402


class Toy:
    def __init__(self, x1, x2):
        self.x1, self.x2 = x1, x2

    def __call__(self, u):
        z = 2 * u[0] - 1
        return {"PCE": 20 - 30 * (self.x1 - 0.3) ** 2 - (5 + 40 * (self.x2 - 0.7) ** 2) * abs(z)}


def test_robust_optimum():
    rd = RobustDesigner([Param("x1", 0, 1), Param("x2", 0, 1)],
                        lambda d: Toy(d["x1"], d["x2"]), d_uncertain=1, n_inner=64, lam=1.0, seed=3)
    r = rd.run(n_init=8, n_iter=20)
    assert abs(r["best"]["x1"] - 0.3) < 0.08, r["best"]
    assert abs(r["best"]["x2"] - 0.7) < 0.08, r["best"]
    return r


if __name__ == "__main__":
    r = test_robust_optimum()
    print("best design:", {k: round(v, 3) for k, v in r["best"].items()}, "(true 0.3, 0.7)",
          "F =", round(r["best_F"], 3))
