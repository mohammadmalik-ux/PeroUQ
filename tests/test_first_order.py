"""Given-data first-order index estimator on the Ishigami function (S1 = 0.314, 0.442, 0)."""
import os
import sys

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_first_order import s1_binned   # noqa: E402


def test_ishigami_given_data():
    U = np.vstack([qmc.LatinHypercube(3, seed=s).random(500) for s in range(3)])
    X = -np.pi + 2 * np.pi * U
    Y = np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])
    s1 = [s1_binned(U[:, j], Y) for j in range(3)]
    assert abs(s1[0] - 0.314) < 0.04 and abs(s1[1] - 0.442) < 0.04 and abs(s1[2]) < 0.03
