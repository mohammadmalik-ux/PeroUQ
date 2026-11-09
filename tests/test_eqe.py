"""EQE: bounded by 1, and its integral over the AM1.5G photon flux equals Jsc (linear response)."""
import os

import numpy as np
import pytest

from perouq import optics, jv
from perouq.constants import q
from perouq.study import CarbonCell

SPEC = os.path.join(os.path.dirname(__file__), "..", "data", "ASTMG173.csv")


@pytest.mark.skipif(not os.path.exists(SPEC), reason="AM1.5G file not present")
def test_eqe_integral_matches_jsc():
    lam, E = optics.load_astm_g173(SPEC)
    sel = (lam >= 280) & (lam <= 900)
    lam, E = lam[sel], E[sel]
    cell = CarbonCell("TiO2", lam, E, eta_opt=1.0)
    u = np.full(cell.d, 0.5)
    wl = np.arange(300, 841, 20.0)
    Q = jv.eqe(cell.device(u), wl)
    assert np.all(Q <= 1.0 + 1e-6) and np.all(Q >= -1e-6)
    J_eqe = q * np.trapezoid(np.interp(lam, wl, Q, left=0, right=0) * optics.photon_flux(lam, E), lam) / 10
    J_sc = cell(u)["Jsc"]
    assert abs(J_eqe - J_sc) / J_sc < 0.01
