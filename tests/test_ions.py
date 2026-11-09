"""Mobile ions: the ion number is conserved, the ions screen the absorber field,
and a vanishing ion density gives the ion-free result."""
import os

import numpy as np
import pytest

from perouq import ions, jv, optics
from perouq.study import CarbonCell

SPEC = os.path.join(os.path.dirname(__file__), "..", "data", "ASTMG173.csv")


@pytest.fixture(scope="module")
def setup():
    lam, E = optics.load_astm_g173(SPEC)
    sel = (lam >= 280) & (lam <= 900)
    cell = CarbonCell("TiO2", lam[sel], E[sel], eta_opt=0.95)
    return cell, cell.device(np.full(cell.d, 0.5)), lam[sel], E[sel]


@pytest.mark.skipif(not os.path.exists(SPEC), reason="AM1.5G file not present")
def test_ion_number_conserved(setup):
    cell, dev, lam, E = setup
    s, u = ions._prepare(dev, lam, E, 1e18, 0.95, 300.0)
    u = s.solve(u, 0.6)
    P = s.profile(u[0::3])
    total = np.sum(P[s.mask] * s.dV[s.mask])
    assert abs(total / (s.N0 * s.L) - 1) < 1e-4
    assert P[s.mask].max() > 2 * s.N0           # ions pile up at an interface (screening)


@pytest.mark.skipif(not os.path.exists(SPEC), reason="AM1.5G file not present")
def test_negligible_ions_match_ion_free(setup):
    cell, dev, lam, E = setup
    r0 = jv.light_jv(dev, lam, E, gen_scale=0.95)
    m0 = jv.metrics(r0["V"], r0["J"], np.trapezoid(E, lam))
    r1 = ions.ion_jv(dev, lam, E, 1e10, gen_scale=0.95)
    assert abs(r1["metrics"]["PCE"] / m0["PCE"] - 1) < 0.01
