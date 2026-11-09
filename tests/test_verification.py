"""Verification of the drift-diffusion solver against analytic results.

Run:  python -m pytest -q tests      (or)      python tests/test_verification.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perouq import Material, Layer, Contact, Device, DDSolver          # noqa: E402
from perouq.constants import q, eps0, CM3, CM2VS                      # noqa: E402
from perouq import optics                                             # noqa: E402

# Silicon-like parameters (textbook values, 300 K)
SI = dict(Eg=1.12, chi=4.05, eps_r=11.7, Nc=2.8e19, Nv=1.04e19, mu_n=1400.0, mu_p=450.0,
          tau_n=1.0, tau_p=1.0)          # tau = 1 s: no bulk recombination (short-base diode)
DOP = 1e17       # cm^-3
T_UM = 3.0       # each side [um]


def pn_device(n_points=150):
    n_side = Material("Si-n", ND=DOP, **SI)
    p_side = Material("Si-p", NA=DOP, **SI)
    big = 1e9   # cm/s: ideal ohmic contacts
    return Device([Layer(n_side, T_UM * 1e3, n_points), Layer(p_side, T_UM * 1e3, n_points)],
                  left=Contact(None, big, big), right=Contact(None, big, big))


def shockley_short_base(V, s):
    """J0 (exp(V/Vt) - 1) for a short-base diode with quasi-neutral widths
    that shrink with the depletion region.  Returns the dark current magnitude."""
    Vt = s.Vt
    ni = s.ni[0]
    Na = Nd = DOP * CM3
    eps = SI["eps_r"] * eps0
    Vbi = Vt * np.log(Na * Nd / ni**2)
    W = np.sqrt(2 * eps * (Vbi - V) / q * (1 / Na + 1 / Nd))
    Wn = T_UM * 1e-6 - W / 2
    Wp = T_UM * 1e-6 - W / 2
    Dn = SI["mu_n"] * CM2VS * Vt
    Dp = SI["mu_p"] * CM2VS * Vt
    J0 = q * ni**2 * (Dp / (Nd * Wn) + Dn / (Na * Wp))
    return J0 * np.expm1(V / Vt)


def test_equilibrium_zero_current():
    s = DDSolver(pn_device())
    u = s.equilibrium()
    J, _ = s.terminal_current(u)
    assert abs(J) < 1e-9, J


def test_shockley_diode():
    s = DDSolver(pn_device())
    u = s.equilibrium()
    rows = []
    for V in np.arange(0.02, 0.62, 0.02):
        u, _ = s.newton(u, V)
        if min(abs(V - 0.3), abs(V - 0.4), abs(V - 0.5)) < 1e-9:
            J, spread = s.terminal_current(u, absolute=True)
            ref = shockley_short_base(V, s)
            rows.append((V, -J, ref, -J / ref, spread))
    for V, Jsim, Jref, ratio, spread in rows:
        assert abs(ratio - 1) < 0.01, (V, ratio)
        assert spread < 1e-3, (V, spread)        # A/m^2  (= 1e-7 A/cm^2)
    return rows


def test_mesh_convergence():
    J = []
    for n in (100, 200, 400):
        s = DDSolver(pn_device(n))
        u = s.equilibrium()
        for V in np.arange(0.05, 0.45, 0.05):
            u, _ = s.newton(u, V)
        J.append(s.terminal_current(u)[0])
    J = np.array(J)
    assert abs(J[2] / J[1] - 1) < 2e-3, J
    return J


def _ideal_hetero(n_pero=150):
    """ETL / absorber / HTL-like stack with long lifetimes and blocking contacts:
    every absorbed photon should be collected at short circuit."""
    from perouq import Interface                                          # noqa: F401
    absorber = Material("abs", Eg=1.55, chi=3.9, eps_r=10, Nc=2e18, Nv=2e18, mu_n=10, mu_p=10,
                        NA=1e13, tau_n=1e-3, tau_p=1e-3,
                        alpha=optics.alpha_sqrt_model(1.55, A=1e5, Eu=0.015))
    etl = Material("etl", Eg=3.2, chi=4.0, eps_r=10, Nc=2e18, Nv=2e18, mu_n=50, mu_p=50, ND=1e18)
    htl = Material("htl", Eg=3.0, chi=2.3, eps_r=3, Nc=2e18, Nv=2e18, mu_n=50, mu_p=50, NA=1e18)
    return Device([Layer(etl, 50, 60), Layer(absorber, 500, n_pero), Layer(htl, 50, 60)],
                  left=Contact(None, Sn=1e7, Sp=0.0), right=Contact(None, Sn=0.0, Sp=1e7))


def test_full_collection():
    from perouq import jv
    lam, E = optics.blackbody_test_spectrum()
    s = DDSolver(_ideal_hetero())
    G = optics.beer_lambert_generation(s, lam, E)
    u = jv.illuminate(s, s.equilibrium(), G)
    Jsc, _ = s.terminal_current(u)
    Jabs = q * np.trapezoid(G, s.x)
    assert abs(Jsc / Jabs - 1) < 0.01, (Jsc, Jabs)
    return Jsc, Jabs


def _carbon_cell(n_pero):
    """FTO / ETL / absorber / carbon (W = 5.0 eV), finite lifetimes, interface recombination."""
    from perouq import Interface
    absorber = Material("abs", Eg=1.55, chi=3.9, eps_r=6.5, Nc=2.2e18, Nv=1.8e19, mu_n=2, mu_p=2,
                        NA=1e13, tau_n=1e-7, tau_p=1e-7,
                        alpha=optics.alpha_sqrt_model(1.55, A=1e5, Eu=0.015))
    etl = Material("etl", Eg=3.2, chi=4.0, eps_r=9, Nc=2.2e18, Nv=1.8e19, mu_n=20, mu_p=10, ND=1e17,
                   tau_n=1e-9, tau_p=1e-9)
    return Device([Layer(etl, 50, 60), Layer(absorber, 500, n_pero)],
                  left=Contact(None, 1e7, 1e7), right=Contact(5.0, 1e7, 1e7),
                  interfaces=[Interface(0, 1e3, 1e3)])


def test_hetero_mesh_convergence():
    from perouq import jv
    lam, E = optics.blackbody_test_spectrum()
    res = []
    for n in (100, 200, 400):
        r = jv.light_jv(_carbon_cell(n), lam, E, V_max=1.3, dV=0.01)
        m = jv.metrics(r["V"], r["J"], r["P_in"])
        res.append((m["Jsc"], m["Voc"], m["PCE"]))
    res = np.array(res)
    rel = np.abs(res[2] / res[1] - 1)
    assert np.all(rel < 2e-3), res
    return res


def test_sq_limit_am15g():
    """With ASTM G173 the SQ maximum is 33.7 % at 1.34 eV (published value)."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "ASTMG173.csv")
    if not os.path.exists(path):
        return None
    lam, E = optics.load_astm_g173(path)
    r = optics.sq_limit(lam, E, 1.34, P_in=1000.0)
    assert abs(r["PCE"] - 0.337) < 0.002, r["PCE"]
    return r


def test_sq_limit_known_value():
    lam, E = optics.blackbody_test_spectrum()
    r = optics.sq_limit(lam, E, 1.34)
    # SQ maximum for a 5778 K sun scaled to 1000 W/m^2 is ~31-33 %
    assert 0.30 < r["PCE"] < 0.34, r


if __name__ == "__main__":
    test_equilibrium_zero_current()
    print("equilibrium: OK")
    for V, Jsim, Jref, ratio, spread in test_shockley_diode():
        print(f"V={V:.2f} V  J_sim={Jsim:.4e}  J_Shockley={Jref:.4e} A/m^2  ratio={ratio:.4f}  "
              f"continuity error={spread:.1e} A/m^2")
    print("mesh convergence J(0.40 V):", test_mesh_convergence())
    test_sq_limit_known_value()
    print("SQ limit: OK")
    Jsc, Jabs = test_full_collection()
    print(f"full collection: Jsc={Jsc/10:.3f} mA/cm2, absorbed={Jabs/10:.3f} mA/cm2, ratio={Jsc/Jabs:.4f}")
    res = test_hetero_mesh_convergence()
    for n, (Jsc, Voc, PCE) in zip((100, 200, 400), res):
        print(f"absorber nodes={n}: Jsc={Jsc/10:.3f} mA/cm2  Voc={Voc:.4f} V  PCE={100*PCE:.3f} %")
