"""Verification of the transfer-matrix optics against analytic results.

Run:  python -m pytest -q tests      (or)      python tests/test_tmm.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perouq import tmm   # noqa: E402


def _absorbed(n, d, lam, j, npts=20001):
    R, T, layers = tmm.coherent(n, d, lam)
    tp, rpp, xi = layers[j]
    x = np.linspace(0, d[j], npts)
    q = tmm.field_absorption(n[0], n[j + 1], d[j], lam, tp, rpp, xi, x)
    return R, T, np.trapezoid(q, x), x, q


def test_fresnel_single_interface():
    # an index-matched (n = n_out) layer leaves the bare n_in / n_out interface
    n_in, n_out = 1.0 + 0j, 1.5 + 0j
    R, T, _ = tmm.coherent([n_in, n_out, n_out], [123.0], 550.0)
    Rf = abs((n_in - n_out) / (n_in + n_out)) ** 2
    assert abs(R - Rf) < 1e-12 and abs(R + T - 1) < 1e-12
    return R, Rf


def test_quarter_wave_antireflection():
    n_in, n_out, lam0 = 1.0, 1.5, 600.0
    n1 = np.sqrt(n_in * n_out)
    R, _, _ = tmm.coherent([n_in + 0j, n1 + 0j, n_out + 0j], [lam0 / (4 * n1)], lam0)
    assert R < 1e-12
    R_off, _, _ = tmm.coherent([n_in + 0j, n1 + 0j, n_out + 0j], [lam0 / (4 * n1)], 400.0)
    assert R_off > 1e-4
    return R, R_off


def test_energy_conservation():
    # glass | ITO-like | absorber | lossless exit
    n = [1.5 + 0j, 1.9 + 0.01j, 2.5 + 0.3j, 1.0 + 0j]
    d = [150.0, 300.0]
    worst = 0.0
    for lam in (400.0, 550.0, 700.0):
        R, T, layers = tmm.coherent(n, d, lam)
        A = 0.0
        for j in range(2):
            x = np.linspace(0, d[j], 20001)
            tp, rpp, xi = layers[j]
            A += np.trapezoid(tmm.field_absorption(n[0], n[j + 1], d[j], lam, tp, rpp, xi, x), x)
        worst = max(worst, abs(R + T + A - 1))
    assert worst < 1e-7, worst
    return worst


def test_beer_lambert_limit():
    # weak absorber index-matched to both sides: no reflections, Q/I0 -> alpha exp(-alpha x)
    nr, k, lam, dd = 2.0, 0.002, 600.0, 10000.0
    n = [nr + 0j, nr + 1j * k, nr + 0j]
    R, T, A, x, q = _absorbed(n, [dd], lam, 0)
    alpha = 4 * np.pi * k / lam
    ref = alpha * np.exp(-alpha * x)
    err = np.max(np.abs(q / ref - 1))
    # the residual comes from the small index step n -> n + ik (|r| ~ k / 2n) and scales with k
    assert err < 2.5e-3, err
    return err


def test_incoherent_glass_slab():
    # a lossless glass slab in air, incoherent: R = 2 R1 / (1 + R1)
    from perouq import Material, Layer, Device
    m = Material("air-like", Eg=1, chi=4, eps_r=1, Nc=1e18, Nv=1e18, mu_n=1, mu_p=1)
    dev = Device([Layer(m, 100.0, 5)])
    st = tmm.OpticalStack(nk=[tmm.nk_constant(1.0)], back=tmm.nk_constant(1.0),
                          substrate=tmm.nk_constant(1.5))
    out = tmm.stack_optics(dev, st, np.array([500.0, 700.0]))
    R1 = 0.04
    ref = 2 * R1 / (1 + R1)
    assert np.allclose(out["R"], ref, atol=1e-12), out["R"]
    return out["R"][0], ref


if __name__ == "__main__":
    R, ref = test_incoherent_glass_slab()
    print(f"incoherent glass slab: R={R:.6f}  analytic={ref:.6f}")
    R, Rf = test_fresnel_single_interface()
    print(f"Fresnel: R_TMM={R:.6f}  R_Fresnel={Rf:.6f}")
    R, Roff = test_quarter_wave_antireflection()
    print(f"quarter-wave AR: R(600 nm)={R:.2e}  R(400 nm)={Roff:.4f}")
    print(f"energy conservation: max |R+T+A-1| = {test_energy_conservation():.1e}")
    print(f"Beer-Lambert limit: max relative deviation = {test_beer_lambert_limit():.1e}")
