"""Spectra, absorption models, Beer-Lambert generation and the
Shockley-Queisser (detailed-balance) limit.

The transfer-matrix optical model is a later module; Beer-Lambert is used
here so that the electrical solver can be tested on its own.
"""
import numpy as np

from .constants import q, h_planck, c_light, kB


# ---------------------------------------------------------------- spectra
def load_astm_g173(path, column="global"):
    """Read the ASTM G173-03 reference spectrum (NREL text/CSV export).

    Returns wavelength [nm] and spectral irradiance [W m^-2 nm^-1].
    Columns in the file: wavelength, extraterrestrial, global tilt, direct.
    """
    col = {"etr": 1, "global": 2, "direct": 3}[column]
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            try:
                vals = [float(v) for v in parts]
            except ValueError:
                continue
            if len(vals) >= 4:
                rows.append(vals)
    a = np.array(rows)
    return a[:, 0], a[:, col]


def blackbody_test_spectrum(T_sun=5778.0, total=1000.0, lam_nm=None):
    """Blackbody spectrum scaled to `total` W/m^2.

    FOR SOLVER TESTING ONLY. Results for publication must use ASTM G173 AM1.5G.
    """
    if lam_nm is None:
        lam_nm = np.arange(280.0, 4000.0, 1.0)
    lam = lam_nm * 1e-9
    E = 2 * h_planck * c_light**2 / lam**5 / np.expm1(h_planck * c_light / (lam * kB * T_sun))
    E = E * 1e-9                                   # per nm
    E *= total / np.trapezoid(E, lam_nm)
    return lam_nm, E


def photon_flux(lam_nm, E):
    """Spectral photon flux [m^-2 s^-1 nm^-1] from spectral irradiance."""
    return E * lam_nm * 1e-9 / (h_planck * c_light)


# ---------------------------------------------------------------- absorption
def alpha_sqrt_model(Eg, A=1e5, Eu=0.015, alpha_tail0=None):
    """Direct-gap absorption, alpha = A sqrt(E - Eg) [cm^-1] above the gap,
    joined continuously to an Urbach tail (energy Eu, eV) below it.

    The tail is matched to the parabolic branch at E = Eg + Eu/2, where the
    two curves have the same value and slope.  Returns f(lambda_nm) -> cm^-1.
    """
    def f(lam_nm):
        E = 1239.84193 / np.asarray(lam_nm, float)
        Em = Eg + Eu / 2.0
        a_m = A * np.sqrt(Eu / 2.0)
        above = A * np.sqrt(np.clip(E - Eg, 0.0, None))
        tail = a_m * np.exp((E - Em) / Eu)
        return np.where(E >= Em, above, tail)
    return f


# ---------------------------------------------------------------- generation
def beer_lambert_generation(solver, lam_nm, E, R_front=0.0):
    """Generation rate G(x) [m^-3 s^-1] on the solver mesh.

    Light enters at x = 0.  Each layer absorbs with its own alpha(lambda);
    layers with alpha = None are transparent.  No reflection inside the stack.
    """
    dev = solver.dev
    phi = photon_flux(lam_nm, E) * (1.0 - R_front)
    dlam = np.gradient(lam_nm)
    G = np.zeros(solver.N)
    x0 = 0.0
    for k, L in enumerate(dev.layers):
        t = L.thickness_nm * 1e-9
        m = L.material
        if m.alpha is not None:
            a = m.alpha(lam_nm) * 1e2                 # cm^-1 -> m^-1
            idx = np.where(solver.lid == k)[0]
            xl = np.clip(solver.x[idx] - x0, 0.0, t)
            G[idx] = np.exp(-np.outer(xl, a)) @ (phi * dlam * a)
            phi = phi * np.exp(-a * t)
        x0 += t
    return G


def jsc_upper_bound(lam_nm, E, Eg):
    """q x (photons above the gap) [A/m^2]: complete absorption, no losses."""
    phi = photon_flux(lam_nm, E)
    lam_g = 1239.84193 / Eg
    m = lam_nm <= lam_g
    return q * np.trapezoid(phi[m], lam_nm[m])


def sq_limit(lam_nm, E, Eg, T=300.0, P_in=None):
    """Shockley-Queisser limit for a step absorber of gap Eg [eV].

    Returns dict with Jsc [A/m^2], Voc [V], FF, PCE (fraction).
    """
    Vt = kB * T / q
    if P_in is None:
        P_in = np.trapezoid(E, lam_nm)
    Jsc = jsc_upper_bound(lam_nm, E, Eg)
    # radiative saturation current: black-body emission above Eg through the
    # front surface into a hemisphere, (2 pi / h^3 c^2) E^2 / (exp(E/kT) - 1)
    Egrid = np.linspace(Eg, Eg + 2.0, 4000)                     # eV
    Ej = Egrid * q
    flux = 2 * np.pi / (h_planck**3 * c_light**2) * Ej**2 / np.expm1(Ej / (kB * T))  # per J
    J0 = q * np.trapezoid(flux, Ej)
    V = np.linspace(0, Eg, 20000)
    J = Jsc - J0 * np.expm1(V / Vt)
    Pm = np.max(J * V)
    Voc = Vt * np.log1p(Jsc / J0)
    return dict(Jsc=Jsc, Voc=Voc, FF=Pm / (Jsc * Voc), PCE=Pm / P_in, J0=J0)
