"""J-V sweeps, figures of merit and the external Rs / Rsh circuit."""
import numpy as np

from .solver import DDSolver
from .optics import beer_lambert_generation


def illuminate(solver, u_eq, G, steps=None):
    """Ramp the generation from 0 to G at V = 0 (logarithmic continuation)."""
    if steps is None:
        steps = np.concatenate([10.0 ** np.arange(-14, 0, 1.0), [0.3, 0.6, 1.0]])
    u = u_eq
    for s in steps:
        solver.G = s * G
        u, _ = solver.newton(u, 0.0)
    return u


def sweep(solver, u0, V_list):
    """Solve at each voltage in V_list, starting from u0 (solution at V_list[0]'s neighbour)."""
    J, spread, states = [], [], []
    u = u0
    for V in V_list:
        u, _ = solver.newton(u, V)
        Jm, s = solver.terminal_current(u)
        J.append(Jm)
        spread.append(s)
        states.append(u.copy())
    return np.array(J), np.array(spread), states


def light_jv(device, lam_nm, E, V_max=1.4, dV=0.02, T=300.0, stop_after_voc=0.1,
             stack=None, gen_scale=1.0):
    """Illuminated J-V curve of `device` under spectrum (lam_nm, E).

    stack: tmm.OpticalStack for transfer-matrix generation; None = Beer-Lambert.
    gen_scale: multiplies G(x); an optical-loss factor (reflection, parasitic
    absorption) used with Beer-Lambert until measured n,k data are available.
    Returns dict with V [V], J [A/m^2], spread (current-continuity check),
    the solver and the solution vectors.
    """
    s = DDSolver(device, T)
    if stack is None:
        G = beer_lambert_generation(s, lam_nm, E)
    else:
        from .tmm import tmm_generation
        G = tmm_generation(s, stack, lam_nm, E)
    G = G * gen_scale
    u = s.equilibrium()
    u = illuminate(s, u, G)
    Vs, Js, sp, states = [], [], [], []
    V = 0.0
    while V <= V_max + 1e-12:
        u, _ = s.newton(u, V)
        Jm, spr = s.terminal_current(u)
        Vs.append(V); Js.append(Jm); sp.append(spr); states.append(u.copy())
        if Jm < 0 and V > 0 and Vs[-1] - _voc_estimate(Vs, Js) > stop_after_voc:
            break
        V = round(V + dV, 10)
    return dict(V=np.array(Vs), J=np.array(Js), spread=np.array(sp),
                solver=s, states=states, G=G, P_in=np.trapezoid(E, lam_nm))


def monochromatic_generation(solver, lam_nm, flux, stack=None):
    """G(x) [m^-3 s^-1] for a monochromatic beam of `flux` photons m^-2 s^-1 at lam_nm."""
    from .constants import h_planck, c_light
    lam = np.array([lam_nm - 1.0, lam_nm, lam_nm + 1.0])   # np.gradient -> 1 nm band
    e = flux * h_planck * c_light / (lam_nm * 1e-9)          # W m^-2 in a 1 nm band
    E = np.array([0.0, e, 0.0])
    if stack is None:
        return beer_lambert_generation(solver, lam, E)
    from .tmm import tmm_generation
    return tmm_generation(solver, stack, lam, E)


def eqe(device, wavelengths, flux=1e19, T=300.0, stack=None, gen_scale=1.0,
        bias=None):
    """External quantum efficiency at short circuit.

    wavelengths [nm]; flux: monochromatic photon flux [m^-2 s^-1] (1e19 m^-2 s^-1
    = 1e15 cm^-2 s^-1, a typical EQE set-up).  bias: optional (lam_nm, E) white bias
    light; EQE is then the differential response on top of it.
    gen_scale multiplies every generation profile (optical-loss factor).
    Returns an array EQE(lambda) in [0, 1].
    """
    from .constants import q
    s = DDSolver(device, T)
    u = s.equilibrium()
    G0 = np.zeros(s.N)
    J0 = 0.0
    if bias is not None:
        G0 = beer_lambert_generation(s, *bias) if stack is None else \
            __import__("perouq.tmm", fromlist=["tmm_generation"]).tmm_generation(s, stack, *bias)
        G0 = G0 * gen_scale
        u = illuminate(s, u, G0)
        J0, _ = s.terminal_current(u)
    out = []
    u_prev = None
    for lam in np.atleast_1d(wavelengths):
        G = G0 + gen_scale * monochromatic_generation(s, float(lam), flux, stack)
        try:
            if u_prev is None:
                raise RuntimeError
            s.G = G
            un, _ = s.newton(u_prev, 0.0)
        except Exception:
            un = illuminate(s, u, G)
        J, _ = s.terminal_current(un)
        u_prev = un
        out.append((J - J0) / (q * flux))
    return np.array(out)


def voc_at(device, lam_nm, E, suns=1.0, T=300.0, Rs=0.0, Rsh=np.inf, stack=None, gen_scale=1.0,
           V_max=1.4, dV=0.01):
    """Open-circuit voltage at a light intensity `suns` (spectrum scaled) and temperature T."""
    r = light_jv(device, lam_nm, suns * np.asarray(E), V_max=V_max, dV=dV, T=T, stack=stack,
                 gen_scale=gen_scale)
    V, J = r["V"], r["J"]
    if Rs > 0 or np.isfinite(Rsh):
        V, J = apply_series_shunt(V, J, Rs, Rsh)
    return float(_voc_estimate(V, J))


def _voc_estimate(Vs, Js):
    Vs, Js = np.asarray(Vs), np.asarray(Js)
    i = np.where(Js < 0)[0]
    if len(i) == 0 or i[0] == 0:
        return np.inf
    k = i[0]
    return Vs[k - 1] - Js[k - 1] * (Vs[k] - Vs[k - 1]) / (Js[k] - Js[k - 1])


def apply_series_shunt(V, J, Rs=0.0, Rsh=np.inf):
    """External series and shunt resistance [ohm m^2] (1 ohm cm^2 = 1e-4 ohm m^2).

    J is in the generator convention (photocurrent > 0).  The shunt sits in
    parallel with the internal diode and Rs in series with the terminals.
    """
    Jt = J - (V / Rsh if np.isfinite(Rsh) else 0.0)
    Vt = V - Jt * Rs
    order = np.argsort(Vt)
    return Vt[order], Jt[order]


def metrics(V, J, P_in):
    """Jsc [A/m^2], Voc [V], FF, PCE (fraction) from a J-V curve (photocurrent > 0)."""
    V, J = np.asarray(V), np.asarray(J)
    Jsc = np.interp(0.0, V, J)
    Voc = _voc_estimate(V, J)
    Vf = np.linspace(0, min(Voc, V.max()), 2001)
    P = Vf * np.interp(Vf, V, J)
    k = np.argmax(P)
    return dict(Jsc=Jsc, Voc=Voc, FF=P[k] / (Jsc * Voc), PCE=P[k] / P_in, Vmpp=Vf[k])
