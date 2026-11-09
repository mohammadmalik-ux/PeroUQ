"""Mobile ions in the absorber: steady-state and frozen-ion (fast-scan) J-V curves.

One mobile species (positive iodide vacancies, density P(x)) moves inside the
perovskite only; an equal, uniform density N0 of immobile negative charge keeps
the layer neutral on average (the usual Schottky-defect picture, as in
IonMonger).  The ions carry no current at steady state, so they follow a
Boltzmann distribution in the electrostatic potential,

    P(x) = exp(y - psi(x)/Vt)   in the perovskite,

with the scalar y fixed by conservation, int P dx = N0 * L.  Poisson's equation
gains the charge q (P - N0).  y is found by a secant iteration around the
Newton solve.

Two scan limits:
  steady  - ions re-equilibrate at every voltage (slow scan / stabilised output);
  frozen  - ions equilibrate at a preconditioning voltage V_pre under light and
            stay fixed during the scan (fast scan).  V_pre ~ Voc stands for a
            reverse scan after light soaking, V_pre = 0 for a forward scan from
            short circuit.  The difference between the two gives a hysteresis
            index.
"""
import numpy as np

from .constants import q, eps0, CM3
from .solver import DDSolver
from .optics import beer_lambert_generation
from . import jv


class IonSolver(DDSolver):
    def __init__(self, device, N0_cm3, layer=1, T=300.0):
        super().__init__(device, T)
        self.N0 = N0_cm3 * CM3
        mask = self.lid == layer
        mask[-1] = False                     # contact node keeps its boundary condition
        self.mask = mask
        self.ion_scale = 0.0                 # 0 = ions off, ramped to 1
        self.mode = "mobile"
        self.y = None
        self.P_frozen = None
        self.L = float(np.sum(self.dV[mask]))

    def profile(self, psi):
        if self.mode == "frozen":
            return self.P_frozen
        P = np.zeros(self.N)
        P[self.mask] = np.exp(np.clip(self.y - psi[self.mask] / self.Vt, -700, 700))
        return P

    def residual(self, u, V):
        r = super().residual(u, V)
        if self.ion_scale > 0:
            psi = u[0::3]
            rho = np.where(self.mask, self.profile(psi) - self.N0, 0.0) * self.ion_scale
            r[0::3] += q * rho * self.dV / eps0
        return r

    def _y_exact(self, psi):
        """y that conserves the ion number for the present potential."""
        a = -psi[self.mask] / self.Vt
        m = a.max()
        return np.log(self.N0 * self.L) - m - np.log(np.sum(np.exp(a - m) * self.dV[self.mask]))

    def solve(self, u, V, tol=1e-5, maxit=60, verbose=False):
        """Newton solve with mobile ions in equilibrium (secant iteration on y)."""
        if self.mode == "frozen" or self.ion_scale == 0:
            return self.newton(u, V)[0]
        if self.y is None:
            self.y = self._y_exact(u[0::3])
        hist = []
        for _ in range(maxit):
            u = self.newton(u, V)[0]
            g = self._y_exact(u[0::3]) - self.y          # zero when ions are conserved
            hist.append((self.y, g))
            if verbose:
                print(f'  y={self.y:.6f} g={g:.3e}')
            if abs(g) < tol:
                return u
            if len(hist) >= 2 and hist[-1][1] != hist[-2][1]:
                (y0, g0), (y1, g1) = hist[-2], hist[-1]
                step = -g1 * (y1 - y0) / (g1 - g0)
                self.y = y1 + float(np.clip(step, -3.0, 3.0))
            else:
                self.y = self.y + g                      # fixed-point step
        raise RuntimeError(f"ion conservation not reached at V={V:.3f}")


def _prepare(device, lam_nm, E, N0_cm3, gen_scale, T):
    s = IonSolver(device, N0_cm3, T=T)
    G = beer_lambert_generation(s, lam_nm, E) * gen_scale
    u = s.equilibrium()
    u = jv.illuminate(s, u, G)
    # switch the ions on gradually at short circuit under light
    for f in [1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0]:
        s.ion_scale = f
        s.y = None if f == 1e-4 else s.y
        u = s.solve(u, 0.0)
    return s, u


def ion_jv(device, lam_nm, E, N0_cm3, mode="steady", V_pre=None, V_max=1.3, dV=0.02,
           gen_scale=1.0, Rs=0.0, Rsh=np.inf, T=300.0, P_in=None):
    """Illuminated J-V curve with mobile ions.  Returns dict(V, J, metrics, P) (J in A/m^2)."""
    s, u = _prepare(device, lam_nm, E, N0_cm3, gen_scale, T)
    if mode == "frozen":
        Vp = 0.0
        while Vp < V_pre - 1e-9:                  # walk to the preconditioning voltage
            Vp = min(V_pre, round(Vp + 0.05, 10))
            u = s.solve(u, Vp)
        s.P_frozen = s.profile(u[0::3]).copy()
        s.mode = "frozen"
        # come back to 0 V with frozen ions
        while Vp > 1e-9:
            Vp = max(0.0, round(Vp - 0.05, 10))
            u = s.solve(u, Vp)
    Vs, Js = [], []
    V = 0.0
    while V <= V_max + 1e-12:
        u = s.solve(u, V)
        Jm, _ = s.terminal_current(u)
        Vs.append(V); Js.append(Jm)
        if Jm < 0 and V > 0.1 and Vs[-1] - jv._voc_estimate(Vs, Js) > 0.06:
            break
        V = round(V + dV, 10)
    Vv, Jv = np.array(Vs), np.array(Js)
    if Rs > 0 or np.isfinite(Rsh):
        Vv, Jv = jv.apply_series_shunt(Vv, Jv, Rs, Rsh)
    m = jv.metrics(Vv, Jv, P_in if P_in else np.trapezoid(E, lam_nm))
    return dict(V=Vv, J=Jv, metrics=m, P=s.profile(u[0::3]) / CM3, x=s.x)
