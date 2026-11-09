"""Bayesian calibration of unknown device parameters against published results.

Unknown parameters (e.g. interface recombination velocity, carbon work
function, series and shunt resistance) are found by minimising the misfit
between simulated and published figures of merit, optionally plus a
digitised J-V curve.

Misfit (chi-square with a stated tolerance per quantity):

    C(theta) = sum_k ((m_k(theta) - t_k) / s_k)^2 ,   s_k = rel_k * |t_k|

Optimiser: Bayesian optimisation.  A Gaussian process (Matern 5/2 kernel
with one length scale per parameter, plus a noise term) is fitted to
log(1 + C) on parameters scaled to [0, 1] (log10 scale for parameters that
span decades).  The next point maximises the expected improvement over a
Sobol candidate set plus local perturbations around the best points.

Two phases:
  1. search  - expected improvement finds a best fit;
  2. mapping - the 'straddle' acquisition (1.96 sigma - |mu - T|) spends
     simulations where the surrogate is unsure whether C is above or below
     the acceptance level T = n_targets (on average one tolerance per
     quantity).  Different parameters can compensate each other (e.g. interface
     recombination, back-contact work function and series resistance), so
     the set of acceptable parameters is a region, not a point, and phase 1
     alone would only find one corner of it.

Result: the best fit; the acceptable region estimated on a dense Sobol set
from the surrogate; and a check of that region with true simulations
(precision = fraction of region samples that are truly acceptable).
The spread of the region shows which parameters the data constrain; a
parameter whose region spans most of its prior range is not identifiable
from these data.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.stats import norm, qmc
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# GP hyper-parameter fits often hit a bound on small data sets; that is expected here
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from . import jv

DEFAULT_REL = {"Jsc": 0.02, "Voc": 0.01, "FF": 0.02, "PCE": 0.02}


@dataclass
class Param:
    name: str
    low: float
    high: float
    log: bool = False

    def from_unit(self, u):
        if self.log:
            return float(10 ** (np.log10(self.low) + u * (np.log10(self.high) - np.log10(self.low))))
        return float(self.low + u * (self.high - self.low))

    def to_unit(self, v):
        if self.log:
            return (np.log10(v) - np.log10(self.low)) / (np.log10(self.high) - np.log10(self.low))
        return (v - self.low) / (self.high - self.low)


def device_simulator(build_device, lam_nm, E, stack=None, V_max=1.4, dV=0.02, P_in=None):
    """Wrap a device builder into simulate(theta) -> metrics.

    build_device(theta) -> Device.  theta may contain 'Rs' and 'Rsh' in
    ohm cm^2; they are applied as an external circuit, so they cost nothing.
    'eta_opt' (optional) scales the generation rate (optical-loss factor).
    Returned metrics: Jsc [mA/cm^2], Voc [V], FF, PCE [%], plus the J-V arrays.
    """
    def simulate(theta):
        r = jv.light_jv(build_device(theta), lam_nm, E, V_max=V_max, dV=dV, stack=stack,
                        gen_scale=theta.get("eta_opt", 1.0))
        V, J = r["V"], r["J"]
        Rs = theta.get("Rs", 0.0) * 1e-4
        Rsh = theta.get("Rsh", np.inf) * 1e-4
        if Rs > 0 or np.isfinite(Rsh):
            V, J = jv.apply_series_shunt(V, J, Rs, Rsh)
        m = jv.metrics(V, J, P_in if P_in else r["P_in"])
        return dict(Jsc=m["Jsc"] / 10, Voc=m["Voc"], FF=m["FF"], PCE=100 * m["PCE"],
                    V=V, J=J / 10)
    return simulate


@dataclass
class Calibrator:
    params: List[Param]
    simulate: Callable[[Dict[str, float]], Dict]
    target: Dict[str, float]                 # any of Jsc, Voc, FF, PCE (same units as simulate)
    rel: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REL))
    jv_target: Optional[tuple] = None        # (V [V], J [mA/cm^2]) digitised curve
    jv_tol: float = 0.5                      # mA/cm^2 tolerance per curve point (RMS)
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.X, self.C, self.M, self.thetas = [], [], [], []
        self.failures = 0

    # ---------------------------------------------------------- misfit
    def n_targets(self):
        return len(self.target) + (1 if self.jv_target is not None else 0)

    def misfit(self, m):
        c = 0.0
        for k, t in self.target.items():
            s = self.rel.get(k, 0.02) * abs(t)
            c += ((m[k] - t) / s) ** 2
        if self.jv_target is not None:
            Vt, Jt = self.jv_target
            Js = np.interp(Vt, m["V"], m["J"])
            c += np.mean((Js - Jt) ** 2) / self.jv_tol ** 2
        return float(c)

    def theta_of(self, u):
        return {p.name: p.from_unit(ui) for p, ui in zip(self.params, u)}

    def evaluate(self, u):
        th = self.theta_of(u)
        try:
            m = self.simulate(th)
            c = self.misfit(m)
        except Exception:                     # non-converged device: penalise
            self.failures += 1
            m, c = None, (max(self.C) * 10 if self.C else 1e6)
        self.X.append(np.asarray(u, float)); self.C.append(c); self.M.append(m); self.thetas.append(th)
        return c

    # ---------------------------------------------------------- BO loop
    def _gp(self):
        d = len(self.params)
        k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(d, 0.3),
                                                     length_scale_bounds=(1e-2, 10.0), nu=2.5) \
            + WhiteKernel(1e-4, (1e-8, 1e-1))
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3,
                                      random_state=int(self.rng.integers(1 << 31)))
        gp.fit(np.array(self.X), np.log1p(np.array(self.C)))
        return gp

    def _next(self, gp, n_cand):
        d = len(self.params)
        cand = qmc.Sobol(d, seed=int(self.rng.integers(1 << 31))).random(n_cand)
        y = np.log1p(np.array(self.C))
        best = np.argsort(y)[:5]
        local = np.concatenate([np.clip(self.X[i] + 0.05 * self.rng.standard_normal((n_cand // 10, d)), 0, 1)
                                for i in best])
        cand = np.vstack([cand, local])
        mu, sd = gp.predict(cand, return_std=True)
        sd = np.maximum(sd, 1e-9)
        z = (y.min() - mu) / sd
        ei = (y.min() - mu) * norm.cdf(z) + sd * norm.pdf(z)
        return cand[np.argmax(ei)]

    def _next_level(self, gp, n_cand):
        """Straddle acquisition: most uncertain points near the acceptance level."""
        d = len(self.params)
        cand = qmc.Sobol(d, seed=int(self.rng.integers(1 << 31))).random(n_cand)
        mu, sd = gp.predict(cand, return_std=True)
        T = np.log1p(self.n_targets())
        return cand[np.argmax(1.96 * sd - np.abs(mu - T))]

    def run(self, n_init=None, n_iter=30, n_map=30, n_check=10, n_cand=2048, verbose=False):
        d = len(self.params)
        n_init = n_init or max(8, 2 * d + 2)
        for u in qmc.LatinHypercube(d, seed=int(self.rng.integers(1 << 31))).random(n_init):
            self.evaluate(u)
        for it in range(n_iter):
            c = self.evaluate(self._next(self._gp(), n_cand))
            if verbose:
                print(f"search {it:3d}  C={c:10.3f}  best={min(self.C):10.3f}")
        for it in range(n_map):
            c = self.evaluate(self._next_level(self._gp(), n_cand))
            if verbose:
                print(f"map    {it:3d}  C={c:10.3f}")
        return self.result(n_check=n_check)

    # ---------------------------------------------------------- output
    def region(self, n=16384, gp=None, local=400, sigma=0.05):
        """Unit-scaled samples where the surrogate predicts C <= n_targets.

        Uniform Sobol samples give the volume fraction; in several dimensions a
        small region can be missed by them, so clouds of `local` samples
        around every accepted evaluation are added to map its extent.
        Returns (samples, volume fraction from the uniform part).
        """
        gp = gp or self._gp()
        d = len(self.params)
        T = np.log1p(self.n_targets())
        U = qmc.Sobol(d, seed=int(self.rng.integers(1 << 31))).random(n)
        inside = U[gp.predict(U) <= T]
        frac = len(inside) / n
        acc = [self.X[j] for j in np.where(np.array(self.C) <= self.n_targets())[0]]
        if acc and local:
            L = np.concatenate([np.clip(x + sigma * self.rng.standard_normal((local, d)), 0, 1) for x in acc])
            inside = np.vstack([inside, L[gp.predict(L) <= T]])
        return inside, frac

    def result(self, n_check=10):
        C = np.array(self.C)
        i = int(np.argmin(C))
        T = self.n_targets()
        acc = [self.thetas[j] for j in np.where(C <= T)[0]]
        U, frac = self.region()
        ranges = {}
        for k, p in enumerate(self.params):
            pts = np.concatenate([U[:, k], [self.X[j][k] for j in np.where(C <= T)[0]]])
            if len(pts):
                lo, hi = p.from_unit(pts.min()), p.from_unit(pts.max())
                ranges[p.name] = (lo, hi, pts.max() - pts.min())
        # check the surrogate region with true simulations
        precision = None
        if n_check and len(U):
            pick = U[self.rng.choice(len(U), size=min(n_check, len(U)), replace=False)]
            ok = [self.evaluate(u) <= T for u in pick]
            precision = float(np.mean(ok))
        return dict(best=self.thetas[i], best_cost=float(C[i]), best_metrics=self.M[i],
                    accepted=acc, region=[self.theta_of(u) for u in U], ranges=ranges,
                    region_fraction=frac, precision=precision,
                    n_eval=len(self.C), failures=self.failures, history=np.array(self.C))
