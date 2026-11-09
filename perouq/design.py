"""Robust design: choose layer thicknesses that are good on average AND
reliably good when material parameters are uncertain.

Objective for a design x (e.g. ETL and absorber thickness):

    F(x) = E[PCE(x, theta)] - lambda * std[PCE(x, theta)]

where theta are the uncertain material parameters.  E and std are
estimated from the same n_inner parameter samples for every design
(common random numbers), so differences between designs are not masked
by sampling noise.  F is maximised by Bayesian optimisation (Gaussian
process + expected improvement) over the design box.

lambda = 0 gives the best average design; lambda = 1 trades one point of
mean efficiency for one point less spread.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
from scipy.stats import norm, qmc
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# GP hyper-parameter fits often hit a bound on small data sets; that is expected here
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from . import uq
from .calib import Param


@dataclass
class RobustDesigner:
    design_params: List[Param]
    model_factory: Callable[[Dict[str, float]], Callable]   # design -> model(u) -> {"PCE": ...}
    d_uncertain: int                                        # dimension of u
    n_inner: int = 32
    lam: float = 1.0
    n_jobs: int = 1
    seed: int = 0
    key: str = "PCE"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.U_inner = qmc.LatinHypercube(self.d_uncertain, seed=self.seed).random(self.n_inner)
        self.X, self.F, self.stats, self.designs = [], [], [], []

    def design_of(self, x):
        return {p.name: p.from_unit(xi) for p, xi in zip(self.design_params, x)}

    def evaluate(self, x):
        des = self.design_of(x)
        model = self.model_factory(des)
        res = uq.evaluate(model, self.U_inner, self.n_jobs)
        y = uq._column(res, self.key)
        y = y[np.isfinite(y)]
        mean, sd = float(np.mean(y)), float(np.std(y, ddof=1))
        F = mean - self.lam * sd
        self.X.append(np.asarray(x, float)); self.F.append(F); self.designs.append(des)
        self.stats.append(dict(mean=mean, std=sd, p5=float(np.percentile(y, 5)), n=len(y)))
        return F

    def _gp(self):
        d = len(self.design_params)
        k = ConstantKernel(1.0) * Matern(length_scale=np.full(d, 0.3), length_scale_bounds=(1e-2, 10), nu=2.5) \
            + WhiteKernel(1e-3, (1e-8, 1e-1))
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3,
                                      random_state=int(self.rng.integers(1 << 31)))
        return gp.fit(np.array(self.X), np.array(self.F))

    def run(self, n_init=6, n_iter=10, verbose=False):
        d = len(self.design_params)
        for x in qmc.LatinHypercube(d, seed=self.seed + 1).random(n_init):
            self.evaluate(x)
        for it in range(n_iter):
            gp = self._gp()
            cand = qmc.Sobol(d, seed=int(self.rng.integers(1 << 31))).random(1024)
            mu, sd = gp.predict(cand, return_std=True)
            sd = np.maximum(sd, 1e-9)
            best = max(self.F)
            z = (mu - best) / sd
            ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
            F = self.evaluate(cand[np.argmax(ei)])
            if verbose:
                print(f"design {it:2d}: {self.designs[-1]}  F={F:.2f}  best={max(self.F):.2f}")
        i = int(np.argmax(self.F))
        gp = self._gp()
        grid = None
        if d == 2:
            g = np.linspace(0, 1, 41)
            G1, G2 = np.meshgrid(g, g)
            grid = dict(u=np.c_[G1.ravel(), G2.ravel()],
                        F=gp.predict(np.c_[G1.ravel(), G2.ravel()]).reshape(G1.shape))
        return dict(best=self.designs[i], best_F=self.F[i], best_stats=self.stats[i],
                    designs=self.designs, F=np.array(self.F), stats=self.stats, surface=grid)
