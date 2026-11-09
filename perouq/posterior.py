"""Bayesian posterior of the calibrated parameters, sampled on the GP emulator.

The calibration (calib.Calibrator) searches with Bayesian optimisation; the
result there is a best fit plus a tolerance region.  This module turns the same
simulations into a posterior distribution:

    prior       uniform on the unit cube (= log-uniform for log-scaled parameters),
    likelihood  independent Gaussian errors with standard deviation equal to the
                stated tolerance of each target:  log L = -C(theta) / 2,
    emulator    C(theta) is replaced by the Gaussian-process mean fitted to
                log(1 + C) on all calibration runs (emulator-based Bayesian
                calibration).  The emulator's own predictive variance is
                propagated by drawing C from the GP at each step (optional).

Sampling: affine-invariant ensemble sampler (Goodman & Weare 2010 stretch move)
written here, since the emulator is cheap and vectorised.
The posterior is then checked with true simulations of random posterior draws
(posterior-predictive check).
"""
import numpy as np


def log_post_factory(gp, d, noisy=False, rng=None):
    rng = rng or np.random.default_rng(0)

    def lp(U):
        U = np.atleast_2d(U)
        out = np.full(len(U), -np.inf)
        ok = np.all((U >= 0) & (U <= 1), axis=1)
        if ok.any():
            if noisy:
                mu, sd = gp.predict(U[ok], return_std=True)
                y = mu + sd * rng.standard_normal(len(mu))
            else:
                y = gp.predict(U[ok])
            C = np.expm1(np.maximum(y, 0.0))
            out[ok] = -0.5 * C
        return out
    return lp


def ensemble_sampler(lp, d, n_walkers=64, n_steps=4000, burn=1000, thin=10, seed=0, x0=None, a=2.0):
    """Goodman-Weare stretch-move sampler.  Returns (samples [n, d], acceptance fraction)."""
    rng = np.random.default_rng(seed)
    if x0 is None:
        X = rng.random((n_walkers, d))
    else:
        X = np.clip(np.asarray(x0)[rng.integers(0, len(x0), n_walkers)]
                    + 0.01 * rng.standard_normal((n_walkers, d)), 0, 1)
    L = lp(X)
    half = n_walkers // 2
    chain, acc = [], 0
    for t in range(n_steps):
        for s in (0, 1):
            idx = np.arange(s * half, (s + 1) * half)
            other = np.arange((1 - s) * half, (2 - s) * half)
            z = ((a - 1) * rng.random(half) + 1) ** 2 / a
            partner = X[rng.choice(other, half)]
            Y = partner + z[:, None] * (X[idx] - partner)
            Ly = lp(Y)
            logr = (d - 1) * np.log(z) + Ly - L[idx]
            take = np.log(rng.random(half)) < logr
            X[idx[take]] = Y[take]
            L[idx[take]] = Ly[take]
            acc += take.sum()
        if t >= burn and (t - burn) % thin == 0:
            chain.append(X.copy())
    return np.concatenate(chain), acc / (n_steps * n_walkers)


def summarise(samples_unit, params):
    """Medians, 95 % credible intervals (physical units) and the correlation matrix (unit scale)."""
    phys = np.array([[p.from_unit(u) for p, u in zip(params, row)] for row in samples_unit])
    out = {}
    for j, p in enumerate(params):
        lo, med, hi = np.percentile(phys[:, j], [2.5, 50, 97.5])
        out[p.name] = dict(median=float(med), ci95=[float(lo), float(hi)],
                           prior=[p.low, p.high], log=p.log,
                           width_fraction_of_prior=float(np.subtract(*np.percentile(samples_unit[:, j], [97.5, 2.5]))))
    corr = np.corrcoef(samples_unit.T)
    return out, corr
