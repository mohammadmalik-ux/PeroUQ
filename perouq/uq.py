"""Uncertainty propagation (Monte Carlo) and global sensitivity (Sobol indices).

All models are functions f(u) of a point u in the unit cube [0,1]^d that
return a dict of scalar outputs (e.g. PCE, Voc).  The mapping from u to
physical parameters (log-uniform for densities, mobilities, ...) lives in
the model (see study.py), so the estimators here are model-agnostic.

Sobol indices (Saltelli et al., Comput. Phys. Commun. 181 (2010) 259):
two independent sample matrices A, B (N x d) and A_B^(i) = A with column i
from B; N (d + 2) model runs.

    S_i  = mean( f(B) (f(A_B^i) - f(A)) ) / V          first order
    ST_i = mean( (f(A) - f(A_B^i))^2 ) / (2 V)          total (Jansen)

V is the variance over A and B together.  Confidence intervals by
bootstrap over the N rows.  Failed model runs (NaN) drop their row from
every estimate.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc


# ---------------------------------------------------------------- evaluation
def evaluate(func, U, n_jobs=1, chunksize=4):
    """Evaluate func on each row of U; returns a list of dicts (or None on failure)."""
    def safe(u):
        try:
            return func(u)
        except Exception:
            return None
    if n_jobs == 1:
        return [safe(u) for u in U]
    with ProcessPoolExecutor(n_jobs) as ex:
        return list(ex.map(_SafeCall(func), list(U), chunksize=chunksize))


class _SafeCall:
    def __init__(self, func):
        self.func = func

    def __call__(self, u):
        try:
            return self.func(u)
        except Exception:
            return None


def _column(results, key):
    return np.array([np.nan if r is None or key not in r else float(r[key]) for r in results])


# ---------------------------------------------------------------- Monte Carlo
def propagate(func, d, n, seed=0, n_jobs=1):
    """Latin hypercube Monte Carlo.  Returns (U, results)."""
    U = qmc.LatinHypercube(d, seed=seed).random(n)
    return U, evaluate(func, U, n_jobs)


def exceedance(values, threshold):
    """P(value > threshold) with a 95 % Wilson interval; NaNs (failed runs) are excluded."""
    v = values[np.isfinite(values)]
    n = len(v)
    k = int(np.sum(v > threshold))
    if n == 0:
        return dict(p=np.nan, low=np.nan, high=np.nan, n=0)
    z = 1.96
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return dict(p=p, low=max(c - h, 0.0), high=min(c + h, 1.0), n=n)


def summarize(values):
    v = values[np.isfinite(values)]
    q = np.percentile(v, [5, 25, 50, 75, 95]) if len(v) else [np.nan] * 5
    return dict(mean=float(np.mean(v)) if len(v) else np.nan, std=float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                p5=q[0], p25=q[1], median=q[2], p75=q[3], p95=q[4], n=len(v),
                failed=int(np.sum(~np.isfinite(values))))


# ---------------------------------------------------------------- Sobol
def saltelli_design(d, N, seed=0):
    """A, B (N x d) and the d matrices A_B^(i).  N should be a power of 2."""
    S = qmc.Sobol(2 * d, scramble=True, seed=seed).random(N)
    A, B = S[:, :d], S[:, d:]
    AB = []
    for i in range(d):
        M = A.copy()
        M[:, i] = B[:, i]
        AB.append(M)
    return A, B, AB


def sobol_from_outputs(fA, fB, fAB, n_boot=500, seed=0):
    """First-order and total Sobol indices with bootstrap 95 % intervals."""
    fAB = np.asarray(fAB)                         # (d, N)
    ok = np.isfinite(fA) & np.isfinite(fB) & np.all(np.isfinite(fAB), axis=0)
    fA, fB, fAB = fA[ok], fB[ok], fAB[:, ok]
    N = len(fA)

    def est(idx):
        a, b, ab = fA[idx], fB[idx], fAB[:, idx]
        V = np.var(np.concatenate([a, b]), ddof=1)
        S1 = np.mean(b[None, :] * (ab - a[None, :]), axis=1) / V
        ST = 0.5 * np.mean((a[None, :] - ab) ** 2, axis=1) / V
        return S1, ST

    S1, ST = est(np.arange(N))
    rng = np.random.default_rng(seed)
    boots = [est(rng.integers(0, N, N)) for _ in range(n_boot)]
    b1 = np.array([b[0] for b in boots])
    bT = np.array([b[1] for b in boots])
    return dict(S1=S1, ST=ST,
                S1_ci=np.percentile(b1, [2.5, 97.5], axis=0),
                ST_ci=np.percentile(bT, [2.5, 97.5], axis=0),
                n_used=N, n_dropped=int(np.sum(~ok)))


def sobol(func, d, N, keys, seed=0, n_jobs=1, n_boot=500):
    """Run the Saltelli design through func and return indices for each output key."""
    A, B, AB = saltelli_design(d, N, seed)
    U = np.vstack([A, B] + AB)
    res = evaluate(func, U, n_jobs)
    out = {}
    for k in keys:
        y = _column(res, k)
        fA, fB = y[:N], y[N:2 * N]
        fAB = y[2 * N:].reshape(d, N)
        out[k] = sobol_from_outputs(fA, fB, fAB, n_boot=n_boot, seed=seed)
        out[k]["raw"] = dict(fA=fA, fB=fB, fAB=fAB)     # for convergence checks on prefixes
    out["n_runs"] = len(U)
    return out
