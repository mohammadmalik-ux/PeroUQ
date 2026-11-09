"""Transfer-matrix optics for thin-film solar cells at normal incidence.

Coherent multilayer on a thick, incoherent glass substrate:

    air | glass (incoherent) | front optical-only layers (e.g. FTO) |
        electrical layers (ETL, absorber, ...) | back medium (semi-infinite, e.g. carbon)

Coherent part: L. A. A. Pettersson, L. S. Roman, O. Inganas,
J. Appl. Phys. 86, 487 (1999).  For layer j with complex index n_j,
xi_j = 2 pi n_j / lambda, interface matrices I_jk and layer matrices L_j:

    E_j(x) = t_j+ [ exp(i xi_j x) + r_j'' exp(i xi_j (2 d_j - x)) ]
    r_j''  = S''_21 / S''_11 ,   t_j+ = 1 / (S'_11 + S'_12 r_j'' exp(2 i xi_j d_j))
    Q_j(x) / I_0 = Re(n_j) alpha_j |E_j(x)|^2 / Re(n_0) ,  alpha_j = 4 pi Im(n_j) / lambda

Incoherent glass: light reaching the coherent stack is (1 - R_ag) / (1 - R_ag R_s),
with R_ag the air/glass reflectance and R_s the stack reflectance seen from the glass.

n,k data: every model below is a function lam_nm -> complex index.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .constants import h_planck, c_light


# ---------------------------------------------------------------- n,k models
def nk_constant(n, k=0.0):
    return lambda lam_nm: np.full(np.shape(lam_nm), complex(n, k))


def nk_cauchy(A, B=0.0, C=0.0, k=0.0):
    """n = A + B / lam^2 + C / lam^4 with lam in micrometres; constant k."""
    def f(lam_nm):
        lu = np.asarray(lam_nm, float) * 1e-3
        return (A + B / lu**2 + C / lu**4) + 1j * k
    return f


def nk_from_alpha(n_real, alpha):
    """Real index n_real (constant or callable) with k = alpha lambda / 4 pi.

    alpha: function lam_nm -> absorption coefficient [cm^-1].
    """
    def f(lam_nm):
        lam = np.asarray(lam_nm, float)
        n = n_real(lam) if callable(n_real) else n_real
        k = alpha(lam) * 1e2 * lam * 1e-9 / (4 * np.pi)
        return n + 1j * k
    return f


def load_nk_csv(path):
    """Read n,k data in the refractiveindex.info CSV layout.

    Accepts either one table 'wl,n,k' or two tables 'wl,n' then 'wl,k'.
    Wavelength in micrometres (as on refractiveindex.info).  Returns a
    function lam_nm -> complex index (linear interpolation).
    """
    blocks, cur = [], []
    with open(path) as f:
        for line in f:
            parts = [p for p in line.replace(";", ",").replace("\t", ",").split(",") if p.strip()]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                if cur:
                    blocks.append(np.array(cur)); cur = []
                continue
            cur.append(vals)
    if cur:
        blocks.append(np.array(cur))
    if len(blocks) == 1 and blocks[0].shape[1] >= 3:
        wl, n, k = blocks[0][:, 0], blocks[0][:, 1], blocks[0][:, 2]
        wl_k = wl
    else:
        wl, n = blocks[0][:, 0], blocks[0][:, 1]
        wl_k, k = (blocks[1][:, 0], blocks[1][:, 1]) if len(blocks) > 1 else (wl, np.zeros_like(wl))

    def f(lam_nm):
        lu = np.asarray(lam_nm, float) * 1e-3
        return np.interp(lu, wl, n) + 1j * np.interp(lu, wl_k, k)
    return f


# ---------------------------------------------------------------- core TMM
def _I(n1, n2):
    r = (n1 - n2) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    return np.array([[1, r], [r, 1]], complex) / t


def _L(n, d, lam):
    xi = 2 * np.pi * n / lam
    return np.array([[np.exp(-1j * xi * d), 0], [0, np.exp(1j * xi * d)]], complex)


def coherent(n, d, lam):
    """Coherent multilayer at one wavelength.

    n : complex indices [n_in, n_1, ..., n_m, n_out]
    d : thicknesses [d_1, ..., d_m] (same length unit as lam)
    Returns R, T and, per layer, (t_plus, r_pp, xi) to evaluate the field.
    T is the power entering the exit medium (meaningful when it is lossless).
    """
    m = len(d)
    I = [_I(n[j], n[j + 1]) for j in range(m + 1)]
    L = [_L(n[j + 1], d[j], lam) for j in range(m)]
    # prefix S'_j = I_01 L_1 ... I_(j-1)j   (j = 1..m)  and suffix S''_j
    pre = [I[0]]
    for j in range(1, m):
        pre.append(pre[-1] @ L[j - 1] @ I[j])
    suf = [None] * m
    acc = I[m]
    suf[m - 1] = acc
    for j in range(m - 2, -1, -1):
        acc = I[j + 1] @ L[j + 1] @ acc
        suf[j] = acc
    S = pre[m - 1] @ L[m - 1] @ suf[m - 1] if m > 0 else I[0]
    R = abs(S[1, 0] / S[0, 0]) ** 2
    T = abs(1 / S[0, 0]) ** 2 * n[-1].real / n[0].real
    layers = []
    for j in range(m):
        xi = 2 * np.pi * n[j + 1] / lam
        rpp = suf[j][1, 0] / suf[j][0, 0]
        tp = 1 / (pre[j][0, 0] + pre[j][0, 1] * rpp * np.exp(2j * xi * d[j]))
        layers.append((tp, rpp, xi))
    return R, T, layers


def field_absorption(n0, nj, dj, lam, tp, rpp, xi, x):
    """Absorbed power density per unit incident intensity, Q/I0 [1/length], at x in [0, dj]."""
    E = tp * (np.exp(1j * xi * x) + rpp * np.exp(1j * xi * (2 * dj - x)))
    alpha = 4 * np.pi * nj.imag / lam
    return nj.real * alpha * np.abs(E) ** 2 / n0.real


# ---------------------------------------------------------------- device stack
@dataclass
class OpticalLayer:
    name: str
    nk: Callable[[np.ndarray], np.ndarray]
    thickness_nm: float


@dataclass
class OpticalStack:
    """Optical description around an electrical Device.

    front : optical-only coherent layers between the glass and the first
            electrical layer (e.g. FTO), in order of light travel.
    nk    : n,k function for each electrical layer (same order as device.layers).
    back  : semi-infinite exit medium (e.g. carbon), n,k function.
    substrate : glass n,k function, treated incoherently; None = light enters
            directly from air into the first coherent layer.
    """
    nk: List[Callable]
    front: List[OpticalLayer] = field(default_factory=list)
    back: Callable = None
    substrate: Optional[Callable] = None


def stack_optics(device, stack, lam_nm, x_eval=None):
    """Spectral R, per-layer absorptance and Q(x)/I_air at electrical-layer positions.

    x_eval: list (one per electrical layer) of positions measured from the
    start of that layer [nm].  Returns dict with
      R[lam], A_front[i][lam], A_elec[k][lam], A_back[lam], Q[k] (len(x), len(lam)) in 1/nm.
    """
    lam_nm = np.asarray(lam_nm, float)
    nl = len(lam_nm)
    nf, ne = len(stack.front), len(device.layers)
    d = [L.thickness_nm for L in stack.front] + [L.thickness_nm for L in device.layers]
    out = dict(R=np.zeros(nl), A_front=np.zeros((nf, nl)), A_elec=np.zeros((ne, nl)),
               A_back=np.zeros(nl))
    if x_eval is not None:
        out["Q"] = [np.zeros((len(xe), nl)) for xe in x_eval]
    back = stack.back if stack.back is not None else nk_constant(1.0)
    for i, lam in enumerate(lam_nm):
        n_in = complex(stack.substrate(lam)) if stack.substrate is not None else 1.0 + 0j
        n_layers = [complex(L.nk(lam)) for L in stack.front] + [complex(f(lam)) for f in stack.nk]
        n = [n_in] + n_layers + [complex(back(lam))]
        Rs, _, layers = coherent(n, d, lam)
        if stack.substrate is not None:
            Rag = abs((1 - n_in) / (1 + n_in)) ** 2
            into = (1 - Rag) / (1 - Rag * Rs)
            out["R"][i] = Rag + (1 - Rag) ** 2 * Rs / (1 - Rag * Rs)
        else:
            into = 1.0
            out["R"][i] = Rs
        A_tot = 0.0
        for j, (tp, rpp, xi) in enumerate(layers):
            dj = d[j]
            xg = np.linspace(0, dj, 801)
            q = field_absorption(n[0], n[j + 1], dj, lam, tp, rpp, xi, xg) * into
            A = np.trapezoid(q, xg)
            A_tot += A
            if j < nf:
                out["A_front"][j, i] = A
            else:
                k = j - nf
                out["A_elec"][k, i] = A
                if x_eval is not None:
                    out["Q"][k][:, i] = field_absorption(n[0], n[j + 1], dj, lam, tp, rpp, xi,
                                                         np.asarray(x_eval[k])) * into
        out["A_back"][i] = max(1.0 - out["R"][i] - A_tot, 0.0)
    return out


def tmm_generation(solver, stack, lam_nm, E, return_optics=False):
    """Generation rate G(x) [m^-3 s^-1] on the solver mesh from the TMM field.

    E: spectral irradiance [W m^-2 nm^-1] on lam_nm.  Every absorbed photon in
    an electrical layer creates one electron-hole pair.
    """
    dev = solver.dev
    starts = np.cumsum([0.0] + [L.thickness_nm for L in dev.layers])[:-1]
    x_nm = solver.x * 1e9
    x_eval = [np.clip(x_nm[solver.lid == k] - starts[k], 0, dev.layers[k].thickness_nm)
              for k in range(len(dev.layers))]
    opt = stack_optics(dev, stack, lam_nm, x_eval)
    phi = E * lam_nm * 1e-9 / (h_planck * c_light)          # photons m^-2 s^-1 nm^-1
    dlam = np.gradient(lam_nm)
    G = np.zeros(solver.N)
    for k in range(len(dev.layers)):
        Qk = opt["Q"][k] * 1e9                                # 1/nm -> 1/m
        G[solver.lid == k] = Qk @ (phi * dlam)
    return (G, opt) if return_optics else G
