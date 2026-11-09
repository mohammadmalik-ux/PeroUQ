"""Steady-state 1D drift-diffusion solver for heterostructure solar cells.

Unknowns per node: electrostatic potential psi [V] and the quasi-Fermi levels
EFn, EFp [eV].  Energies are referenced to the Fermi level of the left
(electron) contact at equilibrium; the vacuum level is E_vac = -psi.

    Ec = -psi - chi ,  Ev = Ec - Eg
    n  = Nc exp((EFn - Ec)/Vt) ,  p = Nv exp((Ev - EFp)/Vt)

Discretisation: box (finite-volume) method on a non-uniform mesh,
Scharfetter-Gummel fluxes using effective band edges (Ec - Vt ln Nc,
Ev + Vt ln Nv) so that abrupt heterojunctions and DOS steps are handled.
Each heterointerface is represented by two nodes `delta_nm` apart.

Sign convention: current density J is positive in +x (from the electron
contact towards the hole contact).  With this convention the photocurrent
is positive and the dark forward current negative, so P = J * V > 0 is
generated power.  The applied voltage V raises the hole (right) contact.

Newton's method with a column-coloured finite-difference Jacobian
(9 colours: 3 node classes x 3 unknowns) and step-length damping.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .constants import q, eps0, CM3, thermal_voltage

_EXP_MAX = 300.0


def bernoulli(x):
    """B(x) = x / (exp(x) - 1), stable for all x."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-8
    out[small] = 1.0 - x[small] / 2.0
    xs = x[~small]
    with np.errstate(over="ignore"):
        out[~small] = xs / np.expm1(xs)
    return out


def _exp(a):
    return np.exp(np.clip(a, -_EXP_MAX, _EXP_MAX))


class DDSolver:
    def __init__(self, device, T=300.0):
        self.dev = device
        self.Vt = thermal_voltage(T)
        x, lid, pairs = device.build_mesh()
        self.x, self.lid, self.pairs = x, lid, pairs
        self.N = N = len(x)
        P = device.node_arrays(lid)
        self.P = P
        self.eps = P["eps_r"] * eps0
        h = np.diff(x)
        if np.any(h <= 0):
            raise ValueError("mesh is not strictly increasing")
        self.h = h
        # dual (control) volumes
        dV = np.zeros(N)
        dV[:-1] += h / 2
        dV[1:] += h / 2
        self.dV = dV

        def hmean(a):
            return 2 * a[:-1] * a[1:] / (a[:-1] + a[1:])

        self.eps_e = hmean(self.eps)
        self.Dn_e = hmean(P["mu_n"]) * self.Vt
        self.Dp_e = hmean(P["mu_p"]) * self.Vt
        self.Ndop = P["ND"] - P["NA"]
        self.ni = np.sqrt(P["Nc"] * P["Nv"]) * np.exp(-P["Eg"] / (2 * self.Vt))
        # potential that makes each node charge-neutral with EF = 0
        psi0 = -P["chi"] - P["Eg"] / 2 + 0.5 * self.Vt * np.log(P["Nv"] / P["Nc"])
        self.psi_neutral = psi0 + self.Vt * np.arcsinh(self.Ndop / (2 * self.ni))
        # contacts (velocities cm/s -> m/s)
        L, R = device.left, device.right
        self.SnL, self.SpL = L.Sn * 1e-2, L.Sp * 1e-2
        self.SnR, self.SpR = R.Sn * 1e-2, R.Sp * 1e-2
        self.ifaces = []
        for itf in device.interfaces:
            a, b = pairs[itf.k]
            self.ifaces.append((a, b, itf.Sn * 1e-2, itf.Sp * 1e-2))
        self.G = np.zeros(N)

    # ------------------------------------------------------------------ helpers
    def contact_bc(self, V):
        """Boundary potentials and contact Fermi levels for applied voltage V."""
        dev = self.dev
        EFL, EFR = 0.0, -V
        psiL = -dev.left.W if dev.left.W is not None else self.psi_neutral[0]
        psiR = (V - dev.right.W) if dev.right.W is not None else self.psi_neutral[-1] + V
        return psiL, psiR, EFL, EFR

    def unpack(self, u):
        return u[0::3], u[1::3], u[2::3]

    def pack(self, psi, Fn, Fp):
        u = np.empty(3 * self.N)
        u[0::3], u[1::3], u[2::3] = psi, Fn, Fp
        return u

    def carriers(self, psi, Fn, Fp):
        P, Vt = self.P, self.Vt
        Ec = -psi - P["chi"]
        Ev = Ec - P["Eg"]
        n = P["Nc"] * _exp((Fn - Ec) / Vt)
        p = P["Nv"] * _exp((Ev - Fp) / Vt)
        return n, p, Ec, Ev

    def recombination(self, n, p, Fn, Fp):
        """Bulk SRH (midgap trap) + radiative recombination [m^-3 s^-1]."""
        P, Vt = self.P, self.Vt
        np_ = n * p
        excess = np_ * (-np.expm1(-(Fn - Fp) / Vt))          # np - ni^2
        srh = excess / (P["tau_p"] * (n + self.ni) + P["tau_n"] * (p + self.ni))
        rad = P["B_rad"] * excess
        return srh + rad

    # Minimum carrier density used only to keep the continuity equations
    # non-singular where a carrier is vanishingly rare (e.g. holes in a
    # 3 eV-gap ETL, p ~ 1e-27 m^-3).  It adds a current
    # q mu n_floor dEF/dx, ~1e-9 A/m^2 per volt -- far below the solver floor.
    n_floor = 1.0   # m^-3

    def currents(self, n, p, Ec, Ev, Fn=None, Fp=None):
        """Scharfetter-Gummel electron and hole current densities on edges [A/m^2]."""
        P, Vt, h = self.P, self.Vt, self.h
        Ece = Ec - Vt * np.log(P["Nc"])
        Eve = Ev + Vt * np.log(P["Nv"])
        u = (Ece[1:] - Ece[:-1]) / Vt
        w = (Eve[1:] - Eve[:-1]) / Vt
        Jn = q * self.Dn_e / h * (n[1:] * bernoulli(-u) - n[:-1] * bernoulli(u))
        Jp = q * self.Dp_e / h * (p[:-1] * bernoulli(-w) - p[1:] * bernoulli(w))
        if Fn is not None:
            Jn = Jn + q * self.Dn_e / Vt * self.n_floor * (Fn[1:] - Fn[:-1]) / h
            Jp = Jp + q * self.Dp_e / Vt * self.n_floor * (Fp[1:] - Fp[:-1]) / h
        return Jn, Jp

    def interface_rates(self, n, p, Fn, Fp):
        """Interface recombination [m^-2 s^-1] with the nodes that supply n and p."""
        out = []
        Vt = self.Vt
        for a, b, Sn, Sp in self.ifaces:
            if Sn == 0 and Sp == 0:
                continue
            i_n = a if n[a] >= n[b] else b
            i_p = a if p[a] >= p[b] else b
            nn, pp = n[i_n], p[i_p]
            dF = 0.5 * (Fn[a] + Fn[b]) - 0.5 * (Fp[a] + Fp[b])
            nint = np.sqrt(nn * pp) * np.exp(-dF / (2 * Vt))  # midgap trap n1 = p1
            excess = nn * pp * (-np.expm1(-dF / Vt))
            denom = 0.0
            denom += (nn + nint) / Sp if Sp > 0 else np.inf
            denom += (pp + nint) / Sn if Sn > 0 else np.inf
            out.append((i_n, i_p, excess / denom))
        return out

    # ------------------------------------------------------------------ residual
    def residual(self, u, V):
        psi, Fn, Fp = self.unpack(u)
        psiL, psiR, EFL, EFR = self.contact_bc(V)
        n, p, Ec, Ev = self.carriers(psi, Fn, Fp)
        Rb = self.recombination(n, p, Fn, Fp)
        Jn, Jp = self.currents(n, p, Ec, Ev, Fn, Fp)
        P, dV = self.P, self.dV

        # Poisson: d/dx(eps dpsi/dx) + rho = 0
        flux = self.eps_e * (psi[1:] - psi[:-1]) / self.h
        rpsi = np.zeros(self.N)
        rpsi[:-1] += flux
        rpsi[1:] -= flux
        rpsi += q * (p - n + self.Ndop) * dV
        rpsi /= eps0
        sc = self.eps[0] / eps0 / self.h[0]
        rpsi[0] = sc * (psi[0] - psiL)
        rpsi[-1] = sc * (psi[-1] - psiR)

        # equilibrium densities at the contacts (EF of the metal)
        EcL, EcR = -psiL - P["chi"][0], -psiR - P["chi"][-1]
        n0L = P["Nc"][0] * np.exp((EFL - EcL) / self.Vt)
        p0L = P["Nv"][0] * np.exp((EcL - P["Eg"][0] - EFL) / self.Vt)
        n0R = P["Nc"][-1] * np.exp((EFR - EcR) / self.Vt)
        p0R = P["Nv"][-1] * np.exp((EcR - P["Eg"][-1] - EFR) / self.Vt)
        JnL = q * self.SnL * (n[0] - n0L)
        JpL = -q * self.SpL * (p[0] - p0L)
        JnR = -q * self.SnR * (n[-1] - n0R)
        JpR = q * self.SpR * (p[-1] - p0R)

        G = self.G
        rn = np.zeros(self.N)
        rn[:-1] += Jn
        rn[1:] -= Jn
        rn -= q * (Rb - G) * dV
        rn[0] -= JnL
        rn[-1] += JnR

        rp = np.zeros(self.N)
        rp[:-1] += Jp
        rp[1:] -= Jp
        rp -= q * (G - Rb) * dV
        rp[0] -= JpL
        rp[-1] += JpR

        for i_n, i_p, Rs in self.interface_rates(n, p, Fn, Fp):
            rn[i_n] -= q * Rs
            rp[i_p] += q * Rs

        return self.pack(rpsi, rn / q, rp / q)

    def jacobian(self, u, V, r0=None, step=1e-7):
        """Sparse Jacobian by column-coloured forward differences."""
        N = self.N
        if r0 is None:
            r0 = self.residual(u, V)
        rows, cols, vals = [], [], []
        for c in range(9):
            nm, comp = divmod(c, 3)
            nodes = np.arange(nm, N, 3)
            cj = 3 * nodes + comp
            up = u.copy()
            up[cj] += step
            dr = (self.residual(up, V) - r0) / step
            for d in (-1, 0, 1):
                nb = nodes + d
                ok = (nb >= 0) & (nb < N)
                for k in range(3):
                    rr = 3 * nb[ok] + k
                    rows.append(rr)
                    cols.append(cj[ok])
                    vals.append(dr[rr])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        return sp.csc_matrix((vals, (rows, cols)), shape=(3 * N, 3 * N))

    def newton(self, u, V, tol=1e-10, maxit=80, dmax=0.25, growth=10.0):
        """Damped Newton.

        Rows are equilibrated (continuity rows span tens of orders of
        magnitude between majority and minority regions), each update is
        limited component-wise to `dmax` (a floating quasi-Fermi level where
        its carrier is negligible must not shrink the whole step), and a
        backtracking line search on the equilibrated residual guards against
        the exponential overshoot of n and p.  The search is non-monotone
        (growth up to `growth` x is accepted): a quasi-Fermi level that must
        travel ~1 eV through a region where its carrier is rare passes through
        states with larger residuals on the way.
        """
        r = self.residual(u, V)
        for it in range(maxit):
            J = self.jacobian(u, V, r)
            d = 1.0 / np.maximum(abs(J).max(axis=1).toarray().ravel(), 1e-300)
            du = spla.spsolve((sp.diags(d) @ J).tocsc(), -d * r)
            if not np.all(np.isfinite(du)):
                raise RuntimeError(f"Newton step not finite at V={V:.3f}")
            m = np.max(np.abs(du))
            if m < tol:
                return u + du, it + 1
            du = np.clip(du, -dmax, dmax)
            f0 = np.linalg.norm(d * r)
            lam = 1.0
            for _ in range(30):
                un = u + lam * du
                rn = self.residual(un, V)
                if np.all(np.isfinite(rn)) and np.linalg.norm(d * rn) <= growth * f0:
                    break
                lam *= 0.5
            else:
                # no decrease found: accept a small step and let Newton re-linearise
                un = u + lam * du
                rn = self.residual(un, V)
            u, r = un, rn
        raise RuntimeError(f"Newton did not converge at V={V:.3f} (last |du|={m:.2e})")

    # ------------------------------------------------------------------ drivers
    def equilibrium(self):
        psiL, psiR, _, _ = self.contact_bc(0.0)
        psi = self.psi_neutral.copy()
        psi[0], psi[-1] = psiL, psiR
        u = self.pack(psi, np.zeros(self.N), np.zeros(self.N))
        G_saved, self.G = self.G, np.zeros(self.N)
        u, _ = self.newton(u, 0.0)
        self.G = G_saved
        return u

    def terminal_current(self, u, absolute=False):
        """Total current density [A/m^2] and its spread along x.

        The spread is relative to |J| by default, or in A/m^2 with
        absolute=True.  The numerical floor is ~1e-4 A/m^2 (1e-8 A/cm^2),
        so relative values are only meaningful for currents well above it.
        """
        psi, Fn, Fp = self.unpack(u)
        n, p, Ec, Ev = self.carriers(psi, Fn, Fp)
        Jn, Jp = self.currents(n, p, Ec, Ev)
        J = Jn + Jp
        # the edge inside an interface carries the interface-recombination
        # current between its two nodes, so it is left out of the check
        keep = np.ones(len(J), bool)
        for a, _ in self.pairs:
            keep[a] = False
        Jk = J[keep]
        Jm = np.median(Jk)
        spread = Jk.max() - Jk.min()
        if not absolute:
            spread /= max(abs(Jm), 1e-30)
        return Jm, spread

    def state(self, u):
        psi, Fn, Fp = self.unpack(u)
        n, p, Ec, Ev = self.carriers(psi, Fn, Fp)
        return dict(x=self.x, psi=psi, EFn=Fn, EFp=Fp, n=n / CM3, p=p / CM3, Ec=Ec, Ev=Ev)
