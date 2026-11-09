"""The study model: FTO / ETL / MAPbI3 / carbon without hole-transport layer.

CarbonCell(etl) maps a point u in [0,1]^d to a device whose uncertain
parameters are drawn from the literature ranges of the database (with the
author's decisions applied), and returns its figures of merit.  It is a
picklable callable, so uq.propagate / uq.sobol can run it in parallel.

Dimensions of u, in order:
  MAPbI3 parameters that differ between sources (sorted by name),
  ETL parameters that differ between sources (sorted by name),
  extra uncertain quantities given in `extra` (default: ETL/MAPbI3 interface
  recombination velocity S and carbon work function W).
Parameters reported identically by every source are held fixed.
"""
import numpy as np

from . import params, optics, jv
from .device import Layer, Contact, Interface, Device
from .calib import Param


def default_extra():
    """Interface S and carbon W ranges from the database."""
    s = params.table("ETL/MAPbI3")["S"]
    w = params.table("carbon")["W"]
    return [Param("S", s["min"], s["max"], log=True), Param("W", w["min"], w["max"])]


class CarbonCell:
    def __init__(self, etl, lam_nm, E, etl_nm=50.0, pero_nm=500.0, extra=None, fixed=None,
                 stack=None, n_etl=50, n_pero=120, V_max=1.4, dV=0.02, Rs=0.0, Rsh=np.inf,
                 eta_opt=1.0, P_in=1000.0, min_spread=None, etl_absorbs=True):
        self.etl, self.lam, self.E = etl, lam_nm, E
        self.etl_nm, self.pero_nm = etl_nm, pero_nm
        self.stack, self.n_etl, self.n_pero = stack, n_etl, n_pero
        self.V_max, self.dV, self.Rs, self.Rsh, self.eta_opt = V_max, dV, Rs, Rsh, eta_opt
        self.P_in = P_in   # W/m^2; AM1.5G is normalised to 1000
        self.fixed = fixed or {}
        self.min_spread = min_spread
        self.etl_absorbs = etl_absorbs
        rp = {p: r for p, r in params.ranges("MAPbI3", min_spread=min_spread).items() if r[0] != r[1]}
        re = {p: r for p, r in params.ranges(etl, min_spread=min_spread).items() if r[0] != r[1]}
        self.p_names = sorted(rp)
        self.e_names = sorted(re)
        self.extra = extra if extra is not None else default_extra()
        self.names = ([f"MAPbI3.{p}" for p in self.p_names] + [f"{etl}.{p}" for p in self.e_names]
                      + [p.name for p in self.extra])
        self.d = len(self.names)

    def theta(self, u):
        u = np.asarray(u, float)
        npp, ne = len(self.p_names), len(self.e_names)
        vp, _ = params.sample_values("MAPbI3", u[:npp], min_spread=self.min_spread)
        ve, _ = params.sample_values(self.etl, u[npp:npp + ne], min_spread=self.min_spread)
        vx = {p.name: p.from_unit(ui) for p, ui in zip(self.extra, u[npp + ne:])}
        vx.update(self.fixed)
        return vp, ve, vx

    def device(self, u):
        vp, ve, vx = self.theta(u)
        Eg = vp.get("Eg", params.table("MAPbI3")["Eg"]["median"])
        pero = params.material("MAPbI3", vp, alpha=optics.alpha_sqrt_model(Eg))
        # the ETL absorbs above its own gap (parasitic for wide-gap oxides, significant for CdS);
        # same sqrt(E - Eg) model as the absorber until measured n,k data are used
        Eg_etl = ve.get("Eg", params.table(self.etl)["Eg"]["median"])
        etl = params.material(self.etl, ve, alpha=optics.alpha_sqrt_model(Eg_etl, A=1e5, Eu=0.03)
                              if self.etl_absorbs else None)
        S = vx.get("S", 0.0)
        return Device([Layer(etl, self.etl_nm, self.n_etl), Layer(pero, self.pero_nm, self.n_pero)],
                      left=Contact(None), right=Contact(vx.get("W", 5.0)),
                      interfaces=[Interface(0, S, S)])

    def __call__(self, u):
        r = jv.light_jv(self.device(u), self.lam, self.E, V_max=self.V_max, dV=self.dV,
                        stack=self.stack, gen_scale=self.eta_opt)
        V, J = r["V"], r["J"]
        if self.Rs > 0 or np.isfinite(self.Rsh):
            V, J = jv.apply_series_shunt(V, J, self.Rs * 1e-4, self.Rsh * 1e-4)
        m = jv.metrics(V, J, self.P_in if self.P_in else r["P_in"])
        return dict(PCE=100 * m["PCE"], Voc=m["Voc"], Jsc=m["Jsc"] / 10, FF=m["FF"])


# ------------------------------------------------------------------ transfer-matrix optics
# Real refractive indices typical of the literature; k follows from the same absorption
# model as the electrical layers.  Measured n,k spectra can replace them (tmm.load_nk_csv).
N_ETL = {"TiO2": 2.40, "SnO2": 1.95, "ZnO": 2.00, "CdS": 2.45}
OPTICAL_SCENARIOS = {   # name: (n MAPbI3, carbon n, carbon k, FTO thickness nm)
    "base": (2.5, 2.0, 0.8, 400.0),
    "low n absorber": (2.3, 2.0, 0.8, 400.0),
    "high n absorber": (2.6, 2.0, 0.8, 400.0),
    "graphite-like back": (2.5, 2.7, 1.4, 400.0),
    "thick FTO": (2.5, 2.0, 0.8, 600.0),
}


def default_stack(etl, Eg_pero, Eg_etl, scenario="base", nk=None):
    """Glass / FTO / ETL / MAPbI3 / carbon optical stack.

    nk: optional dict of measured n,k functions (keys 'ETL', 'MAPbI3', 'FTO', 'carbon'),
    e.g. from tmm.load_nk_csv, which replace the scenario values.
    """
    from . import tmm
    n_p, n_c, k_c, d_fto = OPTICAL_SCENARIOS[scenario]
    nk = nk or {}
    return tmm.OpticalStack(
        nk=[nk.get("ETL") or tmm.nk_from_alpha(N_ETL[etl], optics.alpha_sqrt_model(Eg_etl, A=1e5, Eu=0.03)),
            nk.get("MAPbI3") or tmm.nk_from_alpha(n_p, optics.alpha_sqrt_model(Eg_pero))],
        front=[tmm.OpticalLayer("FTO", nk.get("FTO") or tmm.nk_constant(1.9, 0.005), d_fto)],
        back=nk.get("carbon") or tmm.nk_constant(n_c, k_c), substrate=tmm.nk_constant(1.52))


class TMMCarbonCell(CarbonCell):
    """CarbonCell with transfer-matrix generation, rebuilt for each sample's band gaps (eta_opt = 1)."""
    def __init__(self, *a, scenario="base", nk=None, **k):
        k.setdefault("eta_opt", 1.0)
        super().__init__(*a, **k)
        self.scenario, self.nk = scenario, nk

    def __call__(self, u):
        vp, ve, _ = self.theta(u)
        Eg_p = vp.get("Eg", params.table("MAPbI3")["Eg"]["median"])
        Eg_e = ve.get("Eg", params.table(self.etl)["Eg"]["median"])
        self.stack = default_stack(self.etl, Eg_p, Eg_e, self.scenario, self.nk)
        return super().__call__(u)


def make_cell(optics_model, etl, lam, E, **k):
    """'bl' -> CarbonCell (Beer-Lambert x eta_opt), 'tmm' -> TMMCarbonCell."""
    if optics_model == "tmm":
        k.pop("eta_opt", None)
        return TMMCarbonCell(etl, lam, E, **k)
    return CarbonCell(etl, lam, E, **k)
