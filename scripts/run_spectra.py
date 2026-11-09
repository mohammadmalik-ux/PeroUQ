"""Predicted EQE spectra and J-V curves at several temperatures and light intensities.

    python scripts/run_spectra.py --spectrum data/ASTMG173.csv --W 4.95 5.35 --min-spread --out results/spectra

For each ETL the 'median device' (every uncertain parameter at the middle of its range,
reference 50/500 nm design, transfer-matrix optics) is simulated:
  * EQE from 300 to 850 nm (10 nm steps), short circuit, no bias light, and its
    AM1.5G-integrated current compared with Jsc from the J-V curve;
  * J-V curves at 275, 300, 325 and 350 K with the 'full' temperature model of
    run_temperature.py (Nc, Nv ~ T^1.5, MAPbI3 mobilities ~ T^-1.5, dEg/dT = +0.3 meV/K);
  * J-V curves at 0.1, 0.3 and 1 sun (300 K), with the ideality factor from Voc(intensity).
These are predictions for comparison with future measurements, not fits.
Output: OUT/spectra.json and OUT/spectra.png
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum      # noqa: E402
from perouq import calib, jv, params                     # noqa: E402
from perouq.constants import thermal_voltage             # noqa: E402
from perouq.params import DEFAULT_MIN_SPREAD             # noqa: E402
from perouq.study import TMMCarbonCell, default_extra, default_stack   # noqa: E402
from run_temperature import DEG_DT                       # noqa: E402

TEMPS = [275.0, 300.0, 325.0, 350.0]
SUNS = [0.1, 0.3, 1.0]
WL = np.arange(300.0, 851.0, 10.0)


def theta_at(cell, base, u, T):
    vp, ve, vx = base(u)
    vp, ve = dict(vp), dict(ve)
    f = T / 300.0
    tp, te = params.table("MAPbI3"), params.table(cell.etl)
    for v, tab in ((vp, tp), (ve, te)):
        for k in ("Nc", "Nv"):
            v[k] = v.get(k, tab[k]["median"]) * f ** 1.5
    for k in ("mu_n", "mu_p"):
        vp[k] = vp.get(k, tp[k]["median"]) * f ** -1.5
    vp["Eg"] = vp.get("Eg", tp["Eg"]["median"]) + DEG_DT * (T - 300.0)
    return vp, ve, vx


def run_cell(cell, u, T=300.0, suns=1.0):
    base = cell.theta
    cell.theta = (lambda uu: theta_at(cell, base, uu, T)) if T != 300.0 else base
    try:
        vp, ve, _ = cell.theta(u)
        stack = default_stack(cell.etl, vp.get("Eg", params.table("MAPbI3")["Eg"]["median"]),
                              ve.get("Eg", params.table(cell.etl)["Eg"]["median"]))
        dev = cell.device(u)
        r = jv.light_jv(dev, cell.lam, cell.E * suns, V_max=1.4, dV=0.01, T=T, stack=stack)
        m = jv.metrics(r["V"], r["J"], 1000.0 * suns)
        return dev, stack, r, m
    finally:
        cell.theta = base


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--W", type=float, nargs=2)
    ap.add_argument("--min-spread", action="store_true")
    ap.add_argument("--out", default="results/spectra")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    extra = default_extra()
    if a.W:
        extra = [p for p in extra if p.name != "W"] + [calib.Param("W", a.W[0], a.W[1])]
    ms = DEFAULT_MIN_SPREAD if a.min_spread else None
    jpath = os.path.join(a.out, "spectra.json")
    res = json.load(open(jpath)) if os.path.exists(jpath) else {}
    # AM1.5G photon flux per nm on the EQE grid, for the integrated current
    h, c, q = 6.62607015e-34, 2.99792458e8, 1.602176634e-19
    for etl in a.etl:
        if etl in res:
            continue
        cell = TMMCarbonCell(etl, lam, E, extra=extra, min_spread=ms)
        u = np.full(cell.d, 0.5)
        out = dict(T={}, suns={})
        dev, stack, r, m = run_cell(cell, u)
        Q = jv.eqe(dev, WL, stack=stack)
        phi = np.interp(WL, lam, E) * WL * 1e-9 / (h * c)            # photons m^-2 s^-1 nm^-1
        J_eqe = q * np.trapezoid(Q * phi, WL) / 10.0                  # mA cm^-2 (300-850 nm)
        out["eqe"] = dict(wl=WL.tolist(), eqe=np.round(Q, 5).tolist(), J_int=float(J_eqe),
                          Jsc_jv=float(m["Jsc"] / 10))
        for T in TEMPS:
            _, _, r, m = run_cell(cell, u, T=T)
            out["T"][str(int(T))] = dict(V=r["V"].tolist(), J=(r["J"] / 10).tolist(),
                                         **{k: float(m[k]) for k in ("Voc", "FF")},
                                         Jsc=float(m["Jsc"] / 10), PCE=float(100 * m["PCE"]))
        for s in SUNS:
            _, _, r, m = run_cell(cell, u, suns=s)
            out["suns"][str(s)] = dict(V=r["V"].tolist(), J=(r["J"] / 10).tolist(),
                                       **{k: float(m[k]) for k in ("Voc", "FF")},
                                       Jsc=float(m["Jsc"] / 10), PCE=float(100 * m["PCE"]))
        voc = np.array([out["suns"][str(s)]["Voc"] for s in SUNS])
        out["n_id"] = float(np.polyfit(np.log(SUNS), voc, 1)[0] / thermal_voltage(300.0))
        vt = np.array([out["T"][str(int(T))]["Voc"] for T in TEMPS])
        out["dVoc_dT_mV_per_K"] = float(1e3 * np.polyfit(TEMPS, vt, 1)[0])
        res[etl] = out
        print(etl, f"Jsc {out['eqe']['Jsc_jv']:.2f} vs EQE {J_eqe:.2f} mA/cm2; n_id {out['n_id']:.2f}; "
              f"dVoc/dT {out['dVoc_dT_mV_per_K']:.2f} mV/K", flush=True)
        with open(jpath, "w") as f:
            json.dump(res, f)
    figure(res, os.path.join(a.out, "spectra.png"))


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"TiO2": "#1f5fa8", "SnO2": "#d0731c", "ZnO": "#3b8f3b", "CdS": "#a8322d"}
    lab = {"TiO2": "TiO$_2$", "SnO2": "SnO$_2$", "ZnO": "ZnO", "CdS": "CdS"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for e, o in res.items():
        ax[0].plot(o["eqe"]["wl"], np.array(o["eqe"]["eqe"]) * 100, color=col[e],
                   label=f"{lab[e]} ({o['eqe']['J_int']:.1f} mA cm$^{{-2}}$)")
    ax[0].set(xlabel="Wavelength (nm)", ylabel="EQE (%)", ylim=(0, 100), title="(a) EQE, median devices")
    ax[0].legend(frameon=False, fontsize=8, loc="lower center")
    e = "TiO2" if "TiO2" in res else list(res)[0]
    cm = plt.get_cmap("coolwarm")
    for i, (T, o) in enumerate(res[e]["T"].items()):
        ax[1].plot(o["V"], o["J"], color=cm(i / 3), label=f"{T} K (PCE {o['PCE']:.1f} %)")
    ax[1].set(xlabel="Voltage (V)", ylabel="J (mA cm$^{-2}$)", xlim=(0, 1.15), ylim=(0, 25),
              title=f"(b) {lab[e]}: temperature")
    ax[1].legend(frameon=False, fontsize=8, loc="lower left")
    gm = plt.get_cmap("viridis")
    for i, (s, o) in enumerate(res[e]["suns"].items()):
        ax[2].plot(o["V"], np.array(o["J"]) / float(s), color=gm(i / 3),
                   label=f"{s} sun (Voc {o['Voc']:.3f} V)")
    ax[2].set(xlabel="Voltage (V)", ylabel="J / intensity (mA cm$^{-2}$ sun$^{-1}$)", xlim=(0, 1.15), ylim=(0, 25),
              title=f"(c) {lab[e]}: light intensity")
    ax[2].legend(frameon=False, fontsize=8, loc="lower left")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)


if __name__ == "__main__":
    main()
