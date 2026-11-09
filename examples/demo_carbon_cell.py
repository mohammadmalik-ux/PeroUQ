"""Demo: FTO / ETL / MAPbI3 / carbon cell without hole-transport layer.

!!  PLACEHOLDER PARAMETERS  !!
The material values below are typical of the SCAPS literature and are
used ONLY to exercise the solver.  They are not the sourced parameter
database of the study (month 1 of the plan) and must not be reported as
results.  The spectrum is a 5778 K black body scaled to 1000 W/m^2; the
study uses ASTM G173 AM1.5G (optics.load_astm_g173).

Run:  python examples/demo_carbon_cell.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perouq import Material, Layer, Contact, Interface, Device, optics, jv   # noqa: E402

pero = Material("MAPbI3", Eg=1.55, chi=3.9, eps_r=6.5, Nc=2.2e18, Nv=1.8e19,
                mu_n=2.0, mu_p=2.0, NA=1e13, tau_n=1e-7, tau_p=1e-7, B_rad=0.0,
                alpha=optics.alpha_sqrt_model(1.55, A=1e5, Eu=0.015))
etls = {
    "TiO2": Material("TiO2", Eg=3.2, chi=4.0, eps_r=9.0, Nc=2.2e18, Nv=1.8e19,
                     mu_n=20.0, mu_p=10.0, ND=1e17, tau_n=1e-9, tau_p=1e-9),
    "CdS": Material("CdS", Eg=2.4, chi=4.2, eps_r=10.0, Nc=2.2e18, Nv=1.8e19,
                    mu_n=100.0, mu_p=25.0, ND=1e17, tau_n=1e-9, tau_p=1e-9,
                    alpha=optics.alpha_sqrt_model(2.4, A=1e5, Eu=0.03)),
}


def make_device(etl, n_pero=200, S_int=1e3):
    return Device(
        layers=[Layer(etl, 50.0, 60), Layer(pero, 500.0, n_pero)],
        left=Contact(W=None, Sn=1e7, Sp=1e7),        # FTO: flat-band
        right=Contact(W=5.0, Sn=1e7, Sp=1e7),        # carbon: W = 5.0 eV
        interfaces=[Interface(k=0, Sn=S_int, Sp=S_int)],
    )


if __name__ == "__main__":
    lam, E = optics.blackbody_test_spectrum()
    out = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out, exist_ok=True)

    sq = optics.sq_limit(lam, E, pero.Eg)
    print(f"SQ limit (Eg={pero.Eg} eV, test spectrum): PCE={100*sq['PCE']:.1f} %  "
          f"Jsc={sq['Jsc']/10:.2f} mA/cm2  Voc={sq['Voc']:.3f} V")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, etl in etls.items():
        r = jv.light_jv(make_device(etl), lam, E, V_max=1.3, dV=0.02)
        m = jv.metrics(r["V"], r["J"], r["P_in"])
        Jabs = 1.602176634e-19 * np.trapezoid(r["G"][r["solver"].lid == 1],
                                             r["solver"].x[r["solver"].lid == 1])
        print(f"{name:5s}: Jsc={m['Jsc']/10:.2f} mA/cm2 (absorbed in perovskite {Jabs/10:.2f})  "
              f"Voc={m['Voc']:.3f} V  FF={m['FF']:.3f}  PCE={100*m['PCE']:.2f} %  "
              f"max continuity error={r['spread'][r['J'] > 1].max():.1e}")
        ax[0].plot(r["V"], r["J"] / 10, label=name)
        if name == "TiO2":
            st = r["solver"].state(r["states"][0])
            x = st["x"] * 1e9
            ax[1].plot(x, st["Ec"], label="Ec")
            ax[1].plot(x, st["Ev"], label="Ev")
            ax[1].plot(x, st["EFn"], "--", label="EFn")
            ax[1].plot(x, st["EFp"], "--", label="EFp")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("Voltage (V)"); ax[0].set_ylabel("J (mA/cm$^2$)")
    ax[0].set_ylim(bottom=-5); ax[0].legend(); ax[0].set_title("Illuminated J-V (placeholder parameters)")
    ax[1].set_xlabel("x (nm)"); ax[1].set_ylabel("Energy (eV)")
    ax[1].legend(); ax[1].set_title("TiO2 device, short circuit, illuminated")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "demo_carbon_cell.png"), dpi=150)
    print("figure saved to figures/demo_carbon_cell.png")
