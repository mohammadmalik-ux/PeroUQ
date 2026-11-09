"""Demo: transfer-matrix optics for glass / FTO / TiO2 / MAPbI3 / carbon.

!!  PLACEHOLDER OPTICAL CONSTANTS  !!
Constant real indices plus k from the sqrt absorption model are used ONLY to
exercise the code.  The study replaces them with measured n,k spectra
(e.g. ellipsometry data loaded with tmm.load_nk_csv), each with its source.
The spectrum is the 5778 K black-body test spectrum, not AM1.5G.

Run:  python examples/demo_tmm.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perouq import Layer, Device, DDSolver, optics, jv, tmm          # noqa: E402
from perouq.constants import q, h_planck, c_light                  # noqa: E402
from demo_carbon_cell import pero, etls, make_device               # noqa: E402

TiO2 = etls["TiO2"]
nk_glass = tmm.nk_constant(1.52)
nk_fto = tmm.nk_constant(1.9, 0.005)
nk_tio2 = tmm.nk_from_alpha(2.4, optics.alpha_sqrt_model(3.2, A=1e5, Eu=0.03))
nk_pero = tmm.nk_from_alpha(2.5, pero.alpha)
nk_carbon = tmm.nk_constant(2.0, 0.8)


def make_stack():
    return tmm.OpticalStack(nk=[nk_tio2, nk_pero],
                            front=[tmm.OpticalLayer("FTO", nk_fto, 400.0)],
                            back=nk_carbon, substrate=nk_glass)


def photocurrent(lam, E, A):
    """q * integral(photon flux x absorptance) [mA/cm^2]."""
    phi = E * lam * 1e-9 / (h_planck * c_light)
    return q * np.trapezoid(phi * A, lam) / 10


if __name__ == "__main__":
    lam_all, E_all = optics.blackbody_test_spectrum()
    sel = (lam_all >= 300) & (lam_all <= 900)
    lam, E = lam_all[sel], E_all[sel]
    P_in = np.trapezoid(E_all, lam_all)                 # 1000 W/m^2 over the full spectrum
    out = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out, exist_ok=True)

    dev = make_device(TiO2)
    st = make_stack()
    s = DDSolver(dev)
    G_tmm, opt = tmm.tmm_generation(s, st, lam, E, return_optics=True)
    G_bl = optics.beer_lambert_generation(s, lam, E)

    print("Photon budget (mA/cm2 equivalent, 300-900 nm):")
    print(f"  reflected          {photocurrent(lam, E, opt['R']):6.2f}")
    print(f"  FTO (parasitic)    {photocurrent(lam, E, opt['A_front'][0]):6.2f}")
    print(f"  TiO2               {photocurrent(lam, E, opt['A_elec'][0]):6.2f}")
    print(f"  MAPbI3             {photocurrent(lam, E, opt['A_elec'][1]):6.2f}")
    print(f"  carbon (parasitic) {photocurrent(lam, E, opt['A_back']):6.2f}")

    res = {}
    for name, stack in (("Beer-Lambert", None), ("TMM", st)):
        r = jv.light_jv(dev, lam, E, V_max=1.3, dV=0.01, stack=stack)
        m = jv.metrics(r["V"], r["J"], P_in)
        res[name] = (r, m)
        print(f"{name:12s}: Jsc={m['Jsc']/10:.2f} mA/cm2  Voc={m['Voc']:.3f} V  "
              f"FF={m['FF']:.3f}  PCE={100*m['PCE']:.2f} %")

    # optical thickness sweep of the absorber (interference shows up here)
    thick = np.arange(200, 1001, 25)
    J_pero = []
    for t in thick:
        d2 = Device([Layer(TiO2, 50.0, 10), Layer(pero, float(t), 10)])
        o = tmm.stack_optics(d2, st, lam)
        J_pero.append(photocurrent(lam, E, o["A_elec"][1]))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a = ax[0, 0]
    parts = [opt["A_front"][0], opt["A_elec"][0], opt["A_elec"][1], opt["A_back"], opt["R"]]
    labels = ["FTO", "TiO2", "MAPbI3", "carbon", "reflected"]
    a.stackplot(lam, *parts, labels=labels, alpha=0.85)
    a.set_xlim(300, 900); a.set_ylim(0, 1)
    a.set_xlabel("Wavelength (nm)"); a.set_ylabel("Fraction of incident light")
    a.set_title("Where the light goes (TMM)"); a.legend(loc="lower left", fontsize=8)

    a = ax[0, 1]
    m = s.lid == 1
    x = (s.x[m] - s.x[m][0]) * 1e9
    a.plot(x, G_tmm[m] / 1e6, label="TMM")
    a.plot(x, G_bl[m] / 1e6, "--", label="Beer-Lambert")
    a.set_xlabel("Depth in MAPbI3 (nm)"); a.set_ylabel("G (cm$^{-3}$ s$^{-1}$)")
    a.set_yscale("log"); a.legend(); a.set_title("Generation profile")

    a = ax[1, 0]
    for name, (r, mm) in res.items():
        a.plot(r["V"], r["J"] / 10, label=f"{name}: PCE {100*mm['PCE']:.1f} %")
    a.axhline(0, color="k", lw=0.6); a.set_ylim(bottom=-5)
    a.set_xlabel("Voltage (V)"); a.set_ylabel("J (mA/cm$^2$)"); a.legend()
    a.set_title("J-V with each optical model")

    a = ax[1, 1]
    a.plot(thick, J_pero, "o-", ms=3)
    a.set_xlabel("MAPbI3 thickness (nm)"); a.set_ylabel("Absorbed photocurrent (mA/cm$^2$)")
    a.set_title("Absorber thickness sweep (TMM)")

    fig.suptitle("Placeholder optical constants and test spectrum - not results", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "demo_tmm.png"), dpi=150)
    print("figure saved to figures/demo_tmm.png")
