"""Demo: Bayesian calibration recovering a synthetic device.

The 'published' device is simulated with known S, carbon work function and
series resistance.  Calibration is run twice: from the four figures of merit
only, and from the figures of merit plus a digitised J-V curve, to show how
much each kind of published data constrains the parameters.

Test spectrum (not AM1.5G) and database medians: a verification, not a result.
Run:  python examples/demo_calibration.py   (about 1.5 minutes)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from perouq import optics, calib                                  # noqa: E402
from test_calib import build, PARAMS, TRUE                        # noqa: E402

lam, E = optics.blackbody_test_spectrum()
sim = calib.device_simulator(build, lam, E)
truth = sim(TRUE)
target = {k: truth[k] for k in ("Jsc", "Voc", "FF", "PCE")}
Vd = np.arange(0.0, truth["Voc"], 0.05)
Jd = np.interp(Vd, truth["V"], truth["J"])

runs = {}
for label, jvt in (("metrics only", None), ("metrics + J-V curve", (Vd, Jd))):
    cal = calib.Calibrator(PARAMS, sim, target, jv_target=jvt, seed=1)
    runs[label] = cal.run(n_iter=30)
    r = runs[label]
    print(f"\n{label}: best misfit {r['best_cost']:.3f}, acceptable region "
          f"{100*r['region_fraction']:.1f} % of the prior box, precision {r['precision']}")
    for k, (lo, hi, span) in r["ranges"].items():
        print(f"  {k:3s}: {lo:.4g} .. {hi:.4g}  (true {TRUE[k]}, {100*span:.0f} % of prior)")

fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
for label, r in runs.items():
    ax[0].plot(np.minimum.accumulate(r["history"]), label=label)
ax[0].set_yscale("log"); ax[0].set_xlabel("evaluation"); ax[0].set_ylabel("best misfit so far")
ax[0].legend(fontsize=8); ax[0].set_title("Convergence")
for j, p in enumerate(PARAMS):
    a = ax[j + 1]
    for i, (label, r) in enumerate(runs.items()):
        reg = r["region"][:: max(1, len(r["region"]) // 400)]
        v = np.array([t[p.name] for t in reg])
        a.plot(v, i + 0.15 * np.random.default_rng(i).standard_normal(len(v)), ".", alpha=0.3,
               label=f"{label}: acceptable region")
    a.axvline(TRUE[p.name], color="k", ls="--", lw=1, label="true value")
    a.set_xlim(p.low if not p.log else p.low, p.high)
    if p.log:
        a.set_xscale("log")
    a.set_yticks([0, 1]); a.set_yticklabels(["metrics", "+ J-V"])
    a.set_title({"S": "Interface S (cm/s)", "W": "Carbon work function (eV)",
                 "Rs": "Series resistance (ohm cm$^2$)"}[p.name])
h, l = ax[3].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.suptitle("Calibration recovers a synthetic device (test spectrum; verification only)", fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 1))
out = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "demo_calibration.png"), dpi=150)
print("figure saved to figures/demo_calibration.png")
