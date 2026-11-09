"""How much does the Beer-Lambert approximation (with eta_opt = 0.95) change the results?

    python scripts/check_optics.py --spectrum data/ASTMG173.csv --n 32 --jobs 2

For each ETL the same Latin-hypercube samples are run twice:
  (a) Beer-Lambert generation x eta_opt (as in the study),
  (b) transfer-matrix generation (glass / FTO / ETL / MAPbI3 / carbon), eta_opt = 1,
with refractive indices typical of the literature and k from the same absorption
model as (a).  A photon budget at the median parameters is also computed for
bracketing optical constants.  Output: results/optics_check/optics_check.json.

The optical constants are typical literature values, not measured spectra of the
devices; the check bounds the size of the optical approximation, it does not
replace measured n,k data.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum      # noqa: E402
from perouq import calib, optics, params, tmm, uq        # noqa: E402
from perouq.constants import q, h_planck, c_light        # noqa: E402
from perouq.params import DEFAULT_MIN_SPREAD             # noqa: E402
from perouq.study import CarbonCell, default_extra       # noqa: E402

from perouq.study import N_ETL, OPTICAL_SCENARIOS as SCEN, TMMCarbonCell as TMMCell   # noqa: E402
from perouq.study import default_stack as make_stack                                 # noqa: E402


def jsc_budget(lam, E, A):
    phi = E * lam * 1e-9 / (h_planck * c_light)
    return float(q * np.trapezoid(phi * A, lam) / 10)       # mA/cm^2


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/optics_check")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    extra = [p for p in default_extra() if p.name != "W"] + [calib.Param("W", 4.95, 5.35)]
    ms = DEFAULT_MIN_SPREAD
    res = dict(warning=tag, indices=N_ETL, scenarios=SCEN, etl={})
    sel = (lam >= 300) & (lam <= 850)
    for etl in a.etl:
        t0 = time.time()
        bl = CarbonCell(etl, lam, E, extra=extra, min_spread=ms, eta_opt=0.95)
        tm = TMMCell(etl, lam, E, extra=extra, min_spread=ms, eta_opt=1.0)
        # photon budget at the median parameters
        Eg_p = params.table("MAPbI3")["Eg"]["median"]
        Eg_e = params.table(etl)["Eg"]["median"]
        dev = bl.device(np.full(bl.d, 0.5))
        budget = {}
        for sc in SCEN:
            o = tmm.stack_optics(dev, make_stack(etl, Eg_p, Eg_e, sc), lam[sel])
            budget[sc] = dict(Jmax_absorber=jsc_budget(lam[sel], E[sel], o["A_elec"][1]),
                              J_reflected=jsc_budget(lam[sel], E[sel], o["R"]),
                              J_FTO=jsc_budget(lam[sel], E[sel], o["A_front"][0]),
                              J_ETL=jsc_budget(lam[sel], E[sel], o["A_elec"][0]),
                              J_back=jsc_budget(lam[sel], E[sel], o["A_back"]))
        # Beer-Lambert absorber budget (single pass through ETL then absorber), eta = 1
        a_e = optics.alpha_sqrt_model(Eg_e, A=1e5, Eu=0.03)(lam[sel]) * 1e2      # 1/m
        a_p = optics.alpha_sqrt_model(Eg_p)(lam[sel]) * 1e2
        A_bl = np.exp(-a_e * bl.etl_nm * 1e-9) * (1 - np.exp(-a_p * bl.pero_nm * 1e-9))
        J_bl = jsc_budget(lam[sel], E[sel], A_bl)
        # common-sample comparison of device metrics
        from scipy.stats import qmc
        U = qmc.LatinHypercube(d=bl.d, seed=1).random(a.n)
        r_bl = uq.evaluate(bl, U, a.jobs)
        r_tm = uq.evaluate(tm, U, a.jobs)
        rows = [(x["PCE"], y["PCE"], x["Jsc"], y["Jsc"], x["Voc"], y["Voc"], x["FF"], y["FF"])
                for x, y in zip(r_bl, r_tm) if x and y]
        R = np.array(rows)
        dP = R[:, 1] - R[:, 0]
        res["etl"][etl] = dict(
            n=len(R), budget_tmm=budget, Jmax_beer_lambert_eta1=J_bl,
            eta_equivalent={sc: budget[sc]["Jmax_absorber"] / J_bl for sc in SCEN},
            PCE_bl_median=float(np.median(R[:, 0])), PCE_tmm_median=float(np.median(R[:, 1])),
            dPCE_mean=float(dP.mean()), dPCE_sd=float(dP.std(ddof=1)),
            dPCE_range=[float(dP.min()), float(dP.max())],
            Jsc_ratio_mean=float(np.mean(R[:, 3] / R[:, 2])),
            dVoc_mean=float(np.mean(R[:, 5] - R[:, 4])), dFF_mean=float(np.mean(R[:, 7] - R[:, 6])),
            P15_bl=float(np.mean(R[:, 0] > 15)), P15_tmm=float(np.mean(R[:, 1] > 15)),
            rank_corr=float(np.corrcoef(np.argsort(np.argsort(R[:, 0])), np.argsort(np.argsort(R[:, 1])))[0, 1]),
            seconds=round(time.time() - t0))
        print(etl, json.dumps(res["etl"][etl], indent=1, default=float), flush=True)
        with open(os.path.join(a.out, "optics_check.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
