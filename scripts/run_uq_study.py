"""Monte Carlo and Sobol study of FTO/ETL/MAPbI3/carbon cells for each ETL.

    python scripts/run_uq_study.py --spectrum ASTMG173.csv --jobs 4
    python scripts/run_uq_study.py --spectrum ASTMG173.csv --calibrated results/calibration_Iqbal2023.json

Outputs in results/:
  mc_<ETL>.csv        sampled parameters and PCE, Voc, Jsc, FF per device
  sobol_<ETL>.csv     first-order and total Sobol indices of PCE with 95 % intervals
  summary.json        distributions and exceedance probabilities per ETL
  uq_figure.png       PCE distributions, P(PCE > threshold), top Sobol indices

Cost: Monte Carlo n runs + Sobol N (d + 2) runs per ETL, ~0.8 s per run per core.
With d ~ 21: N = 256 -> ~5900 runs -> ~40 min per ETL on 2 cores.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum      # noqa: E402
from perouq import uq, calib                             # noqa: E402
from perouq.study import CarbonCell, default_extra, make_cell       # noqa: E402


def extras_from_calibration(path):
    """Interface S and carbon W ranges from a calibration; Rs, Rsh, eta_opt at the best fit."""
    with open(path) as f:
        c = json.load(f)
    r, b = c["ranges"], c["best"]
    extra = [calib.Param("S", r["S"][0], r["S"][1], log=True), calib.Param("W", r["W"][0], r["W"][1])]
    return extra, dict(Rs=b["Rs"], Rsh=b["Rsh"], eta_opt=b["eta_opt"]), c


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--etl", nargs="+", default=["TiO2", "SnO2", "ZnO", "CdS"])
    ap.add_argument("--n-mc", type=int, default=1000)
    ap.add_argument("--sobol-N", type=int, default=256)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[15.0, 20.0])
    ap.add_argument("--calibrated", help="calibration JSON from run_calibration.py")
    ap.add_argument("--W", type=float, nargs=2, metavar=("LOW", "HIGH"),
                    help="carbon work-function range [eV], e.g. the calibrated range")
    ap.add_argument("--eta-opt", type=float, help="optical-loss factor applied to every device")
    ap.add_argument("--min-spread", action="store_true",
                    help="give every parameter at least a minimum uncertainty (x/3 for densities and "
                         "mobilities, +/-0.1 eV for energies, +/-10 %% for permittivity) so that materials "
                         "with few sources are not treated as exactly known")
    ap.add_argument("--optics", choices=["bl", "tmm"], default="bl",
                    help="bl: Beer-Lambert x eta_opt; tmm: transfer-matrix generation (eta_opt = 1)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--resume", action="store_true", help="skip ETLs already in OUT/summary.json")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    os.makedirs(a.out, exist_ok=True)
    extra, circuit, cal = default_extra(), {}, None
    if a.calibrated:
        extra, circuit, cal = extras_from_calibration(a.calibrated)
    if a.W:
        extra = [p for p in extra if p.name != "W"] + [calib.Param("W", a.W[0], a.W[1])]
    if a.eta_opt is not None:
        circuit["eta_opt"] = a.eta_opt

    from perouq.params import DEFAULT_MIN_SPREAD
    ms = DEFAULT_MIN_SPREAD if a.min_spread else None
    summary = dict(warning=tag, optics=a.optics, seed=a.seed, min_spread=ms, calibrated_from=cal["device"] if cal else None,
                   extra={p.name: [p.low, p.high] for p in extra}, circuit=circuit, etl={})
    spath = os.path.join(a.out, "summary.json")
    if a.resume and os.path.exists(spath):
        with open(spath) as f:
            summary["etl"].update(json.load(f).get("etl", {}))
    for etl in a.etl:
        if etl in summary["etl"]:
            print(f"== {etl}: already done, skipped (--resume)")
            continue
        t0 = time.time()
        model = make_cell(a.optics, etl, lam, E, extra=extra, min_spread=ms, **circuit)
        print(f"\n== {etl}: d = {model.d}; Monte Carlo {a.n_mc} runs, Sobol {a.sobol_N * (model.d + 2)} runs")
        U, res = uq.propagate(model, model.d, a.n_mc, seed=a.seed, n_jobs=a.jobs)
        with open(os.path.join(a.out, f"mc_{etl}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(model.names + ["PCE", "Voc", "Jsc", "FF"])
            for u, r in zip(U, res):
                vp, ve, vx = model.theta(u)
                vals = [vp.get(n.split(".", 1)[1]) if n.startswith("MAPbI3.") else
                        ve.get(n.split(".", 1)[1]) if n.startswith(etl + ".") else vx.get(n) for n in model.names]
                w.writerow(vals + ([r[k] for k in ("PCE", "Voc", "Jsc", "FF")] if r else [""] * 4))
        pce = uq._column(res, "PCE")
        s = dict(PCE=uq.summarize(pce), Voc=uq.summarize(uq._column(res, "Voc")),
                 exceed={str(t): uq.exceedance(pce, t) for t in a.thresholds})
        if a.sobol_N <= 0:
            s["sobol_top"] = []
            s["seconds"] = round(time.time() - t0)
            summary["etl"][etl] = s
            with open(spath, "w") as f:
                json.dump(summary, f, indent=2, default=float)
            continue
        sob = uq.sobol(model, model.d, a.sobol_N, ["PCE"], seed=a.seed, n_jobs=a.jobs)["PCE"]
        with open(os.path.join(a.out, f"sobol_{etl}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["parameter", "S1", "S1_low", "S1_high", "ST", "ST_low", "ST_high"])
            for i, n in enumerate(model.names):
                w.writerow([n, sob["S1"][i], *sob["S1_ci"][:, i], sob["ST"][i], *sob["ST_ci"][:, i]])
        order = np.argsort(sob["ST"])[::-1]
        s["sobol_top"] = [(model.names[i], float(sob["ST"][i])) for i in order[:8]]
        s["seconds"] = round(time.time() - t0)
        summary["etl"][etl] = s
        print(json.dumps(s, indent=1, default=float))
        with open(spath, "w") as f:             # saved after every ETL, so an interruption loses at most one
            json.dump(summary, f, indent=2, default=float)
    summary["etl"] = {e: summary["etl"][e] for e in a.etl if e in summary["etl"]}
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    figure(summary, a.out, tag)


def figure(summary, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    etls = list(summary["etl"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    data = []
    for e in etls:
        with open(os.path.join(out, f"mc_{e}.csv")) as f:
            rows = list(csv.DictReader(f))
        data.append([float(r["PCE"]) for r in rows if r["PCE"]])
    ax[0].violinplot(data, showmedians=True)
    ax[0].set_xticks(range(1, len(etls) + 1)); ax[0].set_xticklabels(etls)
    ax[0].set_ylabel("PCE (%)"); ax[0].set_title("PCE distribution (literature ranges)")
    th = sorted({k for e in etls for k in summary["etl"][e]["exceed"]}, key=float)
    w = 0.8 / len(th)
    for j, t in enumerate(th):
        p = [summary["etl"][e]["exceed"][t]["p"] for e in etls]
        lo = [p[i] - summary["etl"][e]["exceed"][t]["low"] for i, e in enumerate(etls)]
        hi = [summary["etl"][e]["exceed"][t]["high"] - p[i] for i, e in enumerate(etls)]
        ax[1].bar(np.arange(len(etls)) + j * w, p, w, yerr=[lo, hi], capsize=3, label=f"PCE > {t} %")
    ax[1].set_xticks(np.arange(len(etls)) + w * (len(th) - 1) / 2); ax[1].set_xticklabels(etls)
    ax[1].set_ylabel("probability"); ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].set_title("Probability of exceeding")
    for k, e in enumerate(etls):
        top = summary["etl"][e]["sobol_top"][:5]
        if len(top) < 5:
            continue
        ax[2].barh(np.arange(5) + k * 6, [v for _, v in top][::-1], color=f"C{k}")
        for i, (n, v) in enumerate(top[::-1]):
            ax[2].text(0.01, i + k * 6, n, va="center", fontsize=7)
    ax[2].set_yticks([2 + 6 * k for k in range(len(etls))]); ax[2].set_yticklabels(etls)
    ax[2].set_xlabel("total Sobol index of PCE"); ax[2].set_title("Top 5 parameters per ETL")
    if tag:
        fig.suptitle(tag, color="red")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "uq_figure.png"), dpi=150)
    print("figure:", os.path.join(out, "uq_figure.png"))


if __name__ == "__main__":
    main()
