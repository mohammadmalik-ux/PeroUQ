"""Figure 1 of the paper: PCE distributions, exceedance probabilities and leading Sobol indices.

    python scripts/make_fig1.py --main results/uq_tmm --replicates results/stability_tmm \
        --sobol results/sobol512_tmm --suffix _N256 --out results/uq_tmm/fig1.png
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ETLS = ["TiO2", "CdS", "SnO2", "ZnO"]
COL = {"TiO2": "#1f5fa8", "CdS": "#a8322d", "SnO2": "#d0731c", "ZnO": "#3b8f3b"}
LAB = {"TiO2": "TiO$_2$", "SnO2": "SnO$_2$", "ZnO": "ZnO", "CdS": "CdS"}
PNAME = {"Eg": r"$E_g$", "chi": r"$\chi$", "eps_r": r"$\varepsilon_r$", "Nc": r"$N_c$", "Nv": r"$N_v$",
         "mu_n": r"$\mu_n$", "mu_p": r"$\mu_p$", "ND": r"$N_D$", "NA": r"$N_A$", "Nt": r"$N_t$", "sigma": r"$\sigma$"}


def pretty(p, etl):
    if p == "S":
        return "S (interface)"
    if p == "W":
        return "W (carbon)"
    mat, q = p.split(".")
    return f"{PNAME.get(q, q)} ({'MAPbI$_3$' if mat == 'MAPbI3' else LAB[etl]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/uq_tmm")
    ap.add_argument("--replicates", default="results/stability_tmm")
    ap.add_argument("--sobol", default="results/sobol512_tmm")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="results/uq_tmm/fig1.png")
    a = ap.parse_args()
    summ = json.load(open(os.path.join(a.main, "summary.json")))["etl"]
    stab = json.load(open(os.path.join(a.replicates, "stability.json")))["etl"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    data = []
    for e in ETLS:
        with open(os.path.join(a.main, f"mc_{e}.csv")) as f:
            data.append([float(r["PCE"]) for r in csv.DictReader(f) if r["PCE"]])
    vp = ax[0].violinplot(data, showmedians=True, showextrema=False)
    for b, e in zip(vp["bodies"], ETLS):
        b.set_facecolor(COL[e]); b.set_alpha(0.55)
    vp["cmedians"].set_color("k")
    for i, d in enumerate(data):
        ax[0].text(i + 1, np.median(d) + 0.4, f"{np.median(d):.1f}", ha="center", fontsize=8)
    ax[0].axhline(15, color="0.5", lw=0.8, ls="--"); ax[0].axhline(20, color="0.5", lw=0.8, ls=":")
    ax[0].set_xticks(range(1, 5)); ax[0].set_xticklabels([LAB[e] for e in ETLS])
    ax[0].set_ylabel("PCE (%)"); ax[0].set_title("(a) PCE distribution (500 samples)", fontsize=10)
    w = 0.38
    for j, (t, hatch) in enumerate((("15.0", ""), ("20.0", "//"))):
        x = np.arange(4) + (j - 0.5) * w
        p = np.array([summ[e]["exceed"][t]["p"] for e in ETLS])
        lo = p - np.array([summ[e]["exceed"][t]["low"] for e in ETLS])
        hi = np.array([summ[e]["exceed"][t]["high"] for e in ETLS]) - p
        ax[1].bar(x, p, w, yerr=[lo, hi], capsize=3, color=[COL[e] for e in ETLS], alpha=0.8 if j == 0 else 0.35,
                  hatch=hatch, edgecolor="k", lw=0.5, label=f"PCE > {float(t):.0f} %")
        key = "P15" if t == "15.0" else "P20"
        for i, e in enumerate(ETLS):
            reps = [r[key] for r in stab[e]["replicates"]]
            ax[1].plot([x[i]] * len(reps), reps, "k.", ms=4)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([LAB[e] for e in ETLS])
    ax[1].set_ylim(0, 0.7); ax[1].set_ylabel("Probability")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("(b) Probability of exceeding (dots: replicate runs)", fontsize=10)
    y, ticks, labels = 0, [], []
    for e in ETLS[::-1]:                        # first ETL at the top
        path = os.path.join(a.sobol, f"sobol_{e}{a.suffix}.csv")
        if not os.path.exists(path):
            continue
        rows = sorted(csv.DictReader(open(path)), key=lambda r: -float(r["ST"]))[:5]
        for r in rows[::-1]:
            st = float(r["ST"])
            ax[2].barh(y, st, 0.8, xerr=[[st - float(r["ST_low"])], [float(r["ST_high"]) - st]],
                       color=COL[e], alpha=0.8, capsize=2, error_kw=dict(lw=0.8))
            ticks.append(y); labels.append(pretty(r["parameter"], e))
            y += 1
        ax[2].text(0.47, y - 3, LAB[e], fontsize=10, color=COL[e], weight="bold", ha="right")
        y += 0.8
    ax[2].set_yticks(ticks); ax[2].set_yticklabels(labels, fontsize=7.5)
    ax[2].set_xlim(0, 0.48); ax[2].set_xlabel("Total Sobol index of PCE")
    ax[2].set_title("(c) Five largest total indices per ETL", fontsize=10)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)
    print("figure:", a.out)


if __name__ == "__main__":
    main()
