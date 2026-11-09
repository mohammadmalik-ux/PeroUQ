"""Merge multi-start calibrations of one device.

    python scripts/merge_calibrations.py Yang2014-D

Reads results/calibration_<device>*.json, keeps runs that fitted
(best misfit <= number of targets), and writes
results/calibration_<device>_merged.json with the union of their
acceptable ranges and the best fit over all runs.  Separate runs that land
in different regions reveal parameters that compensate each other.
"""
import glob
import json
import os
import sys


def main(device, folder="results"):
    files = sorted(f for f in glob.glob(os.path.join(folder, f"calibration_{device}*.json"))
                   if not f.endswith("_merged.json"))
    runs = [json.load(open(f)) for f in files]
    ok = [r for r in runs if r["best_cost"] <= len(r["target"])]
    if not ok:
        raise SystemExit("no run fitted the targets")
    names = list(ok[0]["ranges"])
    union = {k: [min(r["ranges"][k][0] for r in ok), max(r["ranges"][k][1] for r in ok)] for k in names}
    best = min(ok, key=lambda r: r["best_cost"])
    out = dict(device=device, runs=[dict(file=os.path.basename(f), seed=r.get("seed"), best_cost=r["best_cost"],
                                         ranges=r["ranges"], best=r["best"]) for f, r in zip(files, runs)],
               n_fitted=len(ok), ranges=union, best=best["best"], best_cost=best["best_cost"],
               target=best["target"], note=best.get("note", ""))
    path = os.path.join(folder, f"calibration_{device}_merged.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    for r in out["runs"]:
        print(r["file"], "misfit", round(r["best_cost"], 3),
              {k: f"{v[0]:.3g}..{v[1]:.3g}" for k, v in r["ranges"].items()})
    print("union:", {k: f"{v[0]:.3g}..{v[1]:.3g}" for k, v in union.items()})
    print("saved", path)


if __name__ == "__main__":
    main(*sys.argv[1:])
