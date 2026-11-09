"""Calibrate unknown parameters against published devices.

    python scripts/run_calibration.py --spectrum ASTMG173.csv --device Iqbal2023
    python scripts/run_calibration.py --spectrum ASTMG173.csv --device Yang2014-D

Writes results/calibration_<device>.json with the best fit and the
acceptable ranges, used by run_uq_study.py --calibrated.

Optics: Beer-Lambert with an optical-loss factor eta_opt (calibrated) until
measured n,k data allow the transfer-matrix model.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import add_spectrum_args, load_spectrum        # noqa: E402
from perouq import Layer, Contact, Interface, Device, optics, params, calib   # noqa: E402

DEVICES = {
    # ITO/SnO2/MAPbI3/carbon, planar, HTL-free (Coatings 13 (2023) 1338)
    "Iqbal2023": dict(etl="SnO2", etl_nm=30.0, pero_nm=315.0,
                      target=dict(Jsc=20.97, Voc=1.01, FF=0.61),
                      note=("Jsc = 20.97 mA/cm2 integrated from EQE for the 0.5 MPa device (J-V gives 22.77); "
                            "PCE left out (= Jsc Voc FF); SnO2 thickness not reported, 30 nm assumed")),
    # FTO/c-TiO2/mp-TiO2/MAPbI3/graphite+carbon black (RSC Adv. 2014, c4ra09519g), device D
    "Yang2014-D": dict(etl="TiO2", etl_nm=50.0, pero_nm=None,
                       target=dict(Jsc=18.73, Voc=0.9527, FF=0.572),
                       note=("mesoporous scaffold approximated as a planar stack; absorber thickness calibrated; "
                             "PCE left out (= Jsc Voc FF)")),
    # FTO/c-TiO2/mp-TiO2/MAPbI3(MACl)/carbon (Energy Adv. 3 (2024) 442), champion reverse scan
    "Nandigana2024": dict(etl="TiO2", etl_nm=50.0, pero_nm=None,
                          target=dict(Jsc=11.03, Voc=0.98, FF=0.6586),
                          note=("mesoporous scaffold approximated as a planar stack; absorber thickness calibrated; "
                                "reverse-scan values; PCE left out (= Jsc Voc FF)")),
}


def make_params(cfg):
    nt = params.table("MAPbI3")["Nt"]
    s = params.table("ETL/MAPbI3")["S"]
    p = [calib.Param("S", s["min"], s["max"], log=True),
         calib.Param("W", 4.95, 5.40),
         calib.Param("Nt", nt["min"], nt["max"], log=True),
         calib.Param("Rs", 0.0, 20.0),
         calib.Param("Rsh", 50.0, 1e5, log=True),
         calib.Param("eta_opt", 0.6, 1.0)]
    if cfg["pero_nm"] is None:
        p.append(calib.Param("pero_nm", 200.0, 1000.0))
    return p


def make_builder(cfg):
    def build(th):
        Eg = params.table("MAPbI3")["Eg"]["median"]
        pero = params.material("MAPbI3", {"Nt": th["Nt"]}, alpha=optics.alpha_sqrt_model(Eg))
        etl = params.material(cfg["etl"])
        t = th.get("pero_nm", cfg["pero_nm"])
        return Device([Layer(etl, cfg["etl_nm"], 50), Layer(pero, t, 120)],
                      left=Contact(None), right=Contact(th["W"]),
                      interfaces=[Interface(0, th["S"], th["S"])])
    return build


def main():
    ap = argparse.ArgumentParser()
    add_spectrum_args(ap)
    ap.add_argument("--device", choices=sorted(DEVICES), required=True)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--n-map", type=int, default=40)
    ap.add_argument("--n-check", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. _seed1 for multi-start runs)")
    a = ap.parse_args()
    lam, E, tag = load_spectrum(a)
    cfg = DEVICES[a.device]
    sim = calib.device_simulator(make_builder(cfg), lam, E, P_in=1000.0)
    cal = calib.Calibrator(make_params(cfg), sim, cfg["target"], seed=a.seed)
    t0 = time.time()
    res = cal.run(n_iter=a.n_iter, n_map=a.n_map, n_check=a.n_check, verbose=True)
    out = dict(device=a.device, seed=a.seed, warning=tag, note=cfg["note"], target=cfg["target"],
               best=res["best"], best_cost=res["best_cost"],
               best_metrics={k: res["best_metrics"][k] for k in cfg["target"]},
               ranges={k: [v[0], v[1]] for k, v in res["ranges"].items()},
               region_fraction=res["region_fraction"], precision=res["precision"],
               n_eval=res["n_eval"], failures=res["failures"], seconds=round(time.time() - t0))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"calibration_{a.device}{a.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
