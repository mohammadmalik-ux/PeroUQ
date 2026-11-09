# PeroUQ

Uncertainty-aware drift–diffusion modelling of perovskite solar cells, written from scratch in Python.
PeroUQ was built for the study *Uncertainty-aware, experiment-calibrated design of hole-transport-free
carbon perovskite solar cells* (M. M. Abood) and reproduces every number in it.

Version 0.8.0 · MIT licence (code) · CC BY 4.0 (parameter database) · Python ≥ 3.10

## What it does

| Module | Content |
| --- | --- |
| `perouq/solver.py` | 1D steady-state drift–diffusion: Poisson + electron/hole continuity, Scharfetter–Gummel fluxes on effective band edges, SRH, radiative and interface recombination, damped Newton |
| `perouq/ions.py` | One mobile ionic species (iodide vacancies) in the absorber: steady-state and frozen-ion (fast-scan) J–V curves, hysteresis index |
| `perouq/optics.py`, `perouq/tmm.py` | Beer–Lambert generation; coherent transfer-matrix optics with an incoherent glass substrate |
| `perouq/jv.py` | J–V curves, Jsc/Voc/FF/PCE, external Rs/Rsh, EQE spectra, Voc at any intensity and temperature |
| `perouq/params/` | Literature database: 203 values from 17 sources, with the author's decisions (`decisions.csv`) |
| `perouq/calib.py`, `perouq/posterior.py` | Calibration against published devices: Bayesian-optimisation search, then an emulator-based Bayesian posterior (ensemble MCMC) |
| `perouq/uq.py` | Latin-hypercube Monte Carlo, Wilson intervals, Saltelli/Jansen Sobol indices with bootstrap intervals |
| `perouq/design.py` | Robust design (maximise mean − λ·std of PCE) by Bayesian optimisation with common random numbers |
| `perouq/study.py` | The FTO/ETL/MAPbI3/carbon cell of the study, with Beer–Lambert or transfer-matrix optics |

## Install

```bash
git clone https://github.com/mohammadmalik-ux/PeroUQ.git
cd PeroUQ
python -m pip install -r requirements.txt
python -m pytest -q tests          # 26 tests, about 3 minutes
```

`data/ASTMG173.csv` is the ASTM G173-03 reference spectrum; every study script needs it (`--spectrum data/ASTMG173.csv`).

## Quick start

```python
import numpy as np
from perouq import optics
from perouq.study import TMMCarbonCell

lam, E = optics.load_astm_g173("data/ASTMG173.csv")
cell = TMMCarbonCell("SnO2", lam, E)          # FTO/SnO2 50 nm/MAPbI3 500 nm/carbon, TMM optics
print(cell(np.full(cell.d, 0.5)))             # every uncertain parameter at the middle of its range
```

## Reproducing the paper

Common options: `S="--spectrum data/ASTMG173.csv --W 4.95 5.35 --min-spread"`. Times are for two CPU cores.

| Result in the paper | Command | Output | Time |
| --- | --- | --- | --- |
| Verification (Table 1) | `python -m pytest -q tests` | — | 2 min |
| Calibration and posterior (Table 2) | `python scripts/run_posterior.py --spectrum data/ASTMG173.csv --device Iqbal2023 --seeds 0` and `--device Yang2014-D --seeds 0 1 2` | `results/posterior/` | 3 + 8 min |
| Independent validation (Table 3) | `python scripts/run_validation.py --spectrum data/ASTMG173.csv --calibrated Yang2014-D --test Nandigana2024` | `results/validation/` | 5 min |
| Efficiency distributions (Table 4, Fig. 1) | `python scripts/run_uq_study.py $S --optics tmm --n-mc 500 --sobol-N 0 --out results/uq_tmm` | `results/uq_tmm/` | 30 min |
| Statistical stability (Table 4) | `python scripts/run_stability.py $S --optics tmm --main results/uq_tmm --seeds 1 2 --out results/stability_tmm` | `results/stability_tmm/` | 60 min |
| Sobol indices (Table 5, Fig. 1) | `python scripts/run_sobol.py $S --optics tmm --N 512 --etl <ETL> --out results/sobol512_tmm` (resumable; `--upto 256` stops at N = 256) | `results/sobol512_tmm/` | 2.5 h per ETL |
| First-order indices from Monte Carlo data | `python scripts/run_first_order.py --main results/uq_tmm --replicates results/stability_tmm --sobol results/sobol512_tmm` | `results/first_order/` | 1 min |
| Optical model check | `python scripts/check_optics.py --spectrum data/ASTMG173.csv` | `results/optics_check/` | 10 min |
| Correlated inputs (Table 6) | `python scripts/run_correlation.py $S --optics tmm --main results/uq_tmm` | `results/correlation/` | 50 min |
| Temperature (Table 6) | `python scripts/run_temperature.py $S --optics tmm --n 100` | `results/temperature/` | 35 min |
| Mobile ions (Table 7) | `python scripts/run_ions.py $S --n 40 --mesh-check` | `results/ions/` | 40 min |
| EQE and J–V at several temperatures and intensities (Fig. 3) | `python scripts/run_spectra.py $S` | `results/spectra/` | 5 min |
| Robust design (search, Fig. 2) | `python scripts/run_design.py $S --optics tmm --out results/design_v3` | `results/design_v3/` | 30 min |
| Robust design (500-sample check, Table 8) | `python scripts/run_design_check.py $S --optics tmm --main results/uq_tmm`, then `python scripts/run_design_eval.py $S --optics tmm --designs results/design_v3` | `results/design_check/` | 90 min |
| Which data identify the unknowns (Table 9) | `python scripts/run_identifiability.py --spectrum data/ASTMG173.csv --truth iqbal` (and `--truth bulk`) | `results/identifiability/` | 30 min each |
| SCAPS-1D cross-check inputs | `python scripts/make_scaps_inputs.py --spectrum data/ASTMG173.csv --out scaps` | `scaps/` | 1 min |
| Database package | `python scripts/export_database.py --out database` | `database/` | seconds |

Every script writes JSON/CSV next to its log; no number in the paper is typed by hand from a figure.

## Data

`database/` is the parameter database as a reusable package (CSV + Excel + data dictionary, CC BY 4.0);
`perouq/params/` holds the copies the code reads. Every value carries its source; `decisions.csv` records
the author's choices (e.g. MAPbI3 permittivity 6.5–10, measured thin-film mobilities only).

## Using your own measurements

* Measured n,k spectra: `tmm.load_nk_csv(path)` and `study.default_stack(..., nk={"MAPbI3": ..., "ETL": ...})`.
* A digitised J–V curve: `calib.Calibrator(..., jv_target=(V, J))`.
* EQE, light-intensity or temperature-dependent data: see `scripts/run_identifiability.py`, which turns them into
  calibration targets (ideality factor, dVoc/dT, EQE at chosen wavelengths).

## Conventions

Input units nm, eV, cm⁻³, cm² V⁻¹ s⁻¹, cm s⁻¹; SI inside. Light enters at x = 0 (FTO/electron contact);
the applied voltage raises the hole contact; photocurrent is positive.

## Citing

See `CITATION.cff`. The version history is in `CHANGELOG.md`.

## Licence

Code: MIT © 2026 Mohammad Malik Abood (`LICENSE`). Database: CC BY 4.0.
