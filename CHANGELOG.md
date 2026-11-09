# PeroUQ change log

Version history up to v0.7.3, moved from the README in v0.8.0.


## v v0.7

- `perouq/design.py` + `scripts/run_design.py`: robust design. Bayesian optimisation of ETL and absorber thickness maximising mean(PCE) − λ·std(PCE) over the material uncertainty, with common random numbers. Verified on an analytic problem: optimum found at (0.299, 0.706) against the true (0.3, 0.7) (`tests/test_design.py`).
- Minimum uncertainty option (`--min-spread`): a parameter reported by one source, or identically by all, gets at least ×/÷3 (densities, mobilities), ±0.1 eV (energies) or ±10 % (permittivity). Every ETL then has the same 22 uncertain parameters, so no layer looks better just because it has fewer sources. This is a methodological choice proposed in the code, not the author's decision; omit the flag to switch it off.
- The ETL now absorbs above its own gap (`CarbonCell(etl_absorbs=True)`, default). Earlier passes treated it as transparent, which favoured CdS (gap 2.4 eV): at 50 nm CdS absorbs about 0.9 mA/cm² that never reaches the perovskite. For the oxides the effect is below 0.1 mA/cm².
- Multi-start calibration: `run_calibration.py --seed N --tag _seedN` and `scripts/merge_calibrations.py <device>`.
- Database: second CdS source (Islam 2023), measured porous-TiO₂ mobility (Aduda 2004, recorded but not sampled pending the author's decision), 203 values from 17 sources.

### Study results (first passes, AM1.5G; `results/`)

Settings: ETL 50 nm, MAPbI₃ 500 nm, carbon W 4.95–5.35 eV, η_opt = 0.95, no Rs/Rsh, 300 Monte Carlo samples and Sobol N = 64 per ETL (about 1840 simulations each, none failed). Pass 3 is the reference: minimum uncertainty on, ETL absorption on.

| ETL | Median PCE (%) | 5–95 % | P(PCE > 15 %) pass 1 / 2 / 3 | P(PCE > 20 %) pass 3 | Top total Sobol index (pass 3) |
| --- | --- | --- | --- | --- | --- |
| TiO₂ | 16.1 | 10.8–20.5 | 0.66 / 0.64 / 0.62 | 0.067 | carbon W 0.40 |
| CdS | 16.1 | 10.6–20.2 | 0.54 / 0.67 / 0.61 | 0.057 | carbon W 0.39 |
| SnO₂ | 14.4 | 8.0–19.7 | 0.46 / 0.44 / 0.43 | 0.033 | SnO₂ χ 0.34 |
| ZnO | 11.9 | 5.9–18.1 | 0.22 / 0.21 / 0.21 | 0.020 | ZnO χ 0.32 |

Pass 1: one CdS source, no minimum uncertainty, transparent ETL. Pass 2: + second CdS source and minimum uncertainty. Pass 3: + ETL absorption. SnO₂ and ZnO are stable across passes; the CdS position depends on its sources and its parasitic absorption.

Robust design (`results/design_v2/`, λ = 1, 24 common samples, 14 designs per ETL): absorber 600–690 nm; ETL at the 20 nm lower bound for TiO₂, SnO₂ and CdS (less parasitic absorption; the model has no pinholes, so 20 nm is a practical floor); ZnO insensitive to ETL thickness. The mean rises by 0.2–0.6 points and the spread falls by 0.1–0.5 points relative to 50/500 nm.

### Multi-start calibration of Yang 2014-D (three seeds)

| Run | Misfit | S (cm/s) | Carbon W (eV) | Rs (Ω cm²) | Absorber (nm) |
| --- | --- | --- | --- | --- | --- |
| seed 0 | 0.054 | 0.11 – 1.5e4 | 4.95 – 5.00 | 6.4 – 11.6 | 315 – 424 |
| seed 1 | 0.173 | 1.4e3 – 1e5 | 4.96 – 5.15 | 6.2 – 10.5 | 263 – 313 |
| seed 2 | 0.054 | 0.42 – 1.6e3 | 4.95 – 5.02 | 2.0 – 8.4 | 273 – 354 |
| union | | 0.11 – 1e5 | 4.95 – 5.15 | 2.0 – 11.6 | 263 – 424 |

The three starts land in different parts of one long valley: S spans seven decades while W stays within 0.2 eV of the lower bound and the thickness within 263–424 nm. Published Jsc/Voc/FF alone cannot fix S for this device.

## v v0.6

- `perouq/uq.py`: Latin-hypercube Monte Carlo, exceedance probabilities with Wilson 95 % intervals, and Sobol indices (Saltelli 2010 first order, Jansen total) with bootstrap intervals; failed runs drop out cleanly; parallel with `n_jobs`.
  - Verified on the Ishigami function: S1 = 0.3140 / 0.4437 / −0.0002 and ST = 0.5574 / 0.4424 / 0.2437 against analytic 0.3139 / 0.4424 / 0 and 0.5576 / 0.4424 / 0.2437.
- `perouq/study.py`: `CarbonCell(etl, ...)` maps a point of the unit cube to an FTO/ETL/MAPbI₃/carbon device whose uncertain parameters come from the database (21 dimensions for SnO₂).
- `scripts/run_sobol.py`: Sobol indices of PCE alone for a larger base sample (N = 512 in the paper, 12 288 runs per ETL); one ETL per call, resumable.
- `scripts/check_optics.py`: compares transfer-matrix generation with Beer–Lambert × η_opt on common samples and gives the photon budget for bracketing optical constants.
- `scripts/run_calibration.py` and `scripts/run_uq_study.py`: the study itself; both require the AM1.5G file (`--spectrum data/ASTMG173.csv`). `--test-spectrum` exists only for plumbing checks and stamps every output "TEST SPECTRUM - NOT RESULTS".
- `data/ASTMG173.csv`: ASTM G173-03 reference spectra (integrates to 1000.4 W/m²). With it the SQ limit at 1.34 eV is 33.7 % and Jsc 35.0 mA/cm², the published values (new test).
- `gen_scale` / `eta_opt`: optical-loss factor applied to Beer–Lambert generation, calibrated, until measured n,k data allow the TMM model.

### First calibrations on published devices (AM1.5G)

Targets: Jsc, Voc, FF (PCE left out because it equals Jsc·Voc·FF). For Iqbal 2023 the EQE-integrated Jsc (20.97 mA/cm²) is used instead of the J–V value (22.77). In a first run with that J–V value and an optical factor capped at 1, the model could not reach it.

| | Iqbal 2023 (ITO/SnO₂/MAPbI₃/C, planar) | Yang 2014-D (FTO/TiO₂/mp-TiO₂/MAPbI₃/C) |
| --- | --- | --- |
| Best misfit (accept ≤ 3) | 0.49 | 0.05 |
| Best Jsc / Voc / FF | 21.12 / 1.007 / 0.604 | 18.69 / 0.955 / 0.572 |
| S (cm/s) | 2.1e3 – 1e5 | 0.11 – 1.5e4 |
| Carbon W (eV) | 5.07 – 5.31 | 4.95 – 5.00 (a first run found 5.13 – 5.35 with S > 3e4) |
| Rs (Ω cm²) | 6.6 – 12.9 | 6.4 – 11.6 |
| η_opt | 0.91 – 0.99 | 0.82 – 0.89 |
| MAPbI₃ Nt | not constrained | not constrained |

S and the carbon work function compensate each other. Two independent calibrations of Yang 2014-D found two separate acceptable regions, so a single run maps only part of the acceptable set. Next improvement: multi-start calibration and a longer mapping phase.

## v v0.5: Bayesian calibration (`perouq/calib.py`)

- Targets: published Jsc, Voc, FF, PCE (each with a tolerance, default 2 %, 1 %, 2 %, 2 %) and, when available, a digitised J–V curve.
- Unknowns: any device parameter (interface recombination velocity, carbon work function, …) plus external Rs and Rsh in Ω cm², scaled to [0, 1] (log10 for parameters spanning decades).
- Phase 1 (search): Gaussian process (Matérn 5/2, one length scale per parameter, noise term) on log(1 + misfit), expected improvement.
- Phase 2 (mapping): the straddle acquisition spends simulations where the surrogate is unsure whether a point is acceptable. Parameters compensate each other, so the acceptable set is a region, not a point; phase 1 alone found only one corner of it.
- Output: best fit, the acceptable region on a dense Sobol set, its ranges, and its precision checked with true simulations.

Verification (`tests/test_calib.py`, `examples/demo_calibration.py`): a synthetic TiO₂/MAPbI₃/carbon device with S = 300 cm/s, W = 5.15 eV and Rs = 3 Ω cm².

| | Metrics only | Metrics + J–V curve |
| --- | --- | --- |
| Best misfit | 0.06 | 0.48 |
| Acceptable region | 8.7 % of the prior box | 6.2 % |
| Region checked by true simulations | 10/10 acceptable | 10/10 |
| S range (true 300) | 0.01–2300 cm/s | 0.01–1600 cm/s |
| W range (true 5.15) | 5.04–5.28 eV | 5.05–5.27 eV |
| Rs range (true 3.0) | 0–6.2 Ω cm² | 0.7–5.4 Ω cm² |

The true values lie inside every region, for two random seeds. The ranges themselves are the finding: published metrics bound the interface recombination velocity only from above (below about 2000 cm/s) and pin the carbon work function to about ±0.12 eV. Each calibration takes about 80 s (78 simulations).

**Real calibration needs the AM1.5G spectrum** (ASTM G173-03 from NREL, loaded with `optics.load_astm_g173`). The black-body test spectrum gives about 6 % less Jsc, which would be absorbed into wrong parameter values.

## v v0.4

- `params/decisions.csv`: the author's decisions, applied when the data are read (`load_rows()`); the raw literature rows stay untouched (`load_rows(decisions=False)`).
  - MAPbI₃ ε_r restricted to 6.5–10 (static value 30 excluded).
  - MAPbI₃ mobilities taken only from measured thin-film values (Herz 2017): electrons 0.7–35.5, holes 0.4–35.5 cm²/Vs. The upper end splits the THz/microwave electron+hole sum equally between carriers (stated assumption).
- New entries: measured MAPbI₃ electron affinity 3.7 eV (Schulz, Cahen & Kahn); FTO work function 4.4–5.0 eV (Helander et al. 2011, XPS 5.0 ± 0.1 eV); ETL/perovskite interface recombination velocity 0.01–1e5 cm/s (Krückemeier et al.); second source for CdS defect density.
- `calibration_devices.csv`: 9 published devices (was 3), with architecture and intended use.
- Modelling note: the FTO/ETL contact is treated as flat-band (ohmic) by default because the model has no tunnelling through the degenerate FTO/ETL junction; the FTO work function is therefore not sampled in the base study.
- 192 values from 15 sources; 15 tests pass.

Steady-state 1D drift–diffusion solver for perovskite solar cells, written from
scratch in Python (numpy + scipy). It is the physical core of the planned study
on uncertainty-aware design of hole-transport-free carbon perovskite cells.

## v v0.3: literature parameter database (`perouq/params/`)

| File | Content |
| --- | --- |
| `parameters.csv` | 167 values (MAPbI₃, TiO₂, SnO₂, ZnO, CdS, contacts), each with unit, source and note |
| `sources.csv` | 11 sources with DOI or URL |
| `nk_sources.csv` | Candidate n,k sources for TMM (still to be completed) |
| `calibration_devices.csv` | Published devices for calibration (3 so far; target 8–12) |
| `__init__.py` | `table()`, `material()`, `ranges()`, `sample_values()`, `summary_markdown()` |

- `params.material("SnO2")` builds a `Material` from the medians; lifetimes follow from τ = 1/(σ v_th N_t).
- `params.sample_values(name, u)` maps u ∈ [0,1]^k to values inside the literature range (log-uniform for densities, mobilities and defect densities). This is the input to the Monte Carlo and Sobol modules.
- `tests/test_params.py` checks that every row has a known source, a consistent unit and a physical value, and that sampled materials stay inside their ranges.
- Convergence check: 60 devices with randomly sampled parameters (15 per ETL) all converged, about 0.8 s per J–V curve.
- **Before publication** every value must be checked against the PDF of its source: values were read from tables by automated extraction, and some exponents in Yusuf2024 were garbled in the extracted text.

## v v0.2: transfer-matrix optics (`perouq/tmm.py`)

- Coherent multilayer at normal incidence (Pettersson, Roman & Inganäs, J. Appl. Phys. 86, 487, 1999) on a thick incoherent glass substrate.
- Optical-only front layers (FTO) and a semi-infinite absorbing back medium (carbon).
- Absorption profile per layer, spectral R and parasitic losses, and the generation rate G(x) on the solver mesh.
- n,k models: constant, Cauchy, k from α, and a loader for refractiveindex.info CSV files.
- `jv.light_jv(..., stack=OpticalStack(...))` switches the J–V calculation from Beer–Lambert to TMM.

| TMM test (`tests/test_tmm.py`) | Result |
| --- | --- |
| Single interface vs. Fresnel | exact (1e-12) |
| Quarter-wave antireflection layer | R = 1e-32 at the design wavelength |
| Energy conservation R + T + A = 1 with absorbing layers | error 2e-9 |
| Weak index-matched absorber vs. Beer–Lambert | within 0.1 % |
| Incoherent glass slab vs. 2R/(1+R) | exact (1e-12) |

`examples/demo_tmm.py` shows the photon budget, the TMM vs. Beer–Lambert
generation profile and J–V, and an absorber thickness sweep. Its optical constants are **placeholders**.

## Solver core (v0.1)

| File | Content |
| --- | --- |
| `perouq/device.py` | Materials, layers, contacts (work function, surface recombination velocities), interfaces, graded mesh with a double node at each heterointerface |
| `perouq/solver.py` | Poisson + electron/hole continuity; Scharfetter–Gummel fluxes on effective band edges (handles band offsets and DOS steps); SRH + radiative bulk recombination; interface recombination; damped Newton with a column-coloured finite-difference Jacobian |
| `perouq/optics.py` | Beer–Lambert generation, √(E−Eg) absorption model with Urbach tail, ASTM G173 loader, Shockley–Queisser limit |
| `perouq/jv.py` | Illumination ramp, J–V sweep, Jsc/Voc/FF/PCE, external Rs and Rsh |
| `tests/test_verification.py` | Verification against analytic results (below) |
| `examples/demo_carbon_cell.py` | FTO/ETL/MAPbI₃/carbon demo with **placeholder** parameters |

## Verification (all pass: `python -m pytest -q tests`)

| Test | Result |
| --- | --- |
| Zero current at equilibrium | \|J\| < 1e-9 A/m² |
| Si p–n diode vs. short-base Shockley equation (0.3–0.5 V) | agreement within 0.13 % |
| Current continuity along x | error ≤ 1e-4 A/m² (1e-8 A/cm²) |
| Mesh convergence (100 → 400 nodes) | Jsc, Voc, PCE change < 0.03 % |
| Full collection in an ideal heterostructure | Jsc / (q ∫G dx) = 1.0000 |
| Shockley–Queisser limit | within the known 30–34 % range for the test spectrum |

One J–V curve takes about 1 s, so a Monte Carlo study of a few thousand
devices per ETL is practical without a surrogate model.

## Conventions

- Input units: nm, eV, cm⁻³, cm²/Vs, cm/s, s. Converted to SI internally.
- Light enters at x = 0 (the FTO / electron contact). The applied voltage raises the right (hole) contact.
- Current is positive in +x, so photocurrent is positive and P = J·V is generated power.
- Each heterointerface is two nodes 0.1 nm apart; interface recombination uses midgap traps.

## Important limitations of v0.1

- **The demo parameters are placeholders**, not the sourced database. Do not report them.
- **The test spectrum is a 5778 K black body.** For the study, download ASTM G173-03 from NREL and load it with `optics.load_astm_g173(path)`.
- TMM needs measured n,k spectra for each layer (ellipsometry data with sources); the demo values are placeholders.
- No ion migration, no trapped charge in Poisson, no Auger recombination yet.
- Licence: MIT (see `LICENSE`). Citation metadata: `CITATION.cff`; Zenodo metadata: `.zenodo.json`.

## Next modules (plan order)

1. ~~`tmm.py`: transfer-matrix optics~~ (done in v0.2).
2. `params/`: sourced parameter database (YAML/CSV, every value with a DOI and a range).
3. `calib.py`: Bayesian calibration against published J–V curves.
4. `uq.py`: Latin hypercube Monte Carlo and Sobol indices.
5. `design.py`: robust design by Bayesian optimisation.
6. Cross-check against SCAPS-1D or Solcore on identical inputs.

## Release checklist (GitHub + Zenodo)

1. Create the repository `PeroUQ` on the same GitHub account used for AgNP-CompLab and push this folder
   (it already contains `LICENSE`, `CITATION.cff` and `.zenodo.json`).
2. In Zenodo, sign in with GitHub, open *GitHub* settings and switch the `PeroUQ` repository on.
3. On GitHub create a release tagged `v0.7.3`; Zenodo archives it and issues the DOI.
4. Put the account name in `CITATION.cff` and the account and DOI in the paper's Code availability section.

## Licence

MIT © 2026 Mohammad Malik Abood. See `LICENSE`.
