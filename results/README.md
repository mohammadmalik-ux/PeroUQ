# Results

Every number in the paper comes from a file in this folder; the script that writes each
folder is listed in the main README.

| Folder / file | Paper | Content |
|---|---|---|
| `calibration_*.json`, `posterior/` | Table 2 | Bayesian-optimisation search and emulator-based posterior for the published devices |
| `validation/` | Table 3 | Prediction of the independent Nandigana device |
| `uq_tmm/` | Table 4, Fig. 1 | Main Monte Carlo run (500 samples per ETL, transfer-matrix optics) |
| `stability_tmm/` | Table 4 | Bootstrap intervals, running estimates and two replicate runs |
| `sobol512_tmm/` | Table 5, Fig. 1 | Sobol indices (Saltelli/Jansen) with convergence over nested N |
| `first_order/` | Section 3.4 | Given-data first-order indices from 1500 Monte Carlo samples |
| `optics_check/` | Section 3.5 | Transfer matrix vs. Beer–Lambert and optical-constant scenarios |
| `correlation/`, `temperature/` | Table 6 | Correlated inputs (Gaussian copula); 275–350 K |
| `ions/` | Table 7 | Mobile-ion scenarios and hysteresis |
| `design_v2/`, `design_v3/`, `design_check/` | Table 8, Fig. 2 | Robust-design searches (Beer–Lambert, transfer matrix) and the 500-sample check |
| `identifiability/` | Table 9 | Synthetic test of which measurements identify the unknowns |
| `spectra/` | Fig. 3 | Predicted EQE and J–V curves at several temperatures and intensities |
| `uq_pass1/` | Section 3.5 | First modelling pass (before the minimum-uncertainty rule and the second CdS source) |
