# SCAPS-1D input for the PeroUQ cross-check

Structure (illuminated from the left): FTO (front contact, flat band, S = 1e7 cm/s) / ETL 50 nm / MAPbI3 500 nm / carbon (back contact, work function 5.15 eV, S = 1e7 cm/s).

Settings: AM1.5G 1000 W/m2, 300 K, no reflection, no series or shunt resistance; radiative recombination off; absorption model sqrt(hv - Eg) with A = 1e5 cm^-1 eV^-1/2 for every layer; one neutral midgap bulk defect per layer with sigma_n = sigma_p = 2e-14 cm2 and vth = 1e+07 cm/s; interface ETL/MAPbI3: neutral midgap defect with N_if = S / (sigma vth) = 2.92e+08 cm-2 (S = 58.5 cm/s).

## TiO2

| Parameter | Unit | ETL (TiO2) | MAPbI3 |
|---|---|---|---|
| Eg | eV | 3.2 | 1.55 |
| chi | eV | 4.1 | 3.9 |
| eps_r | - | 9 | 6.6 |
| Nc | cm-3 | 2.2e+18 | 2.1e+18 |
| Nv | cm-3 | 1e+19 | 3.4e+18 |
| mu_n | cm2/Vs | 20 | 10 |
| mu_p | cm2/Vs | 10 | 10 |
| ND | cm-3 | 1e+18 | 1e+09 |
| NA | cm-3 | 0 | 6.5e+15 |
| Nt | cm-3 | 1e+15 | 1e+14 |

## SnO2

| Parameter | Unit | ETL (SnO2) | MAPbI3 |
|---|---|---|---|
| Eg | eV | 3.55 | 1.55 |
| chi | eV | 4 | 3.9 |
| eps_r | - | 9 | 6.6 |
| Nc | cm-3 | 2.2e+18 | 2.1e+18 |
| Nv | cm-3 | 1.8e+19 | 3.4e+18 |
| mu_n | cm2/Vs | 17.5 | 10 |
| mu_p | cm2/Vs | 1.128 | 10 |
| ND | cm-3 | 1.5e+17 | 1e+09 |
| NA | cm-3 | 0 | 6.5e+15 |
| Nt | cm-3 | 1e+15 | 1e+14 |

## ZnO

| Parameter | Unit | ETL (ZnO) | MAPbI3 |
|---|---|---|---|
| Eg | eV | 3.3 | 1.55 |
| chi | eV | 4.3 | 3.9 |
| eps_r | - | 9 | 6.6 |
| Nc | cm-3 | 4e+18 | 2.1e+18 |
| Nv | cm-3 | 1e+19 | 3.4e+18 |
| mu_n | cm2/Vs | 100 | 10 |
| mu_p | cm2/Vs | 25 | 10 |
| ND | cm-3 | 1e+16 | 1e+09 |
| NA | cm-3 | 0 | 6.5e+15 |
| Nt | cm-3 | 1e+15 | 1e+14 |

## CdS

| Parameter | Unit | ETL (CdS) | MAPbI3 |
|---|---|---|---|
| Eg | eV | 2.41 | 1.55 |
| chi | eV | 3.965 | 3.9 |
| eps_r | - | 10 | 6.6 |
| Nc | cm-3 | 6.95e+18 | 2.1e+18 |
| Nv | cm-3 | 1.156e+19 | 3.4e+18 |
| mu_n | cm2/Vs | 130 | 10 |
| mu_p | cm2/Vs | 20 | 10 |
| ND | cm-3 | 1e+18 | 1e+09 |
| NA | cm-3 | 0 | 6.5e+15 |
| Nt | cm-3 | 5.5e+15 | 1e+14 |
