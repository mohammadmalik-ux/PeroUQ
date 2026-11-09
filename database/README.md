# PeroUQ parameter database

203 parameter values from 17 sources for MAPbI3, TiO2, SnO2, ZnO, CdS, the ETL/MAPbI3 interface and the carbon electrode, as used in the PeroUQ study. Licence: CC BY 4.0 (data); the code is MIT.

| File | Content |
|---|---|
| parameters.csv | every value with unit, source key and note (one row per value) |
| sources.csv | the sources, with kind (measurement or simulation input), citation, DOI and URL |
| decisions.csv | the author's decisions applied before sampling (excluded or restricted sources) |
| calibration_devices.csv | published devices and their metrics |
| nk_sources.csv | candidate sources of measured n,k data and their status |
| parameter_summary.csv | one row per material and parameter: spread, scale and sampled range |
| PeroUQ_database.xlsx | the same tables as sheets, with a data dictionary |

## Columns

| File | Column | Meaning |
|---|---|---|
| parameters.csv | material | Layer or interface: MAPbI3, TiO2, SnO2, ZnO, CdS, ETL/MAPbI3 (interface), carbon |
| parameters.csv | param | Eg band gap; chi electron affinity; eps_r relative permittivity; Nc, Nv effective densities of states; mu_n, mu_p mobilities; ND, NA doping; Nt bulk defect density; sigma capture cross-section; vth thermal velocity; S interface recombination velocity; W work function |
| parameters.csv | value | Numerical value as reported by the source |
| parameters.csv | unit | Unit of value |
| parameters.csv | source | Key into sources.csv |
| parameters.csv | note | Remarks (e.g. measured vs simulation input, page or table) |
| sources.csv | id | Source key |
| sources.csv | kind | measurement or simulation-input |
| sources.csv | citation / doi / url | Bibliographic reference |
| decisions.csv | material, param | What the decision applies to |
| decisions.csv | action, arg | exclude_source / only_sources and the source keys concerned |
| decisions.csv | reason, date | Why, and when it was recorded (author's decisions) |
| calibration_devices.csv | * | Published devices used for calibration or validation and their metrics |
| nk_sources.csv | * | Candidate sources of measured optical constants and their status |
| parameter_summary.csv | n_values, n_sources | Number of values and of distinct sources after the decisions |
| parameter_summary.csv | min, median, max | Spread of the retained values |
| parameter_summary.csv | scale | log (sampled log-uniformly) or linear (uniform) |
| parameter_summary.csv | sampled_low, sampled_high | Range sampled in the study (minimum-uncertainty rule x/3 for log parameters, +/-0.1 eV for energies, +/-10 % for eps_r); empty = held fixed |

The code reads the copies in `perouq/params/`; `python scripts/export_database.py` regenerates this folder from them.
