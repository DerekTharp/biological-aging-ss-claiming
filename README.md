# Biological aging, survival beliefs, and Social Security claiming

Replication code for:

> Tharp, D. Biological aging, survival beliefs, and Social Security claiming. Under review at *Health Economics*.

The code builds the claiming sample from the Health and Retirement Study (HRS) files and produces the estimates and counts reported in Tables 1 to 3 and Appendix Tables A1 to A5.

## Software

Python 3.14 with the packages pinned in `requirements.txt`.

## Data

The data are not included. HRS distributes them to registered users at https://hrsdata.isr.umich.edu/.

| File | Access | Place it at |
|---|---|---|
| [RAND HRS Longitudinal File 2022 (V1)](https://hrsdata.isr.umich.edu/data-products/rand-hrs-archived-data-products), Stata format | Public, with HRS registration | `data/raw/public/randhrs1992_2022v1.dta` |
| [HRS Pace of Aging contributed dataset](https://hrsdata.isr.umich.edu/data-products/pace-aging-hrs-hrspoa), October 2025 release, `HRSPoA_Shared.dta` | Sensitive health data, with a separate HRS order form (https://hrsdata.isr.umich.edu/data-products/sensitive-health) | `data/raw/restricted/HRSPoA_Shared.dta` |
| [HRS Tracker 2022 file](https://hrsdata.isr.umich.edu/data-products/cross-wave-tracker-file), final version 1, Stata format | Public, with HRS registration | `data/raw/public/trk2022tr_r.dta` |

Use the named releases. If the final version 1 of the Tracker 2022 file is unavailable after sign-in, request it from HRS User Support.

The Tracker file supplies National Death Index deaths for one estimate in the note to Table 1. Without it, `mortality.py` skips that estimate and everything else is unchanged. The environment variables `HRS_POA_DTA` and `HRS_TRACKER_DTA` can point to the Pace of Aging and Tracker files in another location.

The statutory full retirement ages and delayed-retirement credits in `benefit_schedule.py` are from https://www.ssa.gov/oact/ProgData/ar_drc.html.

## Running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the scripts in the order of the table below and stops if one fails. A full run takes about half an hour. Bootstrap intervals use 999 household replicates and a fixed seed (`config.py`); the environment variable `BOOTSTRAP_B` changes the number of replicates, and the reported intervals need the default.

The estimates and counts are written to `output/tables/` as CSV files. Intermediate files with one row per respondent or per respondent-month are written to `data/derived/`. They are derived from the HRS files, fall under the HRS access conditions, and must not be shared.

## Scripts

`config.py` holds the paths, the covariate lists, the seed, and the counts the paper reports, which the scripts assert. `cox_tools.py` computes Cox standard errors and baseline survival on delayed-entry risk sets. `benefit_schedule.py` holds the Social Security schedule.

| Script | Produces | Output in `output/tables/` |
|---|---|---|
| `claiming_sample.py` | Appendix Table A1; the sample counts, Pace of Aging percentiles, and SD ratios in Section 2, the table titles, and the note to Table A1 | `sample_flow.csv`, `sample_counts.csv` |
| `mortality.py` | Table 1; the calibration check in the note to Appendix Table A5 | `mortality_model.csv`, `mortality_by_quartile.csv`, `reference_curve_calibration.csv` |
| `claiming.py` | Table 2; the primary and Cox rows and the age-interaction test of Appendix Table A3; the 90% intervals in Appendix Table A5, panel B | `claiming_by_quartile.csv`, `claiming_shares.csv`, `claiming_hazard.csv` |
| `subjective_survival.py` | Table 3, panels A and B, with the shares under panel B; the claiming rows of Appendix Table A4; the summaries of the subjective survival gap in Section 2.1 | `subjective_survival.csv`, `gap_claiming_shares.csv` |
| `mortality_gap.py` | Table 3, panel C, and the hazard ratios in its note; the mortality rows of Appendix Table A4 | `mortality_gap.csv` |
| `benchmark.py` | Appendix Table A5 | `benchmark.csv`, `benchmark_test.csv`, `benchmark_reference_curves.csv`, `equivalence_test.csv` |
| `retrospective.py` | Input to the retrospective row of Appendix Table A3 | `retrospective_regression.csv` |
| `sensitivities.py` | Appendix Table A3, other rows | `sensitivities.csv` |
| `interactions.py` | Wealth and partnership interaction tests in the note to Appendix Table A3 | `interactions.csv` |
| `timing.py` | Appendix Table A2; the profile of respondents unclaimed at 70 in the note to Appendix Table A1 | `timing.csv` |
| `claims_at_70.py` | The recount of claims at 70 in the note to Appendix Table A1 | `claims_at_70.csv` |

## License

MIT (see `LICENSE`).
