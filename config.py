"""Paths, covariate lists, the claiming window, the random seed, and the counts the paper reports."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
TABLES_DIR = ROOT / "output" / "tables"

RAND_HRS_DTA = DATA_DIR / "raw" / "public" / "randhrs1992_2022v1.dta"
POA_DTA = Path(os.environ.get(
    "HRS_POA_DTA", str(DATA_DIR / "raw" / "restricted" / "HRSPoA_Shared.dta")))
# Public HRS Tracker file with National Death Index dates. Only the estimate
# with those deaths added (note to Table 1) needs it.
TRACKER_DTA = Path(os.environ.get(
    "HRS_TRACKER_DTA", str(DATA_DIR / "raw" / "public" / "trk2022tr_r.dta")))
TRACKER_AVAILABLE = TRACKER_DTA.exists()

# Respondent-level intermediate files, written by claiming_sample.py and mortality.py.
PERSON_PARQUET = DERIVED_DIR / "person.parquet"
PANEL_PARQUET = DERIVED_DIR / "person_month.parquet"
START_BOUNDS_PARQUET = DERIVED_DIR / "start_date_bounds.parquet"
REFERENCE_PARQUET = DERIVED_DIR / "reference_survival.parquet"

POA_VAR = "paceofaging"
CLAIMING_AGE_VAR = "rassageb"      # RAND: age at first Social Security receipt
CLAIMING_AGE_MIN = 62
CLAIMING_AGE_MAX = 70

# Controls. The claiming hazards use DEMOGRAPHICS with five-year birth-cohort
# and completion-wave indicators. The mortality models and the regression of
# claiming age on all observed claimants use COVARIATES, which adds birth year.
DEMOGRAPHICS = ["female", "raedyrs", "black", "otherrace", "hisp"]
COVARIATES = ["rabyear", "raedyrs"] + [c for c in DEMOGRAPHICS if c != "raedyrs"]
CLUSTER_VAR = "hhid"               # household

RANDOM_SEED = 20260403
N_BOOT = int(os.environ.get("BOOTSTRAP_B", "999"))   # household bootstrap replicates
N_IMPUTATIONS = 25

# Label of the claiming sample in the disposition column of the person file.
CLAIMING_SAMPLE = "The claiming sample"

# Counts reported in the paper, asserted by the scripts that produce them.
N_RELEASE = 13_358                 # Pace of Aging release
N_CLAIMING_SAMPLE = 2_484
N_CLAIMS = 1_393
N_DEATHS_BEFORE_CLAIM = 73
N_HOUSEHOLDS = 2_107
N_RETROSPECTIVE = 7_460            # all observed claimants, Appendix Table A3
N_DEATHS_AFTER_70 = 16             # Appendix Table A4

# Appendix Table A1: respondents by disposition.
FLOW_COUNTS = {
    "Missing covariates": 84,
    "No under-70 claiming window": 6_766,
    "Known beneficiary, start date unavailable": 853,
    "OASDI receipt before entry": 2_636,
    "Death before entry": 60,
    "No observation after entry": 475,
    CLAIMING_SAMPLE: N_CLAIMING_SAMPLE,
    "Claim in window": N_CLAIMS,
    "Death before claim": N_DEATHS_BEFORE_CLAIM,
    "Censored at last observation": 912,
    "Unclaimed at 70": 106,
}
# Note to Appendix Table A1: the highest third of the release (all, measurement
# complete at or after 70, in the claiming sample), and the unclaimed at 70.
RELEASE_HIGHEST_THIRD = (4_453, 3_881, 66)
UNCLAIMED_AT_70 = {"late_receipt": 39, "in_month_after_70th_birthday_month": 22,
                   "no_receipt": 67}

# Table 1 and its note: (respondents, deaths).
MORTALITY_COUNTS = {
    "entry_at_completion": (12_782, 3_865),
    "entry_at_year2": (13_191, 3_860),
    "entry_at_completion_ndi": (12_859, 4_164),
}
QUARTILE_MORTALITY_COUNTS = {
    "entered_at_60_to_69": {"all": (4_021, 587), "Q1": (1_006, 61), "Q2": (1_005, 83),
                            "Q3": (1_005, 130), "Q4": (1_005, 313)},
    "claiming_sample": {"all": (2_483, 133), "Q1": (621, 18), "Q2": (620, 17),
                        "Q3": (621, 30), "Q4": (621, 68)},
}

# Table 2, panel A: (respondents, claims, deaths before claiming). Panel B:
# respondents who had entered by each age.
QUARTILE_CLAIMING_COUNTS = {
    "all": (N_CLAIMING_SAMPLE, N_CLAIMS, N_DEATHS_BEFORE_CLAIM),
    "Q1": (621, 323, 7), "Q2": (621, 329, 11), "Q3": (621, 350, 17),
    "Q4": (621, 391, 38),
}
ENTERED_BY_AGE = {63: 2_017, 65: 2_306, "FRA": 2_401, 70: N_CLAIMING_SAMPLE}

# Table 3 and Appendix Table A4: the claiming sample with reported survival
# expectations, (respondents, claims) and, for panel C, (respondents, deaths).
EXPECTATIONS_COUNTS = {"all": (2_447, 1_375), "with_self_rated_health": (2_445, 1_374),
                       "live_to_75": (2_265, 1_265)}
EXPECTATIONS_MORTALITY_COUNTS = (2_444, 129)

# Appendix Table A3: (respondents, claims).
SENSITIVITY_COUNTS = {
    "entry_in_completion_month": (2_560, 1_420),
    "censor_at_last_interview": (2_436, 1_389),
    "cohorts_born_by_1952": (691, 504),
    "attrition_weights": (2_483, 1_393),
    "three_time_points": (731, 530),
    "eight_or_more_slopes": (2_015, 1_150),
}
