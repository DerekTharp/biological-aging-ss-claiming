"""
Death as the outcome among respondents with reported survival expectations
(Table 3, panel C; mortality rows of Appendix Table A4).

Cox models on the age timescale, entered at the later of entry into the
claiming sample and the month after the interview that supplied the reported
chance, censored at the last observation, with target-age indicators and the
demographic controls, on the complete-case sample with self-rated health.
Household-clustered standard errors are computed on the delayed-entry risk
sets. Coefficients are log hazard ratios.

The last three rows of Appendix Table A4 split each follow-up spell at age
70 and interact Pace of Aging, self-rated health, and the gap with an
after-70 indicator in one counting-process Cox model (model-based standard
errors).

Writes output/tables/mortality_gap.csv.
"""

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter

from config import (
    CLUSTER_VAR, COVARIATES, EXPECTATIONS_MORTALITY_COUNTS, N_DEATHS_AFTER_70,
    TABLES_DIR,
)
from cox_tools import fit_cox
from subjective_survival import GAP_UNIT, load_sample, target_dummies

MODELS = {
    "poa": ["poa_z"],
    "poa_srh": ["poa_z", "srh"],
    "poa_srh_gap": ["poa_z", "srh", "gap_10"],
    "gap": ["gap_10"],
    "gap_poa": ["gap_10", "poa_z"],
}
# Table 3 reports hazard ratios from the two models with the gap and Pace of
# Aging together. Appendix Table A4 reports log hazard ratios from the other
# three, and only the Pace of Aging row of the model with self-rated health.
HAZARD_RATIO_MODELS = ("poa_srh_gap", "gap_poa")
UNREPORTED_TERMS = {"poa_srh": ["srh"]}
SPLIT_AGE = 70.0


def survival_frame(sample):
    surv = sample.copy()
    surv["died"] = surv["death_month_idx"].notna().astype(int)
    exit_month = np.where(surv["died"] == 1, surv["death_month_idx"],
                          surv["last_obs_month_idx"])
    surv["entry_age"] = ((surv["gap_start_month_idx"]
                          - surv["birth_month_idx"]) / 12)
    surv["exit_age"] = (exit_month - surv["birth_month_idx"] + 1) / 12
    surv = surv.loc[surv["exit_age"] > surv["entry_age"]]
    surv = surv.dropna(subset=["srh", "gap_10", "poa_z"]).copy()
    surv["srh"] = surv["srh"].astype(float)
    return surv, target_dummies(surv, surv)


def after_70_interactions(surv, covariates):
    """Interactions with an after-70 indicator in the model with all three variables.

    Each follow-up spell is split at SPLIT_AGE into (start, stop] episodes.
    The indicator itself is a function of age and is absorbed by the baseline
    hazard, so only the interactions enter.
    """
    episodes = []
    for _, r in surv.iterrows():
        start, stop = float(r["entry_age"]), float(r["exit_age"])
        base = {c: float(r[c]) for c in covariates}
        base["hhidpn"] = r["hhidpn"]
        if stop <= SPLIT_AGE or start >= SPLIT_AGE:
            late = 1.0 if start >= SPLIT_AGE else 0.0
            episodes.append({**base, "start": start, "stop": stop,
                             "died": int(r["died"]), "late": late})
        else:
            episodes.append({**base, "start": start, "stop": SPLIT_AGE,
                             "died": 0, "late": 0.0})
            episodes.append({**base, "start": SPLIT_AGE, "stop": stop,
                             "died": int(r["died"]), "late": 1.0})
    ep = pd.DataFrame(episodes)
    terms = ["poa_z", "srh", "gap_10"]
    for c in terms:
        ep[f"{c}_late"] = ep[c] * ep["late"]
    fit_cols = covariates + [f"{c}_late" for c in terms]
    ctv = CoxTimeVaryingFitter()
    ctv.fit(ep[["hhidpn", "start", "stop", "died"] + fit_cols],
            id_col="hhidpn", event_col="died", start_col="start",
            stop_col="stop")
    rows = []
    for c in terms:
        s = ctv.summary.loc[f"{c}_late"]
        rows.append({"model": "after_70_interactions", "term": f"{c}_x_after70",
                     "estimate": float(s["coef"]),
                     "ci_lo": float(s["coef lower 95%"]),
                     "ci_hi": float(s["coef upper 95%"]), "p": float(s["p"])})
    deaths_after_70 = int(ep.loc[ep["late"] == 1.0, "died"].sum())
    assert deaths_after_70 == N_DEATHS_AFTER_70
    rows.append({"model": "after_70_interactions", "term": "deaths_after_70",
                 "estimate": deaths_after_70})
    return rows


def main():
    sample = load_sample()
    surv, tcols = survival_frame(sample)
    n, deaths = len(surv), int(surv["died"].sum())
    assert (n, deaths) == EXPECTATIONS_MORTALITY_COUNTS
    base = ["entry_age", "exit_age", "died", CLUSTER_VAR]
    counts = pd.read_csv(TABLES_DIR / "sample_counts.csv").set_index("metric")["value"]
    gap_iqr = float(sample["gap"].quantile(0.75) - sample["gap"].quantile(0.25))
    rows = []
    for model, terms in MODELS.items():
        covariates = terms + tcols + COVARIATES
        cph = fit_cox(surv[base + covariates], duration_col="exit_age",
                      event_col="died", entry_col="entry_age",
                      cluster_col=CLUSTER_VAR, formula=" + ".join(covariates))
        for term in terms:
            if term in UNREPORTED_TERMS.get(model, []):
                continue
            s = cph.inference.loc[term]
            row = {"model": model, "term": f"{term}_log_hr"}
            if model in HAZARD_RATIO_MODELS:
                row.update({"hr": float(np.exp(s["coef"])),
                            "hr_ci_lo": float(np.exp(s["coef lower 95%"])),
                            "hr_ci_hi": float(np.exp(s["coef upper 95%"]))})
            else:
                row.update({"estimate": float(s["coef"]),
                            "ci_lo": float(s["coef lower 95%"]),
                            "ci_hi": float(s["coef upper 95%"])})
            rows.append({**row, "p": float(s["p"]), "n": n, "n_events": deaths})
        if model == "gap_poa":
            # Hazard ratios across each interquartile range (note to Table 3):
            # the gap's 25th percentile against its 75th, and Pace of Aging's
            # 75th against its 25th.
            b = cph.params_
            rows.append({"model": model, "term": "gap_p25_vs_p75",
                         "hr": float(np.exp(-b["gap_10"] * gap_iqr / GAP_UNIT))})
            rows.append({"model": model, "term": "poa_p75_vs_p25",
                         "hr": float(np.exp(b["poa_z"] * float(counts["poa_iqr"])))})
        if model == "poa_srh_gap":
            rows += after_70_interactions(surv, covariates)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "mortality_gap.csv", index=False,
                              float_format="%.6f")


if __name__ == "__main__":
    main()
