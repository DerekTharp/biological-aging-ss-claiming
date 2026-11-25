"""
Retrospective regression of claiming age on all observed claimants (a row of
Appendix Table A3).

Ordinary least squares of the age at first Social Security receipt on Pace
of Aging among every release respondent whose first receipt fell between 62
and 70, with birth year, sex, years of education, and race and ethnicity as
controls and household-clustered standard errors. Pace of Aging is in SD
units of the full release here; sensitivities.py rescales the row to the
claiming sample's SD.

Writes output/tables/retrospective_regression.csv.
"""

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm

from config import (
    CLAIMING_AGE_MAX, CLAIMING_AGE_MIN, CLAIMING_AGE_VAR, CLUSTER_VAR,
    COVARIATES, N_RETROSPECTIVE, POA_DTA, POA_VAR, RAND_HRS_DTA, TABLES_DIR,
)


def load_claimants():
    """Release respondents with a first receipt at 62 to 70 and complete controls."""
    poa, _ = pyreadstat.read_dta(str(POA_DTA))
    poa["hhidpn"] = pd.to_numeric(poa["hhidpn"], errors="coerce").astype("Int64")
    poa = poa.dropna(subset=["hhidpn"]).set_index(poa["hhidpn"].astype("int64"))
    for c in [POA_VAR, "year_baseline"]:
        poa[c] = pd.to_numeric(poa[c], errors="coerce")
    poa["poa_z_release"] = (poa[POA_VAR] - poa[POA_VAR].mean()) / poa[POA_VAR].std()

    cols = ["hhidpn", CLAIMING_AGE_VAR, "rabyear", "ragender", "raedyrs",
            "raracem", "rahispan"]
    rand, _ = pyreadstat.read_dta(str(RAND_HRS_DTA), usecols=["hhid"] + cols)
    for c in cols:
        rand[c] = pd.to_numeric(rand[c], errors="coerce")
    rand = rand.dropna(subset=["hhidpn"]).set_index(rand["hhidpn"].astype("int64"))
    # A missing source value stays missing, so the respondent is dropped below.
    rand["female"] = np.where(rand["ragender"].isna(), np.nan, rand["ragender"] == 2)
    rand["black"] = np.where(rand["raracem"].isna(), np.nan, rand["raracem"] == 2)
    rand["otherrace"] = np.where(rand["raracem"].isna(), np.nan, rand["raracem"] == 3)
    rand["hisp"] = np.where(rand["rahispan"].isna(), np.nan, rand["rahispan"] >= 1)

    df = rand.join(poa[["poa_z_release", "year_baseline"]], how="left")
    in_window = df[CLAIMING_AGE_VAR].between(CLAIMING_AGE_MIN, CLAIMING_AGE_MAX)
    return df.loc[in_window].dropna(
        subset=["poa_z_release", CLAIMING_AGE_VAR] + COVARIATES).copy()


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_claimants()
    assert len(df) == N_RETROSPECTIVE
    s = df[[CLAIMING_AGE_VAR, "poa_z_release", CLUSTER_VAR] + COVARIATES].astype(float)
    m = sm.OLS(s[CLAIMING_AGE_VAR],
               sm.add_constant(s[["poa_z_release"] + COVARIATES])).fit(
        cov_type="cluster", cov_kwds={"groups": s[CLUSTER_VAR]})
    ci = m.conf_int().loc["poa_z_release"]
    # Respondents whose measurement window opened after the year of first receipt.
    claim_year = df["rabyear"] + df[CLAIMING_AGE_VAR]
    pd.DataFrame([{
        "estimate": m.params["poa_z_release"], "ci_lo": ci.iloc[0],
        "ci_hi": ci.iloc[1], "p": m.pvalues["poa_z_release"], "n": int(m.nobs),
        "n_measured_after_claim": int((df["year_baseline"] > claim_year).sum()),
    }]).to_csv(TABLES_DIR / "retrospective_regression.csv", index=False,
               float_format="%.6f")


if __name__ == "__main__":
    main()
