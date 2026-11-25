"""
Pace of Aging and death (Table 1), and the reference survival curves for the
present-value benchmark (Appendix Table A5).

Mortality model: Cox regression on the age timescale over the Pace of Aging
release, with entry the month after the measurement window closed, censoring
at the last observation, birth year, sex, years of education, and race and
ethnicity as controls, and household-clustered standard errors on the
delayed-entry risk sets. The note to Table 1 reports two variants: entry at
mid-year of the second assessment year, and National Death Index deaths
added from the public Tracker file.

Observed mortality: Kaplan-Meier probability of death within five years of
entry, by quartile of Pace of Aging, for respondents who entered follow-up
at ages 60 to 69 (panel B) and for the claiming sample followed from entry
into that sample (panel C). The model-implied five-year chance from age 65
is evaluated at the claiming sample's 25th and 75th percentiles.

Reference curves: survival from age 62 for a respondent at the claiming
sample's mean Pace of Aging (z = 0); the benchmark raises a curve to HR^z to
move it to another value. The curve the benchmark uses, ``mixture_z0``, is
solved so that averaging the curves S_0^(HR^z) over the mortality sample's
Pace of Aging reproduces that sample's delayed-entry Kaplan-Meier survival
in every month. Three alternatives are the Kaplan-Meier curve itself read as
the curve at the sample's mean Pace of Aging (``km``) and Cox-based curves
with and without birth year (``cox_full``, ``cox_nobyear``).

Writes:
- output/tables/mortality_model.csv               Table 1, panel A and notes
- output/tables/mortality_by_quartile.csv         Table 1, panels A to C
- output/tables/reference_curve_calibration.csv   note to Appendix Table A5
- data/derived/reference_survival.parquet         monthly curves, ages 62 to 100
"""

import numpy as np
import pandas as pd
import pyreadstat
from lifelines import KaplanMeierFitter

from config import (
    CLAIMING_SAMPLE, CLUSTER_VAR, COVARIATES, MORTALITY_COUNTS, PERSON_PARQUET,
    QUARTILE_MORTALITY_COUNTS, REFERENCE_PARQUET, TABLES_DIR, TRACKER_AVAILABLE,
    TRACKER_DTA,
)
from cox_tools import breslow_survival, fit_cox as fit_cox_clustered

YEAR2_MIDYEAR_MONTH = 6
MAX_AGE = 100
MONTH_GRID = np.arange(62 * 12, MAX_AGE * 12 + 1)
CRUDE_BAND_YEARS = 2
MIXTURE_TOL = 1e-10
MIXTURE_MAX_ITER = 100
SIXTIES_ENTRY = (60.0, 70.0)   # entry ages of the Table 1 panel B sample
HORIZON_YEARS = 5.0
IMPLIED_FROM_AGE = 65.0

# The public Tracker codes the National Death Index month as a quarter.
NDI_QUARTER_MIDPOINT = {1: 2, 2: 5, 3: 8, 4: 11}
NDI_YEAR_ONLY_MONTH = 7


def ndi_death_month(p):
    """Death months from the Tracker's National Death Index fields for
    respondents the RAND file records as alive.

    The reported death interval is a quarter, or a year when the quarter is
    missing. The reference date is the later of the last observed month and
    the Tracker's own last-known-alive date. An interval that ends before the
    reference date is excluded. Otherwise the death is dated at the
    interval's midpoint (July for a year), or at the reference month when the
    midpoint precedes it. Returns NaN where no death is added.
    """
    cols = ["HHID", "PN", "NYEAR", "NMONTH", "NSCORE", "LASTALIVEYR", "LASTALIVEMO"]
    trk, _ = pyreadstat.read_dta(str(TRACKER_DTA), usecols=cols)
    for c in cols[2:]:
        trk[c] = pd.to_numeric(trk[c], errors="coerce")
    trk["hhidpn"] = (trk["HHID"].astype(str).str.strip().astype(int) * 1000
                     + trk["PN"].astype(str).str.strip().astype(int))
    ndi = p[["hhidpn"]].merge(trk.drop(columns=["HHID", "PN"]),
                              on="hhidpn", how="left")
    alive = p["death_month_idx"].isna().to_numpy()
    use = alive & ndi["NYEAR"].notna().to_numpy()
    assert ndi.loc[use, "NSCORE"].notna().all(), "NDI death without a match score"
    assert ndi.loc[use, ["LASTALIVEYR", "LASTALIVEMO"]].notna().all().all(), \
        "NDI death without a last-alive date"
    q = ndi["NMONTH"].to_numpy()
    year_only = np.isnan(q)
    mid = np.where(year_only, NDI_YEAR_ONLY_MONTH,
                   pd.Series(q).map(NDI_QUARTER_MIDPOINT).to_numpy())
    start = ndi["NYEAR"].to_numpy() * 12 + mid - 1
    end = ndi["NYEAR"].to_numpy() * 12 + np.where(year_only, 12, q * 3) - 1
    last_alive = (ndi["LASTALIVEYR"].to_numpy() * 12
                  + ndi["LASTALIVEMO"].to_numpy() - 1)
    ref = np.maximum(p["last_obs_month_idx"].to_numpy(), last_alive)
    ok = use & (end >= ref)
    out = pd.Series(np.nan, index=p.index)
    out[ok] = np.where(start > ref, start, ref)[ok]
    return out


def load_survival_frame():
    """Release respondents with complete covariates: entry ages, exit age, death."""
    p = pd.read_parquet(PERSON_PARQUET)
    p["died"] = p["death_month_idx"].notna().astype(int)
    exit_month = np.where(p["died"] == 1, p["death_month_idx"],
                          p["last_obs_month_idx"])
    p["exit_age"] = (exit_month - p["birth_month_idx"] + 1) / 12
    if TRACKER_AVAILABLE:
        ndi = ndi_death_month(p)
        p["died_ndi"] = ((p["died"] == 1) | ndi.notna()).astype(int)
        exit_ndi = np.where(p["died"] == 1, p["death_month_idx"],
                            np.where(ndi.notna(), ndi, p["last_obs_month_idx"]))
        p["exit_age_ndi"] = (exit_ndi - p["birth_month_idx"] + 1) / 12
    p["entry_age_completion"] = (p["completion_month_idx"] + 1
                                 - p["birth_month_idx"]) / 12
    p["entry_age_sample"] = (p["entry_month_idx"] - p["birth_month_idx"]) / 12
    p["entry_age_year2"] = (p["year_2"] * 12 + YEAR2_MIDYEAR_MONTH
                            - p["birth_month_idx"]) / 12
    keep = [CLUSTER_VAR, "poa_z", "died", "exit_age", "entry_age_completion",
            "entry_age_year2", "entry_age_sample", "disposition",
            "poa_quartile"] + COVARIATES
    if TRACKER_AVAILABLE:
        keep += ["died_ndi", "exit_age_ndi"]
    return p[keep].dropna(subset=["poa_z"] + COVARIATES).copy()


def fit_cox(frame, entry_col, include_byear=True):
    """The mortality model on respondents with follow-up after ``entry_col``."""
    covs = [c for c in COVARIATES if include_byear or c != "rabyear"]
    df = frame.loc[frame["exit_age"] > frame[entry_col],
                   ["exit_age", "died", CLUSTER_VAR, "poa_z", entry_col]
                   + covs].copy()
    cph = fit_cox_clustered(df, duration_col="exit_age", event_col="died",
                            entry_col=entry_col, cluster_col=CLUSTER_VAR,
                            formula="poa_z + " + " + ".join(covs))
    return cph, df


def hazard_ratio_row(label, cph, df):
    assert (len(df), int(df["died"].sum())) == MORTALITY_COUNTS[label], label
    s = cph.inference.loc["poa_z"]
    return {"model": label, "n": len(df), "deaths": int(df["died"].sum()),
            "hr_per_sd": float(np.exp(s["coef"])),
            "hr_ci_lo": float(np.exp(s["coef lower 95%"])),
            "hr_ci_hi": float(np.exp(s["coef upper 95%"]))}


def _to_month_grid(sf):
    grid_years = MONTH_GRID / 12
    s = (sf.reindex(sf.index.union(grid_years)).sort_index()
         .interpolate().reindex(grid_years))
    s = s.ffill().bfill().clip(lower=1e-12).to_numpy()
    return s / s[0]


def cox_curve(cph, df):
    """Survival from 62 for a respondent at z = 0 with the other covariates
    at their means: the Breslow baseline at the mean profile, shifted from
    the sample's mean Pace of Aging to z = 0 by the fit's own hazard ratio."""
    sf = breslow_survival(cph, df, "exit_age", "died", "entry_age_completion")
    hr = float(np.exp(cph.params_["poa_z"]))
    return _to_month_grid(sf[sf.index >= 62]) ** (hr ** (-float(df["poa_z"].mean())))


def km_reference(km_grid, hr, zbar):
    """The Kaplan-Meier curve read as the curve of a respondent at the
    sample's mean Pace of Aging ``zbar`` and shifted to z = 0 by ``hr``."""
    return km_grid ** (hr ** (-zbar))


def _surv_at(curve, age):
    """Step-function value of a survival series indexed by exit age."""
    j = int(curve.index.searchsorted(age, side="right")) - 1
    return 1.0 if j < 0 else float(curve.iloc[j])


def _km_row(group, label, block):
    """Deaths and the Kaplan-Meier probability of death within HORIZON_YEARS of entry."""
    km = KaplanMeierFitter().fit(group["t"], group["died"])
    s = float(km.survival_function_at_times(HORIZON_YEARS).iloc[0])
    ci = km.confidence_interval_
    ci_row = ci.loc[ci.index <= HORIZON_YEARS].iloc[-1]
    lo_s, hi_s = float(ci_row.iloc[0]), float(ci_row.iloc[1])
    n, deaths = int(len(group)), int(group["died"].sum())
    assert (n, deaths) == QUARTILE_MORTALITY_COUNTS[block][label], (block, label)
    row = {"block": block, "group": label, "n": n, "deaths": deaths,
           "mortality_5y": 1 - s, "mortality_5y_ci_lo": 1 - hi_s,
           "mortality_5y_ci_hi": 1 - lo_s}
    if label != "all":
        row.update({"poa_z_min": float(group["poa_z"].min()),
                    "poa_z_max": float(group["poa_z"].max())})
    return row


def mortality_by_quartile(frame, cph, df, p25, p75):
    """Table 1: the hazard ratio across the claiming sample's interquartile
    range and the implied five-year chances (panel A), and observed mortality
    by quartile (panels B and C). ``cph`` and ``df`` are the mortality model
    and its sample; ``p25`` and ``p75`` are the claiming sample's percentiles
    of Pace of Aging."""
    lo_age, hi_age = SIXTIES_ENTRY
    sub = frame.loc[(frame["entry_age_completion"] >= lo_age)
                    & (frame["entry_age_completion"] < hi_age)
                    & (frame["exit_age"] > frame["entry_age_completion"])].copy()
    sub["t"] = sub["exit_age"] - sub["entry_age_completion"]
    rows = [_km_row(sub, "all", "entered_at_60_to_69")]
    sub["quartile"] = pd.qcut(sub["poa_z"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    for q, g in sub.groupby("quartile", observed=True):
        rows.append(_km_row(g, str(q), "entered_at_60_to_69"))

    # The claiming sample, followed from entry into it, by its own quartiles.
    # Respondents without follow-up after entry contribute nothing.
    cs = frame.loc[(frame["disposition"] == CLAIMING_SAMPLE)
                   & (frame["exit_age"] > frame["entry_age_sample"])].copy()
    cs["t"] = cs["exit_age"] - cs["entry_age_sample"]
    rows.append(_km_row(cs, "all", "claiming_sample"))
    for q, g in cs.groupby("poa_quartile", observed=True):
        rows.append(_km_row(g, str(q), "claiming_sample"))

    # Implied chance of dying within HORIZON_YEARS of IMPLIED_FROM_AGE: the
    # model's Breslow baseline at the panel B sample's mean covariates,
    # shifted to each percentile by the Pace of Aging coefficient.
    profile = sub[list(cph.params_.index)].mean()
    curve = breslow_survival(cph, df, "exit_age", "died",
                             "entry_age_completion", profile=profile.to_numpy())
    beta = float(cph.params_["poa_z"])
    zbar = float(profile["poa_z"])
    base_ratio = (_surv_at(curve, IMPLIED_FROM_AGE + HORIZON_YEARS)
                  / _surv_at(curve, IMPLIED_FROM_AGE))
    for label, z in (("p25", p25), ("p75", p75)):
        rows.append({"block": "implied_from_65", "group": label,
                     "mortality_5y": 1 - base_ratio ** np.exp(beta * (z - zbar))})
    s = cph.inference.loc["poa_z"]
    width = p75 - p25
    rows.append({"block": "hazard_ratio", "group": "p75_vs_p25",
                 "estimate": float(np.exp(beta * width)),
                 "ci_lo": float(np.exp(s["coef lower 95%"] * width)),
                 "ci_hi": float(np.exp(s["coef upper 95%"] * width))})
    return pd.DataFrame(rows)


def calibration_population(frame):
    """Sample for the reference curves: the mortality model's sample with
    exits after age 62, entering at the later of its entry and 62."""
    f = frame.loc[(frame["exit_age"] > frame["entry_age_completion"])
                  & (frame["exit_age"] > 62)].copy()
    f["entry62"] = np.maximum(f["entry_age_completion"], 62.0)
    return f


def calibrate_reference(pop, hr):
    """Kaplan-Meier survival from 62 on the monthly grid, and the z = 0
    curve whose Pace of Aging mixture reproduces it."""
    km = KaplanMeierFitter()
    km.fit(pop["exit_age"], event_observed=pop["died"], entry=pop["entry62"])
    km_grid = _to_month_grid(km.survival_function_.iloc[:, 0])
    return km_grid, mixture_z0_curve(km_grid, hr, pop["poa_z"].to_numpy())


def crude_curve(frame):
    """Survival implied by crude death rates in CRUDE_BAND_YEARS-year age bands."""
    e = frame["entry_age_completion"].to_numpy()
    x = frame["exit_age"].to_numpy()
    d = frame["died"].to_numpy()
    s, out = 1.0, np.ones_like(MONTH_GRID, dtype=float)
    for lo in np.arange(62, MAX_AGE, CRUDE_BAND_YEARS):
        hi = lo + CRUDE_BAND_YEARS
        py = np.clip(np.minimum(x, hi) - np.maximum(e, lo), 0, None).sum()
        deaths = ((d == 1) & (x > lo) & (x <= hi)).sum()
        rate = deaths / py if py > 0 else 0.0
        band = (MONTH_GRID / 12 > lo) & (MONTH_GRID / 12 <= hi)
        months_in = np.cumsum(band[band]) / 12
        out[band] = s * np.exp(-rate * months_in)
        s = s * np.exp(-rate * CRUDE_BAND_YEARS)
    return out


def mixture_z0_curve(km, hr, z):
    """Baseline z = 0 curve whose Pace of Aging mixture reproduces ``km``.

    Solves mean_i(S_0(m)^(HR^{z_i})) = KM(m) for S_0(m), month by month, by
    Newton iteration on u = log S_0 started from the previous month's
    solution.
    """
    w = hr ** np.asarray(z, dtype=float)
    u = 0.0
    out = np.empty_like(km)
    for j, k in enumerate(km):
        if k >= 1.0:
            out[j] = 1.0
            u = 0.0
            continue
        for _ in range(MIXTURE_MAX_ITER):
            e = np.exp(w * u)
            f = e.mean() - k
            if abs(f) < MIXTURE_TOL:
                break
            u = min(u - f / (w * e).mean(), 0.0)
        out[j] = np.exp(u)
    return out


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_survival_frame()

    cph, df = fit_cox(frame, "entry_age_completion")
    hr = float(np.exp(cph.params_["poa_z"]))
    rows = [hazard_ratio_row("entry_at_completion", cph, df)]
    rows.append(hazard_ratio_row("entry_at_year2", *fit_cox(frame, "entry_age_year2")))
    if TRACKER_AVAILABLE:
        ndi = frame.drop(columns=["died", "exit_age"]).rename(
            columns={"died_ndi": "died", "exit_age_ndi": "exit_age"})
        rows.append(hazard_ratio_row("entry_at_completion_ndi",
                                     *fit_cox(ndi, "entry_age_completion")))
    pd.DataFrame(rows).to_csv(TABLES_DIR / "mortality_model.csv", index=False,
                              float_format="%.4f")

    counts = pd.read_csv(TABLES_DIR / "sample_counts.csv").set_index("metric")["value"]
    mortality_by_quartile(frame, cph, df, float(counts["poa_p25"]),
                          float(counts["poa_p75"])).to_csv(
        TABLES_DIR / "mortality_by_quartile.csv", index=False, float_format="%.6f")

    # Reference curves, each stored as the curve at z = 0.
    cph_n, df_n = fit_cox(frame, "entry_age_completion", include_byear=False)
    pop = calibration_population(frame)
    zbar_pop = float(pop["poa_z"].mean())
    km_raw, mixture = calibrate_reference(pop, hr)
    curves = {
        "cox_full": cox_curve(cph, df),
        "cox_nobyear": cox_curve(cph_n, df_n),
        "km": km_reference(km_raw, hr, zbar_pop),
        "mixture_z0": mixture,
    }
    pd.DataFrame({"month_of_age": MONTH_GRID, **curves}).to_parquet(
        REFERENCE_PARQUET, index=False)

    # Largest gap between the mixture curve, moved to the mortality sample's
    # mean Pace of Aging, and the curve implied by the crude death rates.
    at_pop_mean = mixture ** (hr ** zbar_pop)
    pd.DataFrame([{
        "curve": "mixture_z0",
        "max_abs_dev_from_crude_at_pop_mean": round(float(
            np.max(np.abs(at_pop_mean - crude_curve(frame)))), 4),
    }]).to_csv(TABLES_DIR / "reference_curve_calibration.csv", index=False)


if __name__ == "__main__":
    main()
