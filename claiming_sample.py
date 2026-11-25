"""
Build the claiming sample (Appendix Table A1).

Links the Pace of Aging release to the RAND HRS Longitudinal File and puts
each respondent on a monthly timeline. The measurement window closes at the
completion-wave interview. A respondent enters at the later of the
62nd-birthday month and the month after the window closed, and exits at the
first of a first OASDI benefit, death, the last observation, or the
70th-birthday month. Respondents whose entry would fall at or after the
70th-birthday month have no claiming window. The claiming sample is expanded
into a person-month panel: one row per month from entry through exit,
labelled with its year of age (62 to 69), with the 70th-birthday month as its
own final interval.

Pace of Aging is standardized within the claiming sample: zero is its mean
there and one unit is its SD there.

Writes:
- data/derived/person.parquet             one row per release respondent
- data/derived/person_month.parquet       the claiming sample's person-month panel
- data/derived/start_date_bounds.parquet  bounds on the start month of beneficiaries
                                          whose start date is unavailable
- output/tables/sample_flow.csv           Appendix Table A1
- output/tables/sample_counts.csv         counts, percentiles, and SD ratios quoted in
                                          Section 2, the table titles, and the note to
                                          Appendix Table A1
"""

import numpy as np
import pandas as pd
import pyreadstat

from config import (
    CLAIMING_SAMPLE, DEMOGRAPHICS, DERIVED_DIR, FLOW_COUNTS, N_CLAIMING_SAMPLE,
    N_CLAIMS, N_DEATHS_BEFORE_CLAIM, N_HOUSEHOLDS, N_RELEASE, PANEL_PARQUET,
    PERSON_PARQUET, POA_DTA, POA_VAR, RAND_HRS_DTA, RELEASE_HIGHEST_THIRD,
    START_BOUNDS_PARQUET, TABLES_DIR,
)

# A month index is year * 12 + (month - 1). RAND's rassagem is the age in
# months at first receipt, so the claim month is the birth month index plus
# rassagem.
AGE62_MONTHS = 62 * 12
AGE70_MONTHS = 70 * 12
AGES = range(62, 70)            # single years of age in the panel
AGE70_LABEL = 70                # panel label of the 70th-birthday month
DEATH_MONTH_FALLBACK = 7        # July for deaths dated to a year only
COHORT_BIN_YEARS = 5

COMPLETION_WAVE = {2010: 10, 2012: 11, 2014: 12, 2016: 13}   # completion year to HRS wave
EXPECTATION_WAVES = range(10, 16)   # waves that can supply the reported survival chance
FOLLOWUP_WAVES = range(10, 17)      # waves that can supply observation after entry
RECEIPT_WAVES = range(1, 17)        # waves with OASDI income reports
INTERVIEW_YEAR_WAVES = range(1, 17)
INTERVIEW_MONTH_WAVES = range(8, 17)

RELEASE_ROW = "Pace of Aging release, merged to RAND"
EXCLUSIONS = [
    "Missing covariates",
    "No under-70 claiming window",
    "Known beneficiary, start date unavailable",
    "OASDI receipt before entry",
    "Death before entry",
    "No observation after entry",
]
OUTCOMES = [
    "Claim in window",
    "Death before claim",
    "Censored at last observation",
    "Unclaimed at 70",
]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _tags(s):
    """Stata extended-missing letter codes from a user_missing=True column."""
    return s.map(lambda v: v if isinstance(v, str) else np.nan)


def load_poa():
    """Pace of Aging release with the completion wave, the score standardized
    over the full release, and the release's thirds."""
    df, _ = pyreadstat.read_dta(str(POA_DTA))
    assert len(df) == N_RELEASE, f"Pace of Aging release N = {len(df)}"
    df["hhidpn"] = pd.to_numeric(df["hhidpn"], errors="coerce").astype("int64")
    assert df["hhidpn"].notna().all() and df["hhidpn"].is_unique
    for c in [POA_VAR, "year_baseline", "follow_up_time", "year_2",
              "three_time_point", "slope_count"]:
        df[c] = _num(df[c])
    assert df[POA_VAR].notna().all()

    df["poa_z_release"] = (df[POA_VAR] - df[POA_VAR].mean()) / df[POA_VAR].std()
    t1, t2 = df["poa_z_release"].quantile([1 / 3, 2 / 3])
    df["release_third"] = np.select(
        [df["poa_z_release"] <= t1, df["poa_z_release"] <= t2],
        ["lowest", "middle"], "highest")

    df["female_poa"] = df["sex"].map({"Female": 1.0, "Male": 0.0})
    assert df["female_poa"].notna().all(), "unmapped sex label"

    df["completion_year"] = df["year_baseline"] + df["follow_up_time"]
    df["completion_wave"] = df["completion_year"].map(COMPLETION_WAVE)
    assert df["completion_wave"].notna().all(), \
        "completion year outside {2010, 2012, 2014, 2016}"

    keep = ["hhidpn", POA_VAR, "poa_z_release", "release_third", "year_2",
            "three_time_point", "slope_count", "completion_year",
            "completion_wave", "female_poa"]
    return df[keep]


def load_rand():
    """RAND columns, with the tagged missing codes of rassagem kept."""
    base = ["hhid", "pn", "hhidpn", "rassrecv", "rassagem", "rabyear",
            "rabmonth", "rabflag", "radyear", "radmonth", "radage_m",
            "raedyrs", "ragender", "raracem", "rahispan"]
    interviews = ([f"r{w}iwendy" for w in INTERVIEW_YEAR_WAVES]
                  + [f"r{w}iwendm" for w in INTERVIEW_MONTH_WAVES])
    receipt = [f"r{w}{s}" for w in RECEIPT_WAVES for s in ("isret", "isdi")]
    expectations = [f"r{w}{s}" for w in EXPECTATION_WAVES
                    for s in ("liv10", "liv10a", "liv10p", "liv10r",
                              "liv75", "liv75p")]
    health = [f"r{w}{s}" for w in EXPECTATION_WAVES for s in ("shlt", "mstat")]
    wealth = [f"h{w}{s}" for w in EXPECTATION_WAVES
              for s in ("achck", "acd", "astck", "abond")]

    cols = base + interviews + receipt + expectations + health + wealth
    df, _ = pyreadstat.read_dta(str(RAND_HRS_DTA), usecols=cols,
                                user_missing=True)

    # Read the claim-status codes before numeric coercion: .s is never
    # started, .m and .t are known beneficiaries whose start date is
    # unavailable.
    tag = _tags(df["rassagem"])
    val = _num(df["rassagem"])
    df["claim_status"] = np.select(
        [val.notna(), tag == "s", tag == "m", tag == "t"],
        ["dated", "never", "m", "t"], default="unknown")

    for c in cols:
        if c not in ("hhid", "pn"):
            df[c] = _num(df[c])
    df["hhid"] = df["hhid"].astype(str).str.strip()
    df["hhidpn"] = df["hhidpn"].astype("int64")
    return df


def build_person(rand, poa):
    """Merge, date the timeline in months, and assign the Table A1 dispositions."""
    df = rand.merge(poa, on="hhidpn", how="inner")
    assert len(df) == N_RELEASE, f"merged N = {len(df)}"
    assert df["claim_status"].isin(["dated", "never", "m", "t"]).all()
    assert (df.loc[df["claim_status"].isin(["dated", "m", "t"]),
                   "rassrecv"] == 1).all()
    assert (df.loc[df["claim_status"] == "never", "rassrecv"] == 0).all()

    # A missing source value stays missing, so the respondent falls in the
    # missing-covariates row instead of the reference category.
    df["female"] = np.where(df["ragender"].isna(), np.nan, df["ragender"] == 2)
    df["black"] = np.where(df["raracem"].isna(), np.nan, df["raracem"] == 2)
    df["otherrace"] = np.where(df["raracem"].isna(), np.nan, df["raracem"] == 3)
    df["hisp"] = np.where(df["rahispan"].isna(), np.nan, df["rahispan"] >= 1)
    assert (df["female"] == df["female_poa"]).all(), "sex differs between the two files"
    df["cohort5"] = (df["rabyear"] // COHORT_BIN_YEARS) * COHORT_BIN_YEARS

    assert df["rabyear"].notna().all() and df["rabmonth"].notna().all()
    assert (df["rabflag"] == 0).all(), "imputed birth month"
    df["birth_month_idx"] = (df["rabyear"] * 12 + df["rabmonth"] - 1).astype(int)
    df["m62"] = df["birth_month_idx"] + AGE62_MONTHS
    df["m70"] = df["birth_month_idx"] + AGE70_MONTHS

    df["claim_month_idx"] = df["birth_month_idx"] + df["rassagem"]
    df["claim_cal_year"] = df["claim_month_idx"] // 12

    df["death_month_idx"] = np.where(
        df["radyear"].notna(),
        df["radyear"] * 12 + df["radmonth"].fillna(DEATH_MONTH_FALLBACK) - 1,
        np.nan)
    # The month arithmetic must agree with RAND's own age at death in months.
    chk = df["radyear"].notna() & df["radmonth"].notna() & df["radage_m"].notna()
    assert (df.loc[chk, "radage_m"]
            == df.loc[chk, "death_month_idx"] - df.loc[chk, "birth_month_idx"]
            ).all(), "radage_m disagrees with the month arithmetic"
    assert not (df["claim_month_idx"] > df["death_month_idx"]).any(), \
        "claim recorded after death"

    # Completion month: the end date of the completion-wave interview, or
    # December of the completion year where that date is missing.
    cw_y = pd.Series(np.nan, index=df.index)
    cw_m = pd.Series(np.nan, index=df.index)
    for w in sorted(set(COMPLETION_WAVE.values())):
        m = df["completion_wave"] == w
        cw_y[m] = df.loc[m, f"r{w}iwendy"]
        cw_m[m] = df.loc[m, f"r{w}iwendm"]
    fallback = cw_y.isna() | cw_m.isna()
    df["completion_month_idx"] = np.where(
        fallback,
        df["completion_year"] * 12 + 11,
        cw_y * 12 + cw_m - 1).astype(int)

    df["entry_month_idx"] = np.maximum(df["m62"],
                                       df["completion_month_idx"] + 1)

    # Last observation: the latest core-interview end month in waves 10-16.
    obs = []
    for w in FOLLOWUP_WAVES:
        y, mo = df[f"r{w}iwendy"], df[f"r{w}iwendm"]
        obs.append(np.where(y.notna(), y * 12 + mo.fillna(12) - 1, np.nan))
    df["last_obs_month_idx"] = pd.DataFrame(np.column_stack(obs)).max(axis=1).values
    assert (df["last_obs_month_idx"] >= df["completion_month_idx"]).all(), \
        "completion-wave interview missing from the follow-up waves"

    # Dispositions, assigned in the order of Appendix Table A1.
    miss_cov = df[DEMOGRAPHICS].isna().any(axis=1)
    no_window = df["entry_month_idx"] >= df["m70"]
    undated = df["claim_status"].isin(["m", "t"])
    pre_entry_claim = df["claim_month_idx"] < df["entry_month_idx"]
    death_pre_entry = df["death_month_idx"] < df["entry_month_idx"]
    alive = df["radyear"].isna()
    post_entry_claim = (df["claim_status"] == "dated") & ~pre_entry_claim
    no_obs = (alive & ~post_entry_claim
              & (df["last_obs_month_idx"] < df["entry_month_idx"]))

    df["disposition"] = np.select(
        [miss_cov, no_window, undated, pre_entry_claim.fillna(False),
         death_pre_entry.fillna(False), no_obs],
        EXCLUSIONS, default=CLAIMING_SAMPLE)

    # Pace of Aging in SD units of the claiming sample, and the claiming
    # sample's quartiles.
    in_sample = df["disposition"] == CLAIMING_SAMPLE
    sample_mean = float(df.loc[in_sample, POA_VAR].mean())
    sample_sd = float(df.loc[in_sample, POA_VAR].std())
    df["poa_z"] = (df[POA_VAR] - sample_mean) / sample_sd
    q1, q2, q3 = df.loc[in_sample, "poa_z"].quantile([0.25, 0.5, 0.75])
    df["poa_quartile"] = np.select(
        [df["poa_z"] <= q1, df["poa_z"] <= q2, df["poa_z"] <= q3],
        ["Q1", "Q2", "Q3"], "Q4")

    # Exit month and outcome within the claiming sample.
    claim_in_window = in_sample & (df["claim_status"] == "dated") & \
        (df["claim_month_idx"] <= df["m70"])
    claim_after_70 = in_sample & (df["claim_status"] == "dated") & \
        (df["claim_month_idx"] > df["m70"])
    died = df["death_month_idx"].notna()
    death_before_claim = in_sample & ~claim_in_window & ~claim_after_70 & died & \
        (df["death_month_idx"] <= df["m70"])
    unclaimed_at_70 = in_sample & ~claim_in_window & ~death_before_claim & (
        claim_after_70
        | (died & (df["death_month_idx"] > df["m70"]))
        | (~died & (df["last_obs_month_idx"] >= df["m70"])))
    censored = in_sample & ~claim_in_window & ~death_before_claim & ~unclaimed_at_70

    df["outcome"] = np.select(
        [claim_in_window, death_before_claim, censored, unclaimed_at_70],
        OUTCOMES, default="")
    df["exit_month_idx"] = np.select(
        [claim_in_window, death_before_claim, unclaimed_at_70, censored],
        [df["claim_month_idx"], df["death_month_idx"], df["m70"],
         df["last_obs_month_idx"]], default=np.nan)
    df["event_claim"] = claim_in_window.astype(int)
    df["event_death"] = death_before_claim.astype(int)

    # Alternative exit that censors at the last core interview (Appendix Table A3).
    df["exit_at_last_interview_idx"] = np.where(
        in_sample, np.minimum(df["exit_month_idx"], df["last_obs_month_idx"]),
        np.nan)

    e = df.loc[in_sample]
    assert len(e) == N_CLAIMING_SAMPLE
    assert int(e["event_claim"].sum()) == N_CLAIMS
    assert int(e["event_death"].sum()) == N_DEATHS_BEFORE_CLAIM
    assert (e["exit_month_idx"] >= e["entry_month_idx"]).all()
    assert (e["entry_month_idx"] < e["m70"]).all()
    assert e[DEMOGRAPHICS].notna().all().all()
    assert df.loc[claim_in_window, "rassagem"].between(
        AGE62_MONTHS, AGE70_MONTHS).all(), "claim outside the window"
    d = df.loc[claim_in_window & died]
    assert (d["claim_month_idx"] <= d["death_month_idx"]).all()
    return df


def attach_completion_wave_covariates(df):
    """Self-rated health, partnership, and liquid wealth at the completion wave."""
    for stem, out in [("shlt", "shlt_cw"), ("mstat", "mstat_cw")]:
        col = pd.Series(np.nan, index=df.index)
        for w in sorted(set(COMPLETION_WAVE.values())):
            m = df["completion_wave"] == w
            col[m] = df.loc[m, f"r{w}{stem}"]
        df[out] = col
    liquid = pd.Series(np.nan, index=df.index)
    for w in sorted(set(COMPLETION_WAVE.values())):
        m = df["completion_wave"] == w
        liquid[m] = (df.loc[m, f"h{w}achck"] + df.loc[m, f"h{w}acd"]
                     + df.loc[m, f"h{w}astck"] + df.loc[m, f"h{w}abond"])
    df["liquid_wealth_cw"] = liquid
    df["married_partnered_cw"] = np.where(
        df["mstat_cw"].isna(), np.nan, df["mstat_cw"].isin([1, 2, 3]))
    return df


def attach_expectations(df):
    """The reported survival chance and self-rated health from the latest
    interview between the close of the measurement window and entry.

    A qualifying interview falls in [completion month, entry month] and has a
    non-missing reported chance (RAND liv10, with its target age liv10a, its
    life-table probability liv10p, and their ratio liv10r). Because the
    interview may fall in the entry month itself, follow-up in the models
    that use the report starts at the later of entry and the month after the
    interview.
    """
    in_sample = df["disposition"] == CLAIMING_SAMPLE
    iw_idx = {w: df[f"r{w}iwendy"] * 12 + df[f"r{w}iwendm"] - 1
              for w in EXPECTATION_WAVES}
    valued = pd.DataFrame({
        w: (iw_idx[w] >= df["completion_month_idx"])
           & (iw_idx[w] <= df["entry_month_idx"])
           & df[f"r{w}liv10r"].notna()
        for w in EXPECTATION_WAVES})
    df["report_wave"] = np.nan
    for w in EXPECTATION_WAVES:     # ascending, so later waves overwrite earlier ones
        df.loc[valued[w].fillna(False), "report_wave"] = w

    for stem, out in [("liv10", "reported_chance"), ("liv10a", "target_age"),
                      ("liv10p", "life_table_chance"), ("liv10r", "chance_ratio"),
                      ("liv75", "reported_chance75"),
                      ("liv75p", "life_table_chance75"), ("shlt", "srh")]:
        col = pd.Series(np.nan, index=df.index)
        for w in EXPECTATION_WAVES:
            m = df["report_wave"] == w
            col[m] = df.loc[m, f"r{w}{stem}"]
        df[out] = col
    # Subjective survival gap: the reported percent chance of living to the
    # target age minus the life-table chance, in percentage points. The
    # live-to-75 question, asked under 65, gets the same construction.
    df["gap"] = df["reported_chance"] - df["life_table_chance"]
    df["gap75"] = df["reported_chance75"] - df["life_table_chance75"]

    report_idx = pd.Series(np.nan, index=df.index)
    for w in EXPECTATION_WAVES:
        m = df["report_wave"] == w
        report_idx[m] = iw_idx[w][m]
    df["report_month_idx"] = report_idx
    df["gap_start_month_idx"] = np.maximum(df["entry_month_idx"], report_idx + 1)
    df["has_report"] = in_sample & df["report_wave"].notna()

    ok = df["has_report"]
    assert df.loc[ok, "chance_ratio"].notna().all()
    assert (df.loc[ok, "report_month_idx"] <= df.loc[ok, "entry_month_idx"]).all()
    assert (df.loc[ok, "report_month_idx"]
            >= df.loc[ok, "completion_month_idx"]).all()
    return df


def month_rows(hhidpn, birth, entry, exit_, m70, claim, death):
    """One row per person-month from entry through exit, inclusive.

    Ages 62 to 69 are years of age by month of age, and the 70th-birthday
    month is its own interval. Events sit on the exit month's row. Each row
    is one month of exposure, so a Bernoulli complementary log-log fit on
    these rows is the discrete-time monthly hazard likelihood.
    """
    hhidpn, birth, entry, exit_, m70 = (
        np.asarray(v) for v in (hhidpn, birth, entry, exit_, m70))
    claim, death = np.asarray(claim, dtype=bool), np.asarray(death, dtype=bool)
    n = (exit_ - entry + 1).astype(int)
    assert (n >= 1).all(), "exit precedes entry"
    idx = np.repeat(np.arange(len(hhidpn)), n)
    offset = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    month = entry[idx] + offset
    age = np.where(month >= m70[idx], AGE70_LABEL, (month - birth[idx]) // 12)
    last = month == exit_[idx]
    return pd.DataFrame({
        "hhidpn": hhidpn[idx], "month_idx": month.astype(int),
        "age_year": age.astype(int),
        "event_claim": (claim[idx] & last).astype(int),
        "event_death": (death[idx] & last).astype(int),
    })


def build_panel(df):
    """Person-month panel of the claiming sample."""
    e = df.loc[df["disposition"] == CLAIMING_SAMPLE]
    panel = month_rows(e["hhidpn"], e["birth_month_idx"], e["entry_month_idx"],
                       e["exit_month_idx"], e["m70"], e["event_claim"],
                       e["event_death"])
    assert set(panel["age_year"].unique()) <= set(AGES) | {AGE70_LABEL}
    covs = ["hhidpn", "hhid", "poa_z", "rabyear", "cohort5",
            "completion_wave"] + DEMOGRAPHICS
    panel = panel.merge(df[covs], on="hhidpn", how="left")

    months = panel.groupby("hhidpn").size()
    span = (e.set_index("hhidpn")["exit_month_idx"]
            - e.set_index("hhidpn")["entry_month_idx"] + 1)
    assert (months.sort_index() == span.sort_index()).all(), \
        "panel rows do not match the entry-to-exit spans"
    assert panel["event_claim"].sum() == N_CLAIMS
    assert panel["event_death"].sum() == N_DEATHS_BEFORE_CLAIM
    assert panel[covs[1:]].notna().all().all()
    return panel


def cohort_adjusted_mean(sub, release_shares):
    """Mean Pace of Aging directly standardized to the release's five-year
    birth-cohort shares, over the cohorts the group contains."""
    cell = sub.groupby("cohort5", observed=True)["poa_z"].mean()
    w = release_shares.reindex(cell.index)
    return float((w * cell).sum() / w.sum())


def flow_table(df):
    """Appendix Table A1: counts and mean Pace of Aging by disposition."""
    shares = df["cohort5"].value_counts(normalize=True).sort_index()
    rows = []

    def add(label, sub):
        rows.append({
            "disposition": label, "n": len(sub),
            "pct_of_release": round(100 * len(sub) / len(df), 1),
            "poa_mean": round(float(sub["poa_z"].mean()), 3),
            "poa_cohort_adjusted_mean": round(cohort_adjusted_mean(sub, shares), 3)})

    add(RELEASE_ROW, df)
    for label in EXCLUSIONS + [CLAIMING_SAMPLE]:
        add(label, df.loc[df["disposition"] == label])
    for label in OUTCOMES:
        add(label, df.loc[df["outcome"] == label])
    out = pd.DataFrame(rows).set_index("disposition")
    assert out.loc[RELEASE_ROW, "n"] == N_RELEASE
    assert out["n"].drop(RELEASE_ROW).to_dict() == FLOW_COUNTS
    assert out.loc[EXCLUSIONS + [CLAIMING_SAMPLE], "n"].sum() == N_RELEASE, \
        "dispositions do not sum to the release"
    assert out.loc[OUTCOMES, "n"].sum() == out.loc[CLAIMING_SAMPLE, "n"], \
        "outcomes do not sum to the claiming sample"
    return out.reset_index()


def start_date_bounds(df):
    """Bounds on the start month of beneficiaries whose start date is
    unavailable, from the wave-by-wave income reports.

    A wave reports OASDI receipt when rXisret > 0 or rXisdi > 0. RAND income
    refers to the calendar year before the interview, so the first wave
    reporting receipt puts the start at or before December of that year, and
    the last earlier wave reporting none puts it at or after January of the
    year after its own. ``in_range`` marks respondents who pass every screen
    that does not depend on the unknown start date.
    """
    b = df.loc[df["disposition"] == "Known beneficiary, start date unavailable"].copy()

    receiving, not_receiving = {}, {}
    for w in RECEIPT_WAVES:
        isret, isdi = df[f"r{w}isret"], df[f"r{w}isdi"]
        receiving[w] = ((isret > 0) | (isdi > 0)).reindex(b.index)
        not_receiving[w] = ((isret == 0) & (isdi == 0)).reindex(b.index)

    first_recv = pd.Series(np.nan, index=b.index)
    for w in sorted(RECEIPT_WAVES, reverse=True):
        first_recv[receiving[w].fillna(False)] = w
    last_norecv = pd.Series(np.nan, index=b.index)
    for w in sorted(RECEIPT_WAVES):
        ok = not_receiving[w].fillna(False) & (first_recv.isna() | (w < first_recv))
        last_norecv[ok] = w

    def income_year(waves):
        out = pd.Series(np.nan, index=b.index)
        for w in RECEIPT_WAVES:
            m = waves == w
            out[m] = df[f"r{w}iwendy"].reindex(b.index)[m] - 1
        return out

    b["bound_lo_idx"] = (income_year(last_norecv) + 1) * 12
    b["bound_hi_idx"] = income_year(first_recv) * 12 + 11

    died = b["death_month_idx"].notna()
    b["in_range"] = (
        (b["completion_month_idx"] < b["m70"])
        & (~died | (b["death_month_idx"] >= b["entry_month_idx"]))
        & (died | (b["last_obs_month_idx"] >= b["entry_month_idx"])))
    return b[["hhidpn", "in_range", "bound_lo_idx", "bound_hi_idx"]]


def sample_counts(df, panel):
    """Counts, percentiles, and SD ratios quoted in the text and table notes."""
    e = df.loc[df["disposition"] == CLAIMING_SAMPLE]
    z = e["poa_z"]
    release_sd = float(e["poa_z_release"].std())
    quartile_mean = e.groupby("poa_quartile")["poa_z"].mean()
    top = df.loc[df["release_third"] == "highest"]
    top_counts = (len(top), int((top["completion_month_idx"] >= top["m70"]).sum()),
                  int((top["disposition"] == CLAIMING_SAMPLE).sum()))
    assert e["hhid"].nunique() == N_HOUSEHOLDS
    assert top_counts == RELEASE_HIGHEST_THIRD
    rows = [
        ("households", int(e["hhid"].nunique())),
        ("person_years", round(len(panel) / 12, 1)),
        ("poa_p25", round(float(z.quantile(0.25)), 3)),
        ("poa_p75", round(float(z.quantile(0.75)), 3)),
        ("poa_iqr", round(float(z.quantile(0.75) - z.quantile(0.25)), 3)),
        # One SD of the claiming sample in SD units of the full release, and
        # the reverse.
        ("claiming_sample_sd_in_release_sd", round(release_sd, 3)),
        ("release_sd_in_claiming_sample_sd", round(1 / release_sd, 3)),
        ("poa_mean_highest_minus_lowest_quarter",
         round(float(quartile_mean["Q4"] - quartile_mean["Q1"]), 3)),
        ("release_highest_third", top_counts[0]),
        ("release_highest_third_complete_at_or_after_70", top_counts[1]),
        ("release_highest_third_in_claiming_sample", top_counts[2]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main():
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = build_person(load_rand(), load_poa())
    df = attach_completion_wave_covariates(df)
    df = attach_expectations(df)
    panel = build_panel(df)

    person_cols = [
        "hhidpn", "hhid", "poa_z", "poa_quartile", "year_2",
        "three_time_point", "slope_count", "completion_wave",
        "completion_month_idx", "rabyear", "cohort5", "birth_month_idx",
        "m62", "m70"] + DEMOGRAPHICS + [
        "claim_status", "rassagem", "claim_month_idx", "claim_cal_year",
        "radyear", "death_month_idx", "entry_month_idx", "last_obs_month_idx",
        "exit_month_idx", "exit_at_last_interview_idx", "disposition",
        "outcome", "event_claim", "event_death", "shlt_cw",
        "married_partnered_cw", "liquid_wealth_cw", "has_report",
        "report_month_idx", "gap_start_month_idx", "reported_chance",
        "target_age", "life_table_chance", "chance_ratio", "gap", "gap75", "srh",
    ]
    df[person_cols].to_parquet(PERSON_PARQUET, index=False)
    panel.to_parquet(PANEL_PARQUET, index=False)
    start_date_bounds(df).to_parquet(START_BOUNDS_PARQUET, index=False)
    flow_table(df).to_csv(TABLES_DIR / "sample_flow.csv", index=False)
    sample_counts(df, panel).to_csv(TABLES_DIR / "sample_counts.csv", index=False)


if __name__ == "__main__":
    main()
