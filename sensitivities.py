"""
The claiming hazard under alternative specifications and samples (Appendix
Table A3).

Each hazard row is the Pace of Aging coefficient of the Section 2 claiming
hazard refit under the stated change. The two claiming-age rows are linear
regressions in years per SD; the one on all observed claimants is the
regression written by retrospective.py, rescaled to the claiming sample's SD.

Writes output/tables/sensitivities.csv.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import SpecificationWarning

from claiming import (
    AGE_LEVELS, COHORT_LEVELS, COHORT_MERGE, COMPLETION_WAVES, claim_design,
    expand_panel, fit_cloglog, load_frames,
)
from config import (
    CLUSTER_VAR, DEMOGRAPHICS, N_IMPUTATIONS, PERSON_PARQUET, RANDOM_SEED,
    SENSITIVITY_COUNTS, START_BOUNDS_PARQUET, TABLES_DIR,
)

MIN_DUMMY_EVENTS = 5   # indicators with fewer events are merged into the reference
WEIGHT_TRUNC_PCT = 0.99
COMPLETED_COHORT_MAX_BYEAR = 1952


def poa_row(specification, res, cols, n, claims):
    if specification in SENSITIVITY_COUNTS:
        assert (n, claims) == SENSITIVITY_COUNTS[specification], specification
    j = cols.index("poa_z")
    ci = np.asarray(res.conf_int())
    return {"specification": specification, "estimate": res.params[j],
            "ci_lo": ci[j][0], "ci_hi": ci[j][1], "p": res.pvalues[j],
            "n": n, "claims": claims}


def prune_design(X, y):
    """Drop columns a subset cannot estimate.

    In a subset a level can vanish (a constant column), a reference category
    can empty (indicators collinear with the intercept), or an indicator can
    hold fewer than MIN_DUMMY_EVENTS events, which lets the likelihood
    diverge. Such columns are dropped, which merges the category into the
    reference.
    """
    y = np.asarray(y, dtype=float)
    keep = [c for c in X.columns if c == "const" or X[c].nunique() > 1]
    X = X[keep]
    dummy = [c for c in X.columns
             if c != "const" and set(np.unique(X[c])) <= {0.0, 1.0}]
    sparse = [c for c in dummy
              if y[X[c].to_numpy() == 1.0].sum() < MIN_DUMMY_EVENTS]
    r = np.abs(np.diag(np.linalg.qr(X.to_numpy(float), mode="r")))
    collinear = [c for c, v in zip(X.columns, r) if v < 1e-8 * r.max()]
    bad = sorted(set(sparse) | set(collinear))
    assert "poa_z" not in bad
    return X.drop(columns=bad)


def fit_claiming(panel):
    """Section 2 claiming hazard on ``panel``, with the design pruned only
    when the full specification cannot be estimated."""
    groups = pd.factorize(panel[CLUSTER_VAR])[0]
    X = claim_design(panel)
    res = fit_cloglog(panel["event_claim"], X, groups=groups)
    if res is not None and np.abs(res.params).max() < 30:
        return res, list(X.columns)
    X = prune_design(X, panel["event_claim"])
    res = fit_cloglog(panel["event_claim"], X, groups=groups)
    assert res is not None and np.abs(res.params).max() < 30, \
        "claiming fit failed"
    return res, list(X.columns)


def claiming_age_row(person):
    """Claiming age (years) on Pace of Aging among the claiming sample's claimers."""
    c = person.loc[person["event_claim"] == 1]
    coh = c["cohort5"].replace(COHORT_MERGE)
    X = pd.DataFrame({"const": 1.0, "poa_z": c["poa_z"]}, index=c.index)
    for lv in COHORT_LEVELS[1:]:
        X[f"cohort_{lv}"] = (coh == lv).astype(float)
    for w in COMPLETION_WAVES[1:]:
        X[f"cw_{w}"] = (c["completion_wave"] == w).astype(float)
    for v in DEMOGRAPHICS:
        X[v] = c[v].astype(float)
    res = sm.OLS(np.asarray(c["rassagem"] / 12, float), X.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(c[CLUSTER_VAR])[0]})
    return poa_row("claiming_age_claiming_sample", res, list(X.columns),
                   len(c), np.nan)


def retrospective_row():
    """The retrospective regression, rescaled from the full release's SD to
    the claiming sample's SD."""
    r = pd.read_csv(TABLES_DIR / "retrospective_regression.csv").iloc[0]
    counts = pd.read_csv(TABLES_DIR / "sample_counts.csv").set_index("metric")["value"]
    k = float(counts["claiming_sample_sd_in_release_sd"])
    return {"specification": "claiming_age_all_claimants",
            "estimate": r["estimate"] * k, "ci_lo": r["ci_lo"] * k,
            "ci_hi": r["ci_hi"] * k, "p": r["p"], "n": r["n"], "claims": np.nan}


def entry_at_completion(p_all):
    """The claiming sample rebuilt with entry at the completion month."""
    entry = np.maximum(p_all["m62"], p_all["completion_month_idx"])
    covs_ok = p_all[DEMOGRAPHICS].notna().all(axis=1)
    no_window = entry >= p_all["m70"]
    undated = p_all["claim_status"].isin(["m", "t"])
    claim = p_all["claim_month_idx"]
    death = p_all["death_month_idx"]
    alive = p_all["radyear"].isna()
    pre = (claim < entry).fillna(False)
    dpre = (death < entry).fillna(False)
    post_claim = (p_all["claim_status"] == "dated") & ~pre
    noobs = alive & ~post_claim & (p_all["last_obs_month_idx"] < entry)
    in_sample = covs_ok & ~no_window & ~undated & ~pre & ~dpre & ~noobs
    e = p_all.loc[in_sample].copy()
    e["entry_alt"] = entry[in_sample]
    ciw = (e["claim_status"] == "dated") & (e["claim_month_idx"] <= e["m70"])
    c70 = (e["claim_status"] == "dated") & (e["claim_month_idx"] > e["m70"])
    died = e["death_month_idx"].notna()
    dbc = ~ciw & ~c70 & died & (e["death_month_idx"] <= e["m70"])
    surv70 = ~ciw & ~dbc & (c70 | (died & (e["death_month_idx"] > e["m70"]))
                            | (~died & (e["last_obs_month_idx"] >= e["m70"])))
    e["exit_alt"] = np.select(
        [ciw, dbc, surv70],
        [e["claim_month_idx"], e["death_month_idx"], e["m70"]],
        default=e["last_obs_month_idx"])
    e["ec_alt"] = ciw.astype(int)
    e["ed_alt"] = dbc.astype(int)
    return e


def attrition_weights(panel, person):
    """Stabilized inverse-probability-of-censoring weights, truncated at the
    99th percentile.

    Denominator: a hazard of censoring at the last observation on age
    indicators and observables fixed before entry (the claiming model's
    covariates, the inverse hyperbolic sine of liquid wealth, self-rated
    health, and partnership at the completion wave). Numerator: age
    indicators only. A month's weight is the product over the respondent's
    earlier months.
    """
    censored = person.loc[person["outcome"] == "Censored at last observation",
                          "hhidpn"]
    pan = panel.sort_values(["hhidpn", "month_idx"]).reset_index(drop=True)
    last = pan.groupby("hhidpn")["month_idx"].transform("max")
    pan["event_censor"] = ((pan["month_idx"] == last)
                           & pan["hhidpn"].isin(set(censored))).astype(int)
    extra = person.set_index("hhidpn")[
        ["liquid_wealth_cw", "shlt_cw", "married_partnered_cw"]]
    pan = pan.join(extra, on="hhidpn")
    pan["ihs_liquid"] = np.arcsinh(pan["liquid_wealth_cw"])
    ok = pan[["ihs_liquid", "shlt_cw", "married_partnered_cw"]].notna().all(axis=1)
    pan = pan.loc[ok].copy()

    Xd = claim_design(pan)
    for c in ["ihs_liquid", "shlt_cw", "married_partnered_cw"]:
        Xd[c] = pan[c].astype(float)
    res_d = fit_cloglog(pan["event_censor"], Xd)
    Xn = Xd[["const"] + [f"age_{a}" for a in AGE_LEVELS[1:]]]
    res_n = fit_cloglog(pan["event_censor"], Xn)
    assert res_d is not None and res_n is not None
    h_d = 1 - np.exp(-np.exp(Xd.to_numpy(float) @ res_d.params))
    h_n = 1 - np.exp(-np.exp(Xn.to_numpy(float) @ res_n.params))
    pan["_r"] = np.log1p(-h_n) - np.log1p(-h_d)
    pan["weight"] = np.exp(pan.groupby("hhidpn", sort=False)["_r"].cumsum() - pan["_r"])
    pan["weight"] = pan["weight"].clip(upper=pan["weight"].quantile(WEIGHT_TRUNC_PCT))
    return pan


def imputed_start_dates(person_all, primary_panel, rng):
    """Multiple imputation of unrecoverable start months.

    Each imputation draws every start month uniformly within its bounds,
    drops respondents whose draw precedes entry, adds the rest to the panel
    as claims (or as censored at 70 or death when the draw falls later), and
    refits the claiming hazard. Coefficients are combined by Rubin's rules.
    """
    b = pd.read_parquet(START_BOUNDS_PARQUET)
    b = b.loc[b["in_range"] & b["bound_lo_idx"].notna() & b["bound_hi_idx"].notna(),
              ["hhidpn", "bound_lo_idx", "bound_hi_idx"]]
    info = person_all.set_index("hhidpn")
    b = b.join(info[["birth_month_idx", "m70", CLUSTER_VAR, "entry_month_idx",
                     "death_month_idx", "poa_z", "cohort5", "completion_wave",
                     "rabyear"] + DEMOGRAPHICS], on="hhidpn")
    died = b["death_month_idx"].notna()
    cap = np.minimum(b["m70"], b["death_month_idx"].fillna(np.inf))

    coefs, variances = [], []
    for _ in range(N_IMPUTATIONS):
        u = rng.integers(b["bound_lo_idx"].to_numpy(int),
                         b["bound_hi_idx"].to_numpy(int) + 1)
        pre = u < b["entry_month_idx"]
        event = (~pre) & (u <= cap)
        post = (~pre) & (u > cap)
        add = b.loc[event | post].copy()
        au = u[(event | post).to_numpy()]
        aevent = event.loc[event | post].to_numpy()
        adied = died.loc[event | post].to_numpy()
        add["exit_mi"] = np.where(
            aevent, au,
            np.where(adied & (add["death_month_idx"] <= add["m70"]),
                     add["death_month_idx"], add["m70"]))
        add["ec_mi"] = aevent.astype(int)
        add["ed_mi"] = ((~aevent) & adied
                        & (add["death_month_idx"] <= add["m70"])).astype(int)
        add = add.loc[add["exit_mi"] >= add["entry_month_idx"]]
        extra_panel = expand_panel(add, exit_col="exit_mi",
                                   claim_flag="ec_mi", death_flag="ed_mi")
        pan = pd.concat([primary_panel[extra_panel.columns], extra_panel],
                        ignore_index=True)
        res, cols = fit_claiming(pan)
        j = cols.index("poa_z")
        coefs.append(res.params[j])
        variances.append(res.bse[j] ** 2)

    m = N_IMPUTATIONS
    qbar = np.mean(coefs)
    wbar = np.mean(variances)
    bvar = np.var(coefs, ddof=1)
    se = np.sqrt(wbar + (1 + 1 / m) * bvar)
    dfree = (m - 1) * (1 + wbar / ((1 + 1 / m) * bvar)) ** 2
    tcrit = stats.t.ppf(0.975, dfree)
    return {"specification": "start_dates_imputed", "estimate": qbar,
            "ci_lo": qbar - tcrit * se, "ci_hi": qbar + tcrit * se,
            "p": 2 * stats.t.sf(abs(qbar / se), dfree), "imputations": m}


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    person, panel = load_frames()
    person_all = pd.read_parquet(PERSON_PARQUET)
    n, claims = len(person), int(panel["event_claim"].sum())
    groups = pd.factorize(panel[CLUSTER_VAR])[0]
    rows = []

    full = claim_design(panel)
    for label, X in [
            ("age_only", full[["const"] + [f"age_{a}" for a in AGE_LEVELS[1:]]
                              + ["poa_z"]]),
            ("age_cohort_wave", full.drop(columns=DEMOGRAPHICS))]:
        res = fit_cloglog(panel["event_claim"], X, groups=groups)
        rows.append(poa_row(label, res, list(X.columns), n, claims))

    alt = entry_at_completion(person_all)
    alt_panel = expand_panel(alt, entry_col="entry_alt", exit_col="exit_alt",
                             claim_flag="ec_alt", death_flag="ed_alt")
    res, cols = fit_claiming(alt_panel)
    rows.append(poa_row("entry_in_completion_month", res, cols, len(alt),
                        int(alt_panel["event_claim"].sum())))

    last = person.copy()
    last["exit_s"] = last["exit_at_last_interview_idx"]
    last = last.loc[last["exit_s"] >= last["entry_month_idx"]].copy()
    last["ec_s"] = ((last["event_claim"] == 1)
                    & (last["claim_month_idx"] <= last["exit_s"])).astype(int)
    last["ed_s"] = 0
    last_panel = expand_panel(last, exit_col="exit_s", claim_flag="ec_s",
                              death_flag="ed_s")
    res, cols = fit_claiming(last_panel)
    rows.append(poa_row("censor_at_last_interview", res, cols, len(last),
                        int(last_panel["event_claim"].sum())))

    comp = panel.loc[panel["rabyear"] <= COMPLETED_COHORT_MAX_BYEAR]
    res, cols = fit_claiming(comp)
    rows.append(poa_row("cohorts_born_by_1952", res, cols,
                        comp["hhidpn"].nunique(), int(comp["event_claim"].sum())))

    # statsmodels warns that the cluster covariance is not fully supported
    # with variance weights.
    pan_w = attrition_weights(panel, person)
    Xw = claim_design(pan_w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpecificationWarning)
        res = sm.GLM(pan_w["event_claim"].to_numpy(float), Xw.to_numpy(float),
                     family=sm.families.Binomial(sm.families.links.CLogLog()),
                     var_weights=pan_w["weight"].to_numpy(float)).fit(
            maxiter=200, cov_type="cluster",
            cov_kwds={"groups": pd.factorize(pan_w[CLUSTER_VAR])[0]})
    assert res.converged and np.isfinite(res.params).all() and np.isfinite(res.llf)
    rows.append(poa_row("attrition_weights", res, list(Xw.columns),
                        pan_w["hhidpn"].nunique(),
                        int(pan_w["event_claim"].sum())))

    rows.append(imputed_start_dates(person_all, panel,
                                    np.random.default_rng(RANDOM_SEED)))

    # RAND dates a year-only report of first receipt to the month after the
    # birthday. Every claim is moved to the earliest, then the latest, month
    # of its calendar year that lies inside the respondent's claiming window.
    # The latest month is also capped at the last observed month, unless the
    # recorded claim date is already later.
    is_claim = person["event_claim"] == 1
    assert (person.loc[is_claim, "claim_cal_year"]
            == person.loc[is_claim, "claim_month_idx"] // 12).all()
    for label, edge in (("earliest", 0), ("latest", 11)):
        h = person.copy()
        bound = h["claim_cal_year"].fillna(0).astype(int) * 12 + edge
        if edge == 11:
            bound = np.minimum(bound, np.maximum(h["last_obs_month_idx"],
                                                 h["claim_month_idx"].fillna(0)))
        h["exit_h"] = np.where(is_claim, np.clip(bound, h["entry_month_idx"], h["m70"]),
                               h["exit_month_idx"]).astype(int)
        h["ec_h"] = is_claim.astype(int)
        h["ed_h"] = h["event_death"].astype(int)
        hp = expand_panel(h, exit_col="exit_h", claim_flag="ec_h", death_flag="ed_h")
        res, cols = fit_claiming(hp)
        rows.append(poa_row(f"claims_{label}_month", res, cols, len(h),
                            int(hp["event_claim"].sum())))

    # Eleven month-of-age indicators (birthday month as the reference) added
    # to the year-of-age indicators.
    pm = panel.merge(person[["hhidpn", "birth_month_idx"]], on="hhidpn", how="left")
    pm["month_offset"] = ((pm["month_idx"] - pm["birth_month_idx"]) % 12).astype(int)
    Xm = claim_design(pm, month_offsets=True)
    res = fit_cloglog(pm["event_claim"], Xm, groups=pm["hh_code"])
    assert res is not None, "month-of-age fit failed"
    rows.append(poa_row("month_of_age", res, list(Xm.columns), n, claims))

    quality = person.set_index("hhidpn")[["three_time_point", "slope_count"]]
    panel_q = panel.join(quality, on="hhidpn")
    for label, mask in [
            ("three_time_points", panel_q["three_time_point"] == 1),
            ("eight_or_more_slopes", panel_q["slope_count"] >= 8)]:
        sub = panel_q.loc[mask]
        res, cols = fit_claiming(sub)
        rows.append(poa_row(label, res, cols, sub["hhidpn"].nunique(),
                            int(sub["event_claim"].sum())))

    rows.append(retrospective_row())
    rows.append(claiming_age_row(person))

    pd.DataFrame(rows).to_csv(TABLES_DIR / "sensitivities.csv", index=False,
                              float_format="%.6f")


if __name__ == "__main__":
    main()
