"""
Subjective survival: what it reflects and what it predicts (Table 3, panels A
and B; claiming rows of Appendix Table A4).

Sample: respondents in the claiming sample with a reported survival chance
between the close of the measurement window and entry, and follow-up
remaining at the later of entry and the month after that interview. The
measure is the subjective survival gap: the reported percent chance of
living to the target age (85 for respondents interviewed under 65, 80 at 65
to 69) minus the life-table chance, entered per 10 points. Every model
controls for the target age.

Panel A: linear regression of the gap on Pace of Aging (demographics, birth
cohort, completion wave, target age, age at the interview), then with
self-rated health from the same interview.

Panel B: the Section 2 claiming hazard with the gap, with the gap and Pace
of Aging, and with self-rated health added, and the standardized share
claimed by 65 at the 25th and 75th percentiles of the gap and of Pace of
Aging (household bootstrap, refitting every hazard in each replicate).

Appendix Table A4: the claiming hazard with the ratio of the reported to the
life-table chance (winsorized, standardized), the reported chance itself,
the gap from the live-to-75 question, categories of the reported chance,
and month-of-age indicators added to the baseline hazard.

Writes:
- output/tables/subjective_survival.csv   coefficients and sample summaries
- output/tables/gap_claiming_shares.csv   shares under Table 3, panel B
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from claiming import (
    COHORT_LEVELS, COHORT_MERGE, COMPLETION_WAVES, Standardization,
    claim_design, death_design, expand_panel, fit_cloglog,
)
from config import (
    CLAIMING_SAMPLE, CLUSTER_VAR, DEMOGRAPHICS, EXPECTATIONS_COUNTS, N_BOOT,
    PERSON_PARQUET, RANDOM_SEED, TABLES_DIR,
)

WINSOR_PCTS = (0.01, 0.99)
GAP_UNIT = 10.0          # the gap enters per 10 points
FOCAL_LEVELS = ["p0", "p_0_50", "p50", "p_50_100", "p100"]
FOCAL_REF = "p50"
START = "gap_start_month_idx"
SHARE_MODELS = {"gap_only": ["gap_10"], "gap_and_poa": ["gap_10", "poa_z"]}
SHARE_ESTIMANDS = ("level_at_p25", "level_at_p75", "p25_p75_contrast")
CONTRAST = "p25_p75_contrast"


def focal_code(v):
    if v == 0:
        return "p0"
    if v == 50:
        return "p50"
    if v == 100:
        return "p100"
    return "p_0_50" if v < 50 else "p_50_100"


def target_dummies(df, X):
    """Add indicators for the target ages that occur (reference: 85) to ``X``
    and return their names."""
    levels = sorted(int(t) for t in df["target_age"].dropna().unique()
                    if t != 85)
    for t in levels:
        X[f"target_is_{t}"] = (df["target_age"] == t).astype(float)
    return [f"target_is_{t}" for t in levels]


def load_sample(person=None):
    """Respondents in the claiming sample with reported survival expectations."""
    p = pd.read_parquet(PERSON_PARQUET) if person is None else person
    reported = p.loc[(p["disposition"] == CLAIMING_SAMPLE) & p["has_report"]].copy()
    lo, hi = reported["chance_ratio"].quantile(WINSOR_PCTS)
    s = reported.loc[reported[START] <= reported["exit_month_idx"]].copy()
    s["gap_10"] = s["gap"] / GAP_UNIT
    s["gap75_10"] = s["gap75"] / GAP_UNIT
    # Ratio of the reported to the life-table chance, winsorized at the 1st
    # and 99th percentiles of all respondents with a report and standardized
    # within the sample.
    wins = s["chance_ratio"].clip(lo, hi)
    s["ratio_z"] = (wins - wins.mean()) / wins.std()
    s["report_age"] = (s["report_month_idx"] - s["birth_month_idx"]) / 12
    s["reported_01"] = s["reported_chance"] / 100
    s["focal"] = s["reported_chance"].map(focal_code)
    s["hh_code"] = pd.factorize(s[CLUSTER_VAR])[0]
    return s


def gap_design(sample, extra_cols=()):
    """Design of the Table 3 panel A regressions."""
    X = pd.DataFrame(index=sample.index)
    X["const"] = 1.0
    X["poa_z"] = sample["poa_z"].astype(float)
    target_dummies(sample, X)
    X["report_age"] = sample["report_age"].astype(float)
    for v in DEMOGRAPHICS:
        X[v] = sample[v].astype(float)
    for w in COMPLETION_WAVES[1:]:
        X[f"cw_{w}"] = (sample["completion_wave"] == w).astype(float)
    coh = sample["cohort5"].replace(COHORT_MERGE)
    for c in COHORT_LEVELS[1:]:
        X[f"cohort_{c}"] = (coh == c).astype(float)
    for col in extra_cols:
        X[col] = sample[col].astype(float)
    return X


def coefficient_rows(model, res, cols, terms, n, n_events=np.nan):
    ci = np.asarray(res.conf_int())
    return [{"model": model, "term": t, "estimate": res.params[cols.index(t)],
             "ci_lo": ci[cols.index(t)][0], "ci_hi": ci[cols.index(t)][1],
             "p": res.pvalues[cols.index(t)], "n": n, "n_events": n_events}
            for t in terms]


def claim_panel(sample):
    """Person-month panel with entry at the start of follow-up for the report."""
    panel = expand_panel(sample, entry_col=START)
    add = sample[["hhidpn", "gap_10", "ratio_z", "reported_01", "srh",
                  "gap75_10", "target_age", "focal"]]
    panel = panel.merge(add, on="hhidpn", how="left")
    panel["hh_code"] = pd.factorize(panel[CLUSTER_VAR])[0]
    for lev in FOCAL_LEVELS:
        panel[f"focal_{lev}"] = (panel["focal"] == lev).astype(float)
    return panel


def claim_design_with(panel, added_cols, month_offsets=False):
    """Section 2 design with target-age indicators and ``added_cols``; Pace of
    Aging enters only when it is listed in ``added_cols``."""
    X = claim_design(panel, month_offsets=month_offsets)
    X = X.drop(columns=["poa_z"])
    target_dummies(panel, X)
    for c in added_cols:
        X[c] = panel[c].astype(float)
    return X


def claim_model(panel, added_cols, month_offsets=False):
    X = claim_design_with(panel, added_cols, month_offsets)
    res = fit_cloglog(panel["event_claim"], X, groups=panel["hh_code"])
    assert res is not None, "claiming fit failed"
    return res, list(X.columns)


def share_tools(sample, panel):
    """Fitting and standardization for the shares under Table 3, panel B.

    Returns ``fit(idx)``, which fits each claiming hazard in SHARE_MODELS and
    the death hazard on the panel rows ``idx``, and ``shares(beta)``, which
    turns the fitted coefficients into the standardized shares claimed by 65
    with each variable set to its own 25th and 75th percentile.
    """
    quantiles = {
        "gap_10": tuple(sample["gap_10"].quantile([0.25, 0.75])),
        "poa_z": tuple(sample["poa_z"].quantile([0.25, 0.75])),
    }
    designs = {m: claim_design_with(panel, cols)
               for m, cols in SHARE_MODELS.items()}
    Xd = death_design(panel)
    yc = panel["event_claim"].to_numpy(float)
    yd = panel["event_death"].to_numpy(float)
    p = sample.copy()
    tcols = target_dummies(p, p)
    stds = {(m, shift): Standardization(p, shift_col=shift,
                                        extra_cols=tcols + cols,
                                        drop_poa=True, entry_col=START)
            for m, cols in SHARE_MODELS.items() for shift in cols}
    for (m, _), std in stds.items():
        assert std.claim_cols == list(designs[m].columns), m

    def fit(idx, starts=None):
        out = {}
        for m, X in designs.items():
            r = fit_cloglog(yc[idx], X.to_numpy(float)[idx],
                            start=None if starts is None else starts[m])
            if r is None:
                return None
            out[m] = r.params
        rd = fit_cloglog(yd[idx], Xd.to_numpy(float)[idx],
                         start=None if starts is None else starts["death"])
        if rd is None:
            return None
        out["death"] = rd.params
        return out

    def shares(beta):
        out = {}
        for (m, shift), std in stds.items():
            est = std.estimands(beta[m], beta["death"], *quantiles[shift])[65]
            for k in SHARE_ESTIMANDS:
                out[f"{m}|{shift}|{k}"] = est[k]
        return out

    return fit, shares


def share_table(sample, panel):
    fit, shares = share_tools(sample, panel)
    point_beta = fit(np.arange(len(panel)))
    assert point_beta is not None, "point fits failed"
    point = shares(point_beta)

    rng = np.random.default_rng(RANDOM_SEED)
    hh_codes = sample["hh_code"].unique()
    by_hh = dict(panel.groupby("hh_code").indices)
    reps = []
    for _ in range(N_BOOT):
        draw = rng.choice(hh_codes, size=len(hh_codes), replace=True)
        idx = np.concatenate([by_hh[h] for h in draw])
        beta = fit(idx, starts=point_beta)
        if beta is not None:
            reps.append(shares(beta))
    reps = pd.DataFrame(reps)
    contrasts = reps[[c for c in reps.columns if c.endswith(CONTRAST)]]
    assert len(reps) == N_BOOT and np.isfinite(contrasts.to_numpy()).all(), \
        "a bootstrap replicate failed"

    rows = []
    for m, cols in SHARE_MODELS.items():
        for shift in cols:
            for k in SHARE_ESTIMANDS:
                row = {"model": m, "shift": shift, "target_age": 65,
                       "estimand": k, "estimate": point[f"{m}|{shift}|{k}"]}
                if k == CONTRAST:
                    q = reps[f"{m}|{shift}|{k}"].quantile([0.025, 0.975])
                    row.update({"boot_ci95_lo": q.loc[0.025],
                                "boot_ci95_hi": q.loc[0.975],
                                "n_boot_used": len(reps)})
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_sample()
    n, n_claims = len(sample), int(sample["event_claim"].sum())
    assert (n, n_claims) == EXPECTATIONS_COUNTS["all"]
    g = sample["gap"]
    rows = [{"model": "sample", "term": term, "estimate": float(value)}
            for term, value in [
                ("n", n), ("claims", n_claims), ("gap_mean", g.mean()),
                ("gap_sd", g.std()), ("gap_p25", g.quantile(0.25)),
                ("gap_p75", g.quantile(0.75)),
                ("life_table_chance_mean", sample["life_table_chance"].mean()),
                ("reported_chance_mean", sample["reported_chance"].mean())]]

    # Table 3, panel A.
    counts = pd.read_csv(TABLES_DIR / "sample_counts.csv").set_index("metric")["value"]
    for model, extra in [("gap_on_poa", ()), ("gap_on_poa_srh", ("srh",))]:
        sub = sample.dropna(subset=list(extra)) if extra else sample
        X = gap_design(sub, extra_cols=extra)
        res = sm.OLS(sub["gap"].to_numpy(float), X.to_numpy(float)).fit(
            cov_type="cluster", cov_kwds={"groups": sub["hh_code"].to_numpy()})
        assert len(sub) == EXPECTATIONS_COUNTS[
            "with_self_rated_health" if extra else "all"][0]
        rows += coefficient_rows(model, res, list(X.columns),
                                 ["poa_z"] + list(extra), len(sub))
        if model == "gap_on_poa":
            # Difference in the gap across the interquartile range of Pace of Aging.
            b = res.params[list(X.columns).index("poa_z")]
            rows.append({"model": model, "term": "difference_across_poa_iqr",
                         "estimate": abs(b) * float(counts["poa_iqr"])})

    # Table 3, panel B, and the claiming rows of Appendix Table A4.
    panel = claim_panel(sample)
    panel_75 = panel.dropna(subset=["gap75_10"])
    panel_srh = panel.dropna(subset=["srh"]).copy()
    panel_srh_moa = panel_srh.merge(sample[["hhidpn", "birth_month_idx"]],
                                    on="hhidpn", how="left")
    panel_srh_moa["month_offset"] = (
        (panel_srh_moa["month_idx"] - panel_srh_moa["birth_month_idx"]) % 12).astype(int)
    focal_terms = [f"focal_{v}" for v in FOCAL_LEVELS if v != FOCAL_REF]
    all_three = ["gap_10", "srh", "poa_z"]
    # (model, panel, sample, terms in the model, terms reported, month of age)
    for model, pan, counts, terms, reported, month_offsets in [
            ("claiming_on_gap", panel, "all", ["gap_10"], ["gap_10"], False),
            ("claiming_on_gap_poa", panel, "all", ["gap_10", "poa_z"],
             ["gap_10", "poa_z"], False),
            ("claiming_on_gap_srh_poa", panel_srh, "with_self_rated_health",
             all_three, all_three, False),
            ("claiming_on_ratio", panel, "all", ["ratio_z"], ["ratio_z"], False),
            ("claiming_on_reported_chance", panel, "all", ["reported_01"],
             ["reported_01"], False),
            ("claiming_on_gap75", panel_75, "live_to_75", ["gap75_10"],
             ["gap75_10"], False),
            ("claiming_on_focal_answers", panel, "all", focal_terms, focal_terms,
             False),
            ("claiming_on_gap_srh_poa_month_of_age", panel_srh_moa,
             "with_self_rated_health", all_three, ["gap_10", "poa_z"], True)]:
        n_model = (int(pan["hhidpn"].nunique()), int(pan["event_claim"].sum()))
        assert n_model == EXPECTATIONS_COUNTS[counts], model
        res, cols = claim_model(pan, terms, month_offsets)
        rows += coefficient_rows(model, res, cols, reported, *n_model)
        if model == "claiming_on_gap":
            # Percent difference in the monthly hazard of claiming per 10 points.
            b = res.params[cols.index("gap_10")]
            rows.append({"model": model, "term": "pct_lower_hazard_per_10_points",
                         "estimate": (1 - np.exp(b)) * 100})

    pd.DataFrame(rows).to_csv(TABLES_DIR / "subjective_survival.csv",
                              index=False, float_format="%.6f")
    share_table(sample, panel).to_csv(TABLES_DIR / "gap_claiming_shares.csv",
                                      index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
