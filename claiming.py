"""
Claiming by Pace of Aging in the claiming sample (Table 2).

Panel A: Aalen-Johansen cumulative incidence of claiming from age 62, with
death as a competing event, overall and by quartile of Pace of Aging.

Panel B: the Section 2 model. A discrete-time complementary log-log hazard
of claiming on the person-month panel (indicators for year of age, five-year
birth cohort, and completion wave; sex, years of education, race and
ethnicity; household-clustered standard errors) and a death hazard with
fewer terms (linear age, Pace of Aging, sex, birth year). The two fitted
hazards give each respondent's probability of having claimed by a given age,
accumulated from his or her own entry month, with Pace of Aging set to the
claiming sample's 25th and then 75th percentile. Averaging over the
respondents who had entered by that age gives the standardized shares and
their difference. The difference per SD is the average slope of the
accumulated probability in Pace of Aging (central difference with step
DERIV_STEP). Confidence intervals are percentile intervals from a household
bootstrap that refits both hazards in each replicate.

The interaction of Pace of Aging with age (62 to 68) and a cause-specific
Cox model feed Appendix Table A3.

Writes:
- output/tables/claiming_by_quartile.csv   Table 2, panel A
- output/tables/claiming_shares.csv        Table 2, panel B; 90% limits for Appendix Table A5
- output/tables/claiming_hazard.csv        coefficients in the note to Table 2 and in
                                           Appendix Table A3
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import DomainWarning

from benefit_schedule import fra_years
from claiming_sample import month_rows
from config import (
    CLAIMING_SAMPLE, CLUSTER_VAR, DEMOGRAPHICS, ENTERED_BY_AGE, N_BOOT,
    N_CLAIMING_SAMPLE, N_CLAIMS, N_DEATHS_BEFORE_CLAIM, PANEL_PARQUET,
    PERSON_PARQUET, QUARTILE_CLAIMING_COUNTS, RANDOM_SEED, TABLES_DIR,
)
from cox_tools import fit_cox

AGE_LEVELS = list(range(62, 71))        # 70 is the 70th-birthday month
INTERACTION_AGES = range(62, 69)
# Merge the 1940-44 and 1945-49 birth cohorts in every claiming model.
COHORT_MERGE = {1940: 1945}
COHORT_LEVELS = [1955, 1945, 1950, 1960]        # first entry is the reference
COMPLETION_WAVES = (12, 10, 11, 13)             # first entry is the reference
TARGET_AGES = [63, 65, "FRA", 70]
DERIV_STEP = 0.1                        # central-difference step, SD units
BIRTHYEAR_CENTER = 1950                 # centering for the death hazard
LEVELS = ("level_at_p25", "level_at_p75")
DIFFERENCES = ("p25_p75_contrast", "avg_derivative_per_sd")

# statsmodels warns on every fit that the complementary log-log link is not
# the canonical link of the binomial family.
warnings.filterwarnings("ignore", category=DomainWarning)


def load_frames():
    """The claiming sample (one row per respondent) and its person-month panel."""
    person = pd.read_parquet(PERSON_PARQUET)
    person = person.loc[person["disposition"] == CLAIMING_SAMPLE].copy()
    panel = pd.read_parquet(PANEL_PARQUET)
    assert len(person) == N_CLAIMING_SAMPLE
    assert panel["event_claim"].sum() == person["event_claim"].sum() == N_CLAIMS
    assert panel["event_death"].sum() == N_DEATHS_BEFORE_CLAIM
    assert sorted(person["cohort5"].unique()) == [1940, 1945, 1950, 1955, 1960]
    assert person.loc[person["cohort5"] == 1940, "event_claim"].sum() == 0
    person["hh_code"] = pd.factorize(person[CLUSTER_VAR])[0]
    panel = panel.merge(person[["hhidpn", "hh_code"]], on="hhidpn", how="left")
    return person, panel


def claim_design(df, month_offsets=False):
    """Design matrix of the Section 2 claiming hazard.

    With ``month_offsets`` the frame must carry a ``month_offset`` column
    (months since the last birthday, 0-11), and eleven month-of-age
    indicators are appended (birthday month as the reference).
    """
    coh = df["cohort5"].replace(COHORT_MERGE)
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    for a in AGE_LEVELS[1:]:
        X[f"age_{a}"] = (df["age_year"] == a).astype(float)
    for c in COHORT_LEVELS[1:]:
        X[f"cohort_{int(c)}"] = (coh == c).astype(float)
    for w in COMPLETION_WAVES[1:]:
        X[f"cw_{w}"] = (df["completion_wave"] == w).astype(float)
    for v in DEMOGRAPHICS:
        X[v] = df[v].astype(float)
    X["poa_z"] = df["poa_z"].astype(float)
    if month_offsets:
        for k in range(1, 12):
            X[f"mo_{k}"] = (df["month_offset"] == k).astype(float)
    return X


def death_design(df):
    """Design matrix of the death hazard: linear age, Pace of Aging, sex, birth year."""
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    X["age_linear"] = df["age_year"].astype(float)
    X["poa_z"] = df["poa_z"].astype(float)
    X["female"] = df["female"].astype(float)
    X["byear_c"] = df["rabyear"].astype(float) - BIRTHYEAR_CENTER
    return X


def fit_cloglog(y, X, groups=None, start=None, maxiter=200):
    """Complementary log-log binomial fit; returns None if it does not converge.

    IRLS can diverge under this link when fitted probabilities approach 1.
    It is tried first (warm-started when ``start`` is given). On failure the
    optimum is located with L-BFGS and then refit by IRLS from there.
    """
    model = sm.GLM(np.asarray(y, dtype=float), np.asarray(X, dtype=float),
                   family=sm.families.Binomial(sm.families.links.CLogLog()))
    cov = ({"cov_type": "cluster", "cov_kwds": {"groups": np.asarray(groups)}}
           if groups is not None else {})

    def _try(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                r = model.fit(maxiter=maxiter, **kw, **cov)
                ok = (r.converged and np.isfinite(r.llf)
                      and np.isfinite(r.params).all())
            except (np.linalg.LinAlgError, ValueError):
                return None
        return r if ok else None

    res = _try(start_params=start, max_start_irls=0) if start is not None else _try()
    if res is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                r_l = model.fit(method="lbfgs", maxiter=5000, disp=False)
                res = _try(start_params=r_l.params, max_start_irls=0)
            except (np.linalg.LinAlgError, ValueError):
                res = None
    return res


def expand_panel(p, entry_col="entry_month_idx", exit_col="exit_month_idx",
                 claim_flag="event_claim", death_flag="event_death"):
    """Person-month rows for other entry and exit months, built like the primary panel."""
    panel = month_rows(p["hhidpn"], p["birth_month_idx"], p[entry_col],
                       p[exit_col], p["m70"], p[claim_flag], p[death_flag])
    covs = ["hhidpn", CLUSTER_VAR, "poa_z", "cohort5", "completion_wave",
            "rabyear"] + DEMOGRAPHICS
    panel = panel.merge(p[covs], on="hhidpn", how="left")
    months = panel.groupby("hhidpn").size().sort_index()
    span = ((p.set_index("hhidpn")[exit_col] - p.set_index("hhidpn")[entry_col]
             + 1).sort_index())
    assert (months == span.loc[months.index]).all(), "panel rows do not match the spans"
    return panel


class Standardization:
    """Standardized shares who have claimed by each target age.

    Holds one cell per respondent and month from entry to the 70th-birthday
    month. ``shift_col`` names the claiming-model column whose value is set
    to a percentile (Pace of Aging in the Section 2 model, the subjective
    survival gap in the Table 3 models). ``extra_cols`` are respondent-level
    columns the claiming model adds to the Section 2 design, and ``drop_poa``
    removes the Section 2 Pace of Aging column for models that list Pace of
    Aging among the added columns or leave it out. The death hazard uses each
    respondent's own Pace of Aging unless Pace of Aging is the shifted column.
    """

    def __init__(self, person, shift_col="poa_z", extra_cols=(),
                 drop_poa=False, entry_col="entry_month_idx"):
        p = person.reset_index(drop=True)
        self.person = p
        self.shift_col = shift_col
        self.shift_is_poa = shift_col == "poa_z"
        self.poa_obs = p["poa_z"].to_numpy(float)
        entry = p[entry_col].astype(int).to_numpy()
        bmi = p["birth_month_idx"].astype(int).to_numpy()
        m70 = p["m70"].astype(int).to_numpy()
        n_months = m70 - entry + 1
        self.seg_start = np.concatenate([[0], np.cumsum(n_months)[:-1]])
        self.cell_person = np.repeat(np.arange(len(p)), n_months)
        month = entry[self.cell_person] + (
            np.arange(len(self.cell_person))
            - self.seg_start[self.cell_person])
        self.cell_age = (month - bmi[self.cell_person]) // 12
        assert self.cell_age.min() == 62 and self.cell_age.max() == 70

        Xp = claim_design(
            pd.DataFrame({"age_year": 62, "cohort5": p["cohort5"],
                          "completion_wave": p["completion_wave"],
                          "poa_z": 0.0, **{v: p[v] for v in DEMOGRAPHICS}}))
        if drop_poa:
            Xp = Xp.drop(columns=["poa_z"])
        for c in extra_cols:
            Xp[c] = p[c].astype(float).to_numpy()
        self.claim_cols = list(Xp.columns)
        self.Xp_base = Xp.drop(columns=[shift_col]).to_numpy(float)
        self.base_cols = [c for c in self.claim_cols if c != shift_col]
        self.z_obs = p[shift_col].to_numpy(float)
        self.female = p["female"].to_numpy(float)
        self.byear_c = p["rabyear"].to_numpy(float) - BIRTHYEAR_CENTER
        # Cutoff month for each target age (through the birthday month) and
        # the respondents who had entered by then.
        self.cutoffs = {}
        fra_m = np.array([int(round(fra_years(int(b)) * 12))
                          for b in p["rabyear"]])
        for a in TARGET_AGES:
            t = bmi + (fra_m if a == "FRA" else 12 * int(a))
            in_pop = entry <= t
            idx = self.seg_start + (t - entry)
            self.cutoffs[a] = (in_pop, np.where(in_pop, idx, 0))

    def _eta_claim(self, beta, z):
        b = dict(zip(self.claim_cols, beta))
        base = self.Xp_base @ np.array([b[c] for c in self.base_cols])
        age_add = np.array([0.0] + [b[f"age_{a}"] for a in AGE_LEVELS[1:]])
        return (base[self.cell_person] + b[self.shift_col] * z[self.cell_person]
                + age_add[self.cell_age - 62])

    def _eta_death(self, beta, z):
        b0, ba, bz, bf, bb = beta
        per = b0 + bz * z + bf * self.female + bb * self.byear_c
        return per[self.cell_person] + ba * self.cell_age

    def claimed_by(self, beta_claim, beta_death, z):
        """Each respondent's probability of having claimed by each target age.

        C(A) = sum over months t <= A of S(t-1) h(t), with
        S(t) = S(t-1) [1 - h(t) - d(t)], where h and d are the fitted monthly
        claiming and death probabilities and S the probability of being alive
        and unclaimed. ``z`` is the shifted column's value per respondent.
        """
        z_death = z if self.shift_is_poa else self.poa_obs
        p_c = -np.expm1(-np.exp(self._eta_claim(beta_claim, z)))
        p_d = -np.expm1(-np.exp(self._eta_death(beta_death, z_death)))
        assert np.all(p_c + p_d < 1.0), "monthly event probabilities sum to 1 or more"
        q = np.log1p(-(p_c + p_d))
        cs = np.cumsum(q)
        seg_base = np.where(self.seg_start > 0, cs[self.seg_start - 1], 0.0)
        log_s_prev = cs - seg_base[self.cell_person] - q
        contrib = np.exp(log_s_prev) * p_c
        ccs = np.cumsum(contrib)
        seg_base_c = np.where(self.seg_start > 0, ccs[self.seg_start - 1], 0.0)
        out = {}
        for a, (in_pop, idx) in self.cutoffs.items():
            out[a] = np.where(in_pop, ccs[idx] - seg_base_c, np.nan)
        return out

    def estimands(self, beta_claim, beta_death, p25, p75):
        """Shares at the two percentiles, their difference, and the average
        slope per unit of the shifted column, at each target age."""
        hi = self.claimed_by(beta_claim, beta_death, self.z_obs + DERIV_STEP / 2)
        lo = self.claimed_by(beta_claim, beta_death, self.z_obs - DERIV_STEP / 2)
        c25 = self.claimed_by(beta_claim, beta_death, np.full_like(self.z_obs, p25))
        c75 = self.claimed_by(beta_claim, beta_death, np.full_like(self.z_obs, p75))
        rows = {}
        for a in TARGET_AGES:
            pop = self.cutoffs[a][0]
            rows[a] = {
                "n_population": int(pop.sum()),
                "avg_derivative_per_sd": float(np.nanmean(
                    (hi[a][pop] - lo[a][pop]) / DERIV_STEP)),
                "p25_p75_contrast": float(np.nanmean(
                    c75[a][pop] - c25[a][pop])),
                "level_at_p25": float(np.nanmean(c25[a][pop])),
                "level_at_p75": float(np.nanmean(c75[a][pop])),
            }
        return rows


def fit_hazards(panel):
    """The Section 2 claiming hazard and the death hazard, household-clustered."""
    res_c = fit_cloglog(panel["event_claim"], claim_design(panel),
                        groups=panel["hh_code"])
    res_d = fit_cloglog(panel["event_death"], death_design(panel),
                        groups=panel["hh_code"])
    assert res_c is not None and res_d is not None, "hazard fit failed"
    return res_c, res_d


def coefficient_rows(person, panel, res_c):
    """Pace of Aging coefficient of the claiming hazard, the joint test of
    its interactions with age, and the cause-specific Cox coefficient."""
    cols = list(claim_design(panel).columns)
    j = cols.index("poa_z")
    ci = res_c.conf_int()
    rows = [{"model": "claiming_hazard", "term": "poa_z",
             "estimate": res_c.params[j], "ci_lo": ci[j][0], "ci_hi": ci[j][1],
             "p": res_c.pvalues[j]}]

    # Pace of Aging by age, ages 62 to 68.
    sub = panel.loc[panel["age_year"].isin(INTERACTION_AGES)]
    Xi = claim_design(sub)
    for a in list(INTERACTION_AGES)[1:]:
        Xi[f"poa_x_age_{a}"] = Xi["poa_z"] * Xi[f"age_{a}"]
    Xi = Xi.drop(columns=[f"age_{a}" for a in (69, 70)])
    res_i = fit_cloglog(sub["event_claim"], Xi, groups=sub["hh_code"])
    assert res_i is not None, "interaction fit failed"
    iterms = [c for c in Xi.columns if c.startswith("poa_x_age_")]
    R = np.zeros((len(iterms), len(Xi.columns)))
    for r, c in enumerate(iterms):
        R[r, list(Xi.columns).index(c)] = 1.0
    wald = res_i.wald_test(R, scalar=True)
    rows.append({"model": "poa_by_age_62_68", "term": "joint_wald_interactions",
                 "p": float(wald.pvalue)})

    # Cause-specific Cox model on the age timescale, with delayed entry at
    # the entry month, exit at the end of the exit month, and the controls
    # of the claiming hazard.
    cox = person.copy()
    cox["entry_age"] = (cox["entry_month_idx"] - cox["birth_month_idx"]) / 12
    cox["exit_age"] = (cox["exit_month_idx"] - cox["birth_month_idx"] + 1) / 12
    coh = cox["cohort5"].replace(COHORT_MERGE)
    cox_covs = ["poa_z"] + DEMOGRAPHICS
    for c in COHORT_LEVELS[1:]:
        cox[f"cohort_{int(c)}"] = (coh == c).astype(float)
        cox_covs.append(f"cohort_{int(c)}")
    for w in COMPLETION_WAVES[1:]:
        cox[f"cw_{w}"] = (cox["completion_wave"] == w).astype(float)
        cox_covs.append(f"cw_{w}")
    fit_df = cox[["entry_age", "exit_age", "event_claim", CLUSTER_VAR] + cox_covs]
    cph = fit_cox(fit_df, duration_col="exit_age", event_col="event_claim",
                  entry_col="entry_age", cluster_col=CLUSTER_VAR,
                  formula=" + ".join(cox_covs))
    s = cph.inference.loc["poa_z"]
    rows.append({"model": "cause_specific_cox", "term": "poa_z_log_hr",
                 "estimate": float(s["coef"]),
                 "ci_lo": float(s["coef lower 95%"]),
                 "ci_hi": float(s["coef upper 95%"]), "p": float(s["p"])})
    return pd.DataFrame(rows)


def aalen_johansen(person):
    """Aalen-Johansen cumulative incidence of claiming with delayed entry.

    At each month of age the claim and death hazards are events over the
    respondents followed in that month, and claiming incidence accumulates as
    the event-free probability times the claim hazard. Computed overall and
    by quartile of Pace of Aging, and reported at ages 63, 65, and 70. Also
    returns each group's (respondents, claims, deaths before claiming).
    """
    rows, group_counts = [], {}
    entry = (person["entry_month_idx"] - person["birth_month_idx"]).to_numpy()
    exit_ = (person["exit_month_idx"] - person["birth_month_idx"]).to_numpy()
    ec = person["event_claim"].to_numpy()
    ed = person["event_death"].to_numpy()
    for label in ["all", "Q1", "Q2", "Q3", "Q4"]:
        m = (np.ones(len(person), dtype=bool) if label == "all"
             else (person["poa_quartile"] == label).to_numpy())
        surv, cif = 1.0, 0.0
        marks = {}
        for t in range(62 * 12, 70 * 12 + 1):
            at_risk = int(((entry[m] <= t) & (exit_[m] >= t)).sum())
            if at_risk > 0:
                dc = int(((exit_[m] == t) & (ec[m] == 1)).sum())
                dd = int(((exit_[m] == t) & (ed[m] == 1)).sum())
                cif += surv * dc / at_risk
                surv *= 1 - (dc + dd) / at_risk
            if t in (63 * 12, 65 * 12, 70 * 12):
                marks[t // 12] = cif
        group_counts[label] = (int(m.sum()), int(ec[m].sum()), int(ed[m].sum()))
        for age, v in marks.items():
            row = {"group": label, "age": age, "share_claimed": round(v, 4),
                   **dict(zip(("n", "claims", "deaths"), group_counts[label]))}
            if label != "all":
                row.update({
                    "poa_z_min": round(float(person.loc[m, "poa_z"].min()), 3),
                    "poa_z_max": round(float(person.loc[m, "poa_z"].max()), 3)})
            rows.append(row)
    return pd.DataFrame(rows), group_counts


def bootstrap(person, panel, std, p25, p75, rng, start_c, start_d):
    """Household bootstrap: refit both hazards on the drawn households
    (warm-started at the point estimates) and standardize over the original
    sample. Returns one row of differences per replicate."""
    hh_codes = person["hh_code"].unique()
    panel_by_hh = dict(panel.groupby("hh_code").indices)
    Xc_all = claim_design(panel).to_numpy(float)
    Xd_all = death_design(panel).to_numpy(float)
    yc = panel["event_claim"].to_numpy(float)
    yd = panel["event_death"].to_numpy(float)

    reps = []
    for _ in range(N_BOOT):
        draw = rng.choice(hh_codes, size=len(hh_codes), replace=True)
        idx = np.concatenate([panel_by_hh[h] for h in draw])
        rc = fit_cloglog(yc[idx], Xc_all[idx], start=start_c)
        rd = fit_cloglog(yd[idx], Xd_all[idx], start=start_d)
        if rc is None or rd is None:
            continue    # caught by the replicate count below
        est = std.estimands(rc.params, rd.params, p25, p75)
        reps.append({f"{k}_{a}": est[a][k] for a in TARGET_AGES
                     for k in DIFFERENCES})
    reps = pd.DataFrame(reps)
    assert len(reps) == N_BOOT and np.isfinite(reps.to_numpy()).all(), \
        "a bootstrap replicate failed"
    return reps


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    person, panel = load_frames()
    p25, p75 = person["poa_z"].quantile([0.25, 0.75])

    res_c, res_d = fit_hazards(panel)
    std = Standardization(person)
    point = std.estimands(res_c.params, res_d.params, p25, p75)
    assert {a: point[a]["n_population"] for a in TARGET_AGES} == ENTERED_BY_AGE
    rng = np.random.default_rng(RANDOM_SEED)
    reps = bootstrap(person, panel, std, p25, p75, rng,
                     res_c.params, res_d.params)

    rows = []
    for a in TARGET_AGES:
        for k in LEVELS:
            rows.append({"target_age": a, "estimand": k,
                         "n_population": point[a]["n_population"],
                         "estimate": point[a][k]})
        for k in DIFFERENCES:
            q = reps[f"{k}_{a}"].quantile([0.025, 0.05, 0.95, 0.975])
            row = {"target_age": a, "estimand": k,
                   "n_population": point[a]["n_population"],
                   "estimate": point[a][k],
                   "boot_ci95_lo": q.loc[0.025], "boot_ci95_hi": q.loc[0.975],
                   "n_boot_used": len(reps)}
            if a == 65:     # for the equivalence test of Appendix Table A5
                row.update({"boot_ci90_lo": q.loc[0.05], "boot_ci90_hi": q.loc[0.95]})
            rows.append(row)

    by_quartile, group_counts = aalen_johansen(person)
    assert group_counts == QUARTILE_CLAIMING_COUNTS
    by_quartile.to_csv(TABLES_DIR / "claiming_by_quartile.csv", index=False)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "claiming_shares.csv", index=False)
    coefficient_rows(person, panel, res_c).to_csv(
        TABLES_DIR / "claiming_hazard.csv", index=False)


if __name__ == "__main__":
    main()
