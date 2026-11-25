"""
The present-value benchmark and the equivalence test (Appendix Table A5).

Benchmark: for each respondent in the claiming sample, the claiming month,
from his or her actual entry, that maximizes the expected present value of
the retired-worker benefit under the cohort's statutory schedule. Survival
follows the reference curve (mortality.py) raised to HR^z, with the
mortality model's hazard ratio, to move it to a chosen Pace of Aging. The
benchmark share claimed by an age is the probability of surviving from entry
to the optimal month when that month falls at or before the age.

The optimal age is a step function of Pace of Aging, so the benchmark is
compared with the data on the difference between the claiming sample's 75th
and 25th percentiles: observed difference (claiming.py) minus benchmark
difference. Each household-bootstrap replicate refits the claiming and death
hazards, the mortality model, and the reference curve, and the benchmark is
rejected when the 95% percentile interval of the difference excludes zero.

Equivalence: the 90% bootstrap interval of the observed difference against a
margin. The per-SD margin is set per SD of the full release and converted to
the claiming sample's SD.

Writes:
- output/tables/benchmark.csv                    panel A, optimal ages and benchmark difference
- output/tables/benchmark_test.csv               panel A, observed minus benchmark, by age
- output/tables/benchmark_reference_curves.csv   alternative reference curves (note)
- output/tables/equivalence_test.csv             panel B
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from benefit_schedule import DISCOUNT_RATES, fra_years, ss_multiplier
from claiming import (
    Standardization, TARGET_AGES, claim_design, death_design, fit_cloglog,
    load_frames,
)
from config import (
    CLUSTER_VAR, COVARIATES, N_BOOT, PERSON_PARQUET, RANDOM_SEED,
    REFERENCE_PARQUET, TABLES_DIR,
)
from mortality import MONTH_GRID, calibrate_reference, fit_cox, load_survival_frame

REFERENCE_CURVE = "mixture_z0"
# Ages other than 65 at which the paper states whether the benchmark is
# rejected, as (discount rate, age).
OTHER_REPORTED_VERDICTS = [(0.03, "FRA"), (0.03, 70),
                           (0.04, 63), (0.04, "FRA"), (0.04, 70)]
MARGIN_PER_RELEASE_SD = 2.0     # points per SD of the full release
MARGIN_P25_P75 = 1.25           # points, 75th minus 25th percentile
CLAIM_GRID_M = np.arange(62 * 12, 70 * 12 + 1)

_MULTIPLIERS = {}


def benefit_multipliers(byear):
    """Benefit as a fraction of PIA for each claiming month from 62 to 70."""
    if byear not in _MULTIPLIERS:
        _MULTIPLIERS[byear] = np.array(
            [ss_multiplier(m / 12, byear) for m in CLAIM_GRID_M])
    return _MULTIPLIERS[byear]


class Benchmark:
    """Benchmark claiming for the claiming sample at one Pace of Aging value.

    At a single Pace of Aging value every respondent has the same survival
    curve, so the optimal claiming month depends only on birth year, entry
    month, and the discount rate. A table per birth year maps each possible
    entry month to the best claiming month at or after it.
    """

    def __init__(self, person):
        p = person.reset_index(drop=True)
        self.entry_m = (p["entry_month_idx"]
                        - p["birth_month_idx"]).astype(int).to_numpy()
        self.byear = p["rabyear"].astype(int).to_numpy()
        self.fra_m = np.array([int(round(fra_years(int(b)) * 12))
                               for b in self.byear])
        self.pops = {}
        for a in TARGET_AGES:
            cutoff = self.fra_m if a == "FRA" else 12 * int(a)
            self.pops[a] = self.entry_m <= cutoff
        self.entry_idx = self.entry_m - CLAIM_GRID_M[0]
        self.ubyears = np.unique(self.byear)

    def _optimal_months(self, s_curve, rate):
        disc = (1.0 + rate) ** (-(MONTH_GRID - MONTH_GRID[0]) / 12)
        tail = np.cumsum((s_curve * disc)[::-1])[::-1]
        tail_g = tail[CLAIM_GRID_M - MONTH_GRID[0]]
        tables = {}
        for by in self.ubyears:
            epv = benefit_multipliers(int(by)) * tail_g
            best = np.empty(len(epv), dtype=int)
            bi, bv = len(epv) - 1, -np.inf
            for k in range(len(epv) - 1, -1, -1):
                if epv[k] >= bv:
                    bv, bi = epv[k], k
                best[k] = bi
            tables[by] = CLAIM_GRID_M[best]
        return tables

    def shares(self, s_curve, rate):
        """Mean benchmark share claimed by each target age, and the mean optimal month."""
        tables = self._optimal_months(s_curve, rate)
        opt = np.empty(len(self.byear))
        for by in self.ubyears:
            m = self.byear == by
            opt[m] = tables[by][self.entry_idx[m]]
        p_alive = (s_curve[opt.astype(int) - MONTH_GRID[0]]
                   / s_curve[self.entry_m - MONTH_GRID[0]])
        out = {}
        for a in TARGET_AGES:
            cutoff = self.fra_m if a == "FRA" else 12 * int(a)
            claimed = np.where(opt <= cutoff, p_alive, 0.0)
            out[a] = float(claimed[self.pops[a]].mean())
        return out, float(opt.mean())

    def contrast(self, ref_curve, hr, rate, z25, z75):
        """Benchmark difference in shares, 75th minus 25th percentile, and
        the mean optimal ages (years) at the two percentiles."""
        c25, o25 = self.shares(ref_curve ** (hr ** z25), rate)
        c75, o75 = self.shares(ref_curve ** (hr ** z75), rate)
        return {a: c75[a] - c25[a] for a in TARGET_AGES}, o25 / 12, o75 / 12


def observed_contrast(std, beta_c, beta_d, z25, z75):
    """Observed difference in standardized shares, 75th minus 25th percentile."""
    c25 = std.claimed_by(beta_c, beta_d, np.full(len(std.person), z25))
    c75 = std.claimed_by(beta_c, beta_d, np.full(len(std.person), z75))
    return {a: float(np.nanmean(c75[a][std.cutoffs[a][0]])
                     - np.nanmean(c25[a][std.cutoffs[a][0]]))
            for a in TARGET_AGES}


def joint_bootstrap(person_all, panel, std, bench, mort_frame, z25, z75,
                    start_c, start_d):
    """Observed minus benchmark differences within shared household draws.

    Households are drawn from the whole release, because the mortality model
    uses respondents outside the claiming sample. Each replicate refits the
    claiming and death hazards on its claiming-sample rows and the mortality
    model and reference curve on its mortality rows.
    """
    Xc = claim_design(panel).to_numpy(float)
    Xd = death_design(panel).to_numpy(float)
    yc = panel["event_claim"].to_numpy(float)
    yd = panel["event_death"].to_numpy(float)
    mort_cols = ["exit_age", "died", "entry_age_completion", "poa_z"] + COVARIATES
    mort = mort_frame[mort_cols + ["entry62"]].reset_index(drop=True)
    formula = "poa_z + " + " + ".join(COVARIATES)

    hh_all = person_all[CLUSTER_VAR].unique()
    panel_by_hh = dict(panel.groupby(CLUSTER_VAR).indices)
    mort_by_hh = dict(mort_frame.reset_index(drop=True).groupby(CLUSTER_VAR).indices)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for _ in range(N_BOOT):
        draw = rng.choice(hh_all, size=len(hh_all), replace=True)
        idx_p = np.concatenate([panel_by_hh[h] for h in draw if h in panel_by_hh])
        idx_m = np.concatenate([mort_by_hh[h] for h in draw if h in mort_by_hh])
        rc = fit_cloglog(yc[idx_p], Xc[idx_p], start=start_c)
        rd = fit_cloglog(yd[idx_p], Xd[idx_p], start=start_d)
        if rc is None or rd is None:
            continue    # caught by the replicate count below
        mf = mort.iloc[idx_m]
        cph = CoxPHFitter()
        cph.fit(mf[mort_cols], duration_col="exit_age", event_col="died",
                entry_col="entry_age_completion", formula=formula)
        hr = float(np.exp(cph.params_["poa_z"]))
        _, curve = calibrate_reference(mf.loc[mf["exit_age"] > 62], hr)
        observed = observed_contrast(std, rc.params, rd.params, z25, z75)
        rec = {}
        for rate in DISCOUNT_RATES:
            benchmark, _, _ = bench.contrast(curve, hr, rate, z25, z75)
            for a in TARGET_AGES:
                rec[(rate, a)] = observed[a] - benchmark[a]
        rows.append(rec)
    reps = pd.DataFrame(rows)
    assert len(reps) == N_BOOT and np.isfinite(reps.to_numpy()).all(), \
        "a bootstrap replicate failed"
    return reps


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    person, panel = load_frames()
    z25, z75 = person["poa_z"].quantile([0.25, 0.75])
    span = z75 - z25
    std = Standardization(person)
    bench = Benchmark(person)
    curves = pd.read_parquet(REFERENCE_PARQUET)

    mort_frame = load_survival_frame()
    mort_frame = mort_frame.loc[mort_frame["exit_age"]
                                > mort_frame["entry_age_completion"]].copy()
    mort_frame["entry62"] = np.maximum(mort_frame["entry_age_completion"], 62.0)
    cph, _ = fit_cox(mort_frame, "entry_age_completion")
    hr = float(np.exp(cph.params_["poa_z"]))

    # Panel A: optimal ages and the benchmark difference in the share claimed by 65.
    s_ref = curves[REFERENCE_CURVE].to_numpy()
    benchmark = {}
    rows = []
    for rate in DISCOUNT_RATES:
        benchmark[rate], o25, o75 = bench.contrast(s_ref, hr, rate, z25, z75)
        rows.append({
            "discount_rate": rate,
            "optimal_age_p25": o25, "optimal_age_p75": o75,
            "optimal_age_shift": o75 - o25,
            "optimal_age_shift_per_sd": (o75 - o25) / span,
            "benchmark_difference_by65": benchmark[rate][65]})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "benchmark.csv", index=False,
                              float_format="%.6f")

    # The same difference under the three alternative reference curves.
    rows = []
    for name in ("km", "cox_full", "cox_nobyear"):
        for rate in DISCOUNT_RATES[:4]:
            diff, _, _ = bench.contrast(curves[name].to_numpy(), hr, rate, z25, z75)
            rows.append({"curve": name, "discount_rate": rate,
                         "benchmark_difference_by65": diff[65]})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "benchmark_reference_curves.csv",
                              index=False, float_format="%.6f")

    # Panel A: observed minus benchmark, bootstrapped jointly.
    res_c = fit_cloglog(panel["event_claim"], claim_design(panel))
    res_d = fit_cloglog(panel["event_death"], death_design(panel))
    observed = observed_contrast(std, res_c.params, res_d.params, z25, z75)
    reps = joint_bootstrap(pd.read_parquet(PERSON_PARQUET), panel, std, bench,
                           mort_frame, z25, z75, res_c.params, res_d.params)
    rows = []
    for rate, a in [(r, 65) for r in DISCOUNT_RATES] + OTHER_REPORTED_VERDICTS:
        q = reps[(rate, a)].quantile([0.025, 0.975])
        row = {"discount_rate": rate, "by_age": str(a),
               "benchmark_rejected": bool(q.loc[0.025] > 0 or q.loc[0.975] < 0),
               "n_boot_used": len(reps)}
        if a == 65:
            row.update({
                "observed_minus_benchmark_pp": (observed[a] - benchmark[rate][a]) * 100,
                "ci95_lo_pp": q.loc[0.025] * 100, "ci95_hi_pp": q.loc[0.975] * 100})
        rows.append(row)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "benchmark_test.csv", index=False,
                              float_format="%.6f")

    # Panel B: equivalence, from the 90% intervals in claiming_shares.csv.
    counts = pd.read_csv(TABLES_DIR / "sample_counts.csv").set_index("metric")["value"]
    shares = pd.read_csv(TABLES_DIR / "claiming_shares.csv")
    shares = shares.loc[shares["target_age"].astype(str) == "65"].set_index("estimand")
    rows = []
    for estimand, margin in [
            ("avg_derivative_per_sd",
             MARGIN_PER_RELEASE_SD * float(counts["claiming_sample_sd_in_release_sd"])),
            ("p25_p75_contrast", MARGIN_P25_P75)]:
        lo = shares.loc[estimand, "boot_ci90_lo"] * 100
        hi = shares.loc[estimand, "boot_ci90_hi"] * 100
        rows.append({"estimand": estimand, "margin_pp": margin,
                     "ci90_lo_pp": lo, "ci90_hi_pp": hi,
                     "equivalent": bool(lo > -margin and hi < margin)})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "equivalence_test.csv", index=False,
                              float_format="%.4f")


if __name__ == "__main__":
    main()
