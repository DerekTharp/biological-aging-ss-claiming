"""
First benefits recorded in the month after the 70th-birthday month, counted
as claims at 70 (note to Appendix Table A1).

The claiming window ends in the 70th-birthday month. A benefit for that month
is paid the month after, so a first receipt recorded in the next month is
consistent with entitlement at 70. Those receipts are recoded as claims in
the 70th-birthday month and three quantities are recomputed: the share
claimed by 70 (Table 2, panel A) and the differences in the standardized
share claimed by 65 between the 25th and 75th percentiles of Pace of Aging
(Table 2, panel B) and of the subjective survival gap (Table 3).

Writes output/tables/claims_at_70.csv.
"""

import numpy as np
import pandas as pd

import claiming
import subjective_survival
from config import (
    CLAIMING_SAMPLE, CLUSTER_VAR, PANEL_PARQUET, PERSON_PARQUET, TABLES_DIR,
    UNCLAIMED_AT_70,
)

MONTHS_AFTER = 1


def recode(person, panel):
    """Copies of the person and panel files with the late first receipts
    recoded as claims in the 70th-birthday month, and their number."""
    person, panel = person.copy(), panel.copy()
    late = (person["outcome"] == "Unclaimed at 70") \
        & (person["claim_status"] == "dated") \
        & ((person["claim_month_idx"] - person["m70"]) <= MONTHS_AFTER)
    assert (person.loc[late, "exit_month_idx"] == person.loc[late, "m70"]).all()
    ids = set(person.loc[late, "hhidpn"])
    person.loc[late, "event_claim"] = 1
    m70 = person.set_index("hhidpn")["m70"]
    hit = panel["hhidpn"].isin(ids) & (panel["month_idx"] == panel["hhidpn"].map(m70))
    assert int(hit.sum()) == len(ids)
    panel.loc[hit, "event_claim"] = 1
    assert panel["event_claim"].sum() == person["event_claim"].sum()
    return person, panel, len(ids)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    person_all, panel, n = recode(pd.read_parquet(PERSON_PARQUET),
                                  pd.read_parquet(PANEL_PARQUET))
    assert n == UNCLAIMED_AT_70["in_month_after_70th_birthday_month"]
    person = person_all.loc[person_all["disposition"] == CLAIMING_SAMPLE].copy()
    person["hh_code"] = pd.factorize(person[CLUSTER_VAR])[0]
    panel = panel.merge(person[["hhidpn", "hh_code"]], on="hhidpn", how="left")

    by_quartile, _ = claiming.aalen_johansen(person)
    by_70 = by_quartile.loc[(by_quartile["group"] == "all")
                            & (by_quartile["age"] == 70), "share_claimed"].iloc[0]

    res_c, res_d = claiming.fit_hazards(panel)
    p25, p75 = person["poa_z"].quantile([0.25, 0.75])
    poa = claiming.Standardization(person).estimands(
        res_c.params, res_d.params, p25, p75)[65]["p25_p75_contrast"]

    sample = subjective_survival.load_sample(person_all)
    gap_panel = subjective_survival.claim_panel(sample)
    fit, shares = subjective_survival.share_tools(sample, gap_panel)
    gap = shares(fit(np.arange(len(gap_panel))))["gap_only|gap_10|p25_p75_contrast"]

    pd.DataFrame([{
        "n_recoded": n, "share_claimed_by_70": by_70,
        "poa_p25_p75_difference_by65": poa,
        "gap_p25_p75_difference_by65": gap,
    }]).to_csv(TABLES_DIR / "claims_at_70.csv", index=False)


if __name__ == "__main__":
    main()
