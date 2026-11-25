"""
Timing in the claiming sample (Appendix Table A2) and the profile of
respondents unclaimed at 70 (note to Appendix Table A1).

Writes output/tables/timing.csv.
"""

import pandas as pd
import pyreadstat

from config import (
    CLAIMING_SAMPLE, PERSON_PARQUET, RAND_HRS_DTA, TABLES_DIR, UNCLAIMED_AT_70,
)


def summary_rows(series, metric, group):
    s = pd.Series(series).dropna()
    return [{"metric": metric, "group": group, "stat": st, "value": round(v, 3)}
            for st, v in [("mean", s.mean()), ("sd", s.std()),
                          ("p25", s.quantile(0.25)), ("p50", s.quantile(0.50)),
                          ("p75", s.quantile(0.75))]]


def unclaimed_profile(sample):
    """Respondents unclaimed at 70, against the claimers.

    Those unclaimed at 70 either have a first receipt dated after the
    70th-birthday month (counted, with the number dated to the month right
    after it) or have none recorded. The second group is compared with the
    claimers on birthplace outside the United States (RAND rabplace 11 or
    13), a longest job in public administration (2007 census industry 19 in
    waves 10-15, else 1980 census industry 13), and fewer than 10 years
    worked (maximum of rXjyears over waves).
    """
    waves = range(1, 16)
    _, meta = pyreadstat.read_dta(str(RAND_HRS_DTA), metadataonly=True)
    want = (["hhidpn", "rabplace"] + [f"r{w}jyears" for w in waves]
            + [f"r{w}jlind" for w in range(1, 10)]
            + [f"r{w}jlindc" for w in range(10, 16)])
    rand, _ = pyreadstat.read_dta(str(RAND_HRS_DTA),
                                  usecols=[c for c in want if c in meta.column_names])
    rand["hhidpn"] = rand["hhidpn"].astype("int64")
    # The latest coded wave carries the longest job to date.
    ind07 = rand[[c for c in rand if c.endswith("jlindc")]].ffill(axis=1).iloc[:, -1]
    ind80 = rand[[c for c in rand if c.endswith("jlind")]].ffill(axis=1).iloc[:, -1]
    rand["foreign_born"] = rand["rabplace"].isin([11, 13])
    rand["public_admin"] = (ind07 == 19) | (ind07.isna() & (ind80 == 13))
    rand["worked_lt10"] = rand[[c for c in rand if c.endswith("jyears")]].max(axis=1) < 10
    d = sample.merge(rand[["hhidpn", "foreign_born", "public_admin", "worked_lt10"]],
                     on="hhidpn", how="left", validate="one_to_one")
    unclaimed = d["outcome"] == "Unclaimed at 70"
    late = d.loc[unclaimed & (d["claim_status"] == "dated")]
    groups = {"claimers": d.loc[d["outcome"] == "Claim in window"],
              "no_receipt": d.loc[unclaimed & (d["claim_status"] != "dated")]}
    assert int(unclaimed.sum()) == len(late) + len(groups["no_receipt"])
    counts = {"late_receipt": len(late),
              "in_month_after_70th_birthday_month":
                  int((late["claim_month_idx"] - late["m70"] <= 1).sum()),
              "no_receipt": len(groups["no_receipt"])}
    assert counts == UNCLAIMED_AT_70

    rows = [{"metric": "unclaimed_at_70", "group": "late_receipt", "stat": "n",
             "value": counts["late_receipt"]},
            {"metric": "unclaimed_at_70", "group": "late_receipt",
             "stat": "n_in_month_after_70th_birthday_month",
             "value": counts["in_month_after_70th_birthday_month"]},
            {"metric": "unclaimed_at_70", "group": "no_receipt", "stat": "n",
             "value": counts["no_receipt"]}]
    for label, g in groups.items():
        for col in ("foreign_born", "public_admin", "worked_lt10"):
            rows.append({"metric": "unclaimed_at_70", "group": label,
                         "stat": f"share_{col}",
                         "value": round(float(g[col].mean()), 3)})
    return rows


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(PERSON_PARQUET)
    sample = p.loc[p["disposition"] == CLAIMING_SAMPLE].copy()
    claimers = sample.loc[sample["event_claim"] == 1]

    entry_age = (sample["entry_month_idx"] - sample["birth_month_idx"]) / 12
    rows = summary_rows(entry_age, "entry_age_years", "claiming_sample")
    rows.append({"metric": "entry_age_years", "group": "claiming_sample",
                 "stat": "share_at_exact_62",
                 "value": round(float(
                     (sample["entry_month_idx"] == sample["m62"]).mean()), 3)})
    rows += summary_rows(claimers["rassagem"] / 12, "claiming_age_years", "claimers")
    rows += summary_rows(
        claimers["claim_month_idx"] - claimers["completion_month_idx"],
        "months_from_completion_to_claim", "claimers")
    rows += summary_rows(
        sample["exit_month_idx"] - sample["entry_month_idx"] + 1,
        "months_of_follow_up", "claiming_sample")
    rows += unclaimed_profile(sample)

    pd.DataFrame(rows).to_csv(TABLES_DIR / "timing.csv", index=False)


if __name__ == "__main__":
    main()
