"""
Interactions of Pace of Aging with liquid wealth and with partnership in the
Section 2 claiming hazard (note to Appendix Table A3).

Liquid wealth is checking, certificates of deposit, stocks, and bonds at the
completion wave, in thirds of the claiming sample. Partnership is married or
partnered at the completion wave. Each model adds the variable and its
product with Pace of Aging to the Section 2 specification.

Writes output/tables/interactions.csv.
"""

import numpy as np
import pandas as pd

from claiming import claim_design, fit_cloglog, load_frames
from config import CLUSTER_VAR, TABLES_DIR


def fit_interaction(panel, mod_cols):
    """Returns the fit, its column names, and the joint Wald p-value of the
    interactions of Pace of Aging with ``mod_cols``."""
    X = claim_design(panel)
    for c in mod_cols:
        X[c] = panel[c].astype(float)
        X[f"poa_x_{c}"] = X["poa_z"] * X[c]
    res = fit_cloglog(panel["event_claim"], X,
                      groups=pd.factorize(panel[CLUSTER_VAR])[0])
    assert res is not None and np.abs(res.params).max() < 30
    cols = list(X.columns)
    iterms = [f"poa_x_{c}" for c in mod_cols]
    R = np.zeros((len(iterms), len(cols)))
    for r, c in enumerate(iterms):
        R[r, cols.index(c)] = 1.0
    return res, cols, float(res.wald_test(R, scalar=True).pvalue)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    person, panel = load_frames()
    t1, t2 = person["liquid_wealth_cw"].quantile([1 / 3, 2 / 3])
    person["wealth_third"] = np.select(
        [person["liquid_wealth_cw"] <= t1, person["liquid_wealth_cw"] <= t2],
        ["W1", "W2"], "W3")
    panel = panel.join(person.set_index("hhidpn")[["wealth_third",
                                                   "married_partnered_cw"]],
                       on="hhidpn")
    for w in ("W2", "W3"):
        panel[f"wealth_{w}"] = (panel["wealth_third"] == w).astype(float)
    _, _, p_wealth = fit_interaction(panel, ["wealth_W2", "wealth_W3"])

    partnered = panel.dropna(subset=["married_partnered_cw"]).copy()
    res, cols, _ = fit_interaction(partnered, ["married_partnered_cw"])
    p_partner = res.pvalues[cols.index("poa_x_married_partnered_cw")]

    pd.DataFrame([
        {"interaction": "poa_x_liquid_wealth_thirds", "p": p_wealth},
        {"interaction": "poa_x_married_partnered", "p": p_partner},
    ]).to_csv(TABLES_DIR / "interactions.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
