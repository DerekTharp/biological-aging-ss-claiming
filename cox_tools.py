"""Cox-model quantities computed from delayed-entry risk sets.

The Cox models run on the age timescale with delayed entry. lifelines
estimates their coefficients from the delayed-entry risk sets
{entry < t <= exit}, but its baseline hazard and the score residuals behind
its cluster-robust variance sum over every subject with exit >= t, and those
score residuals also ignore tied event times. This module computes both
quantities from the delayed-entry risk sets, with Efron's treatment of ties,
at the coefficients lifelines estimates:

- ``breslow_survival``: the Breslow baseline survival curve at a covariate
  profile, the counterpart of R's ``survfit(coxph_fit, ctype = 1)``;
- ``robust_summary``: the coefficient table with cluster-robust (sandwich)
  standard errors, the counterpart of R's ``coxph(..., ties = "efron")``
  with ``cluster()``, laid out like ``CoxPHFitter.summary``.

``fit_cox`` fits a lifelines model and attaches ``robust_summary`` as
``.inference``.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import norm

Z_95 = float(norm.ppf(0.975))


def _arrays(cph, frame, duration_col, event_col, entry_col, weights_col):
    cols = list(cph.params_.index)
    X = frame[cols].to_numpy(dtype=float)
    beta = cph.params_.to_numpy(dtype=float)
    T = frame[duration_col].to_numpy(dtype=float)
    S = frame[entry_col].to_numpy(dtype=float)
    E = frame[event_col].to_numpy(dtype=float) > 0
    w = (np.ones(len(frame)) if weights_col is None
         else frame[weights_col].to_numpy(dtype=float))
    return cols, X, beta, T, S, E, w


def breslow_survival(cph, frame, duration_col, event_col, entry_col,
                     weights_col=None, profile=None):
    """Breslow baseline survival at ``profile`` from delayed-entry risk sets.

    The hazard increment at each distinct exit time t is the (weighted) death
    count at t divided by the summed (weighted) relative risk of subjects with
    entry < t <= exit. ``profile`` defaults to the covariate means of
    ``frame``. Returns survival indexed by the distinct exit times, the index
    lifelines uses for its own curves.
    """
    cols, X, beta, T, S, E, w = _arrays(cph, frame, duration_col, event_col,
                                        entry_col, weights_col)
    xbar = X.mean(axis=0)
    x0 = xbar if profile is None else np.asarray(profile, dtype=float).ravel()
    wr = w * np.exp((X - xbar) @ beta)

    times, inverse = np.unique(T, return_inverse=True)
    deaths = np.bincount(inverse, weights=w * E, minlength=len(times))
    # Summed relative risk with exit >= t minus the part with entry >= t.
    order_t, order_s = np.argsort(T), np.argsort(S)
    tail_t = np.concatenate([np.cumsum(wr[order_t][::-1])[::-1], [0.0]])
    tail_s = np.concatenate([np.cumsum(wr[order_s][::-1])[::-1], [0.0]])
    at_risk = (tail_t[np.searchsorted(T[order_t], times, side="left")]
               - tail_s[np.searchsorted(S[order_s], times, side="left")])
    hazard = np.divide(deaths, at_risk, out=np.zeros_like(deaths),
                       where=deaths > 0)
    cum_hazard = np.cumsum(hazard) * float(np.exp((x0 - xbar) @ beta))
    return pd.Series(np.exp(-cum_hazard), index=times)


def robust_summary(cph, frame, duration_col, event_col, entry_col,
                   cluster_col=None, weights_col=None):
    """Coefficient table with delayed-entry, Efron cluster-robust errors.

    Score residuals and the information matrix are built over the risk sets
    entry < t <= exit with Efron's approximation for ties, following the
    counting-process formulas of R's survival package. The variance is the
    sandwich V (sum over clusters of U_c U_c') V, where V is the inverse
    information and U_c sums the weighted score residuals of cluster c; each
    row is its own cluster when ``cluster_col`` is None.
    """
    cols, X, beta, T, S, E, w = _arrays(cph, frame, duration_col, event_col,
                                        entry_col, weights_col)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    risk = np.exp(Xc @ beta)
    resid = np.zeros((n, p))
    info = np.zeros((p, p))

    for t in np.unique(T[E]):
        idx = np.flatnonzero((S < t) & (T >= t))
        dies = E[idx] & (T[idx] == t)
        x, r = Xc[idx], risk[idx]
        wr = w[idx] * r
        denom, a, c = wr.sum(), wr @ x, (x * wr[:, None]).T @ x
        xd, wrd = x[dies], wr[dies]
        e_denom, a2, c2 = wrd.sum(), wrd @ xd, (xd * wrd[:, None]).T @ xd
        n_d = int(dies.sum())
        mean_wt = float(w[idx][dies].mean())

        haz = haz_d = 0.0
        mh1, mh2, mh3 = np.zeros(p), np.zeros(p), np.zeros(p)
        for k in range(n_d):
            frac = k / n_d
            d2 = denom - frac * e_denom
            mean = (a - frac * a2) / d2
            h = mean_wt / d2
            haz += h
            haz_d += (1.0 - frac) * h
            mh1 += h * mean
            mh2 += (1.0 - frac) * h * mean
            mh3 += mean / n_d
            info += mean_wt * ((c - frac * c2) / d2 - np.outer(mean, mean))

        contrib = -r[:, None] * (x * haz - mh1)
        contrib[dies] = (xd - mh3) - r[dies][:, None] * (xd * haz_d - mh2)
        resid[idx] += contrib

    var_model = np.linalg.inv(info)
    dfbeta = (resid * w[:, None]) @ var_model
    groups = (np.arange(n) if cluster_col is None
              else pd.factorize(frame[cluster_col])[0])
    by_cluster = np.zeros((groups.max() + 1, p))
    np.add.at(by_cluster, groups, dfbeta)
    se = np.sqrt(np.diag(by_cluster.T @ by_cluster))
    z = beta / se
    return pd.DataFrame({
        "coef": beta,
        "exp(coef)": np.exp(beta),
        "se(coef)": se,
        "coef lower 95%": beta - Z_95 * se,
        "coef upper 95%": beta + Z_95 * se,
        "z": z,
        "p": 2.0 * norm.sf(np.abs(z)),
    }, index=pd.Index(cols, name="covariate"))


def fit_cox(frame, *, duration_col, event_col, entry_col, cluster_col, formula):
    """Fit a lifelines Cox model with delayed entry and household clusters.

    The returned model carries ``.inference``, the ``robust_summary`` table;
    read standard errors, p-values, and intervals from it rather than from
    ``.summary``.
    """
    cph = CoxPHFitter()
    cph.fit(frame, duration_col=duration_col, event_col=event_col,
            entry_col=entry_col, cluster_col=cluster_col, formula=formula)
    cph.inference = robust_summary(cph, frame, duration_col, event_col,
                                   entry_col, cluster_col=cluster_col)
    return cph
