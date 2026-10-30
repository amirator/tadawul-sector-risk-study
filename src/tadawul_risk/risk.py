"""VaR / ES estimators, rolling VaR forecasts and backtests (Kupiec POF, Christoffersen)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model


def hist_var_es(x, level=0.99):
    x = np.asarray(x)
    q = np.quantile(x, 1 - level)
    return -q, -x[x <= q].mean()


def normal_var_es(x, level=0.99):
    mu, sd = np.mean(x), np.std(x, ddof=1)
    z = stats.norm.ppf(1 - level)
    return -(mu + sd * z), -(mu - sd * stats.norm.pdf(z) / (1 - level))


def t_var_es(x, level=0.99):
    nu, loc, scale = stats.t.fit(x)
    a = 1 - level
    q = stats.t.ppf(a, nu)
    es_std = (stats.t.pdf(q, nu) / a) * (nu + q ** 2) / (nu - 1)
    return -(loc + scale * q), -(loc - scale * es_std)


def static_table(r: pd.DataFrame, levels=(0.95, 0.99)) -> pd.DataFrame:
    rows = []
    for c in r:
        x = r[c].dropna().values
        for lv in levels:
            for m, f in (("Historical", hist_var_es), ("Normal", normal_var_es), ("Student-t", t_var_es)):
                v, e = f(x, lv)
                rows.append({"series": c, "level": lv, "method": m, "VaR_%": 100 * v, "ES_%": 100 * e})
    return pd.DataFrame(rows)


def kupiec(n_exc: int, n: int, p: float):
    """Kupiec (1995) proportion-of-failures LR test. Returns (LR, p-value)."""
    x, phat = n_exc, n_exc / n
    if x == 0:
        lr = -2 * (n * np.log(1 - p))
    elif x == n:
        lr = -2 * (n * np.log(p))
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen(hits: np.ndarray):
    """Christoffersen (1998) independence LR test on a 0/1 hit sequence."""
    h = np.asarray(hits).astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(len(a), 1)
    def ll(k, pr):
        return 0.0 if pr in (0, 1) or k == 0 else k * np.log(pr)
    l0 = ll(n00 + n10, 1 - p) + ll(n01 + n11, p)
    l1 = ll(n00, 1 - p01) + ll(n01, p01) + ll(n10, 1 - p11) + ll(n11, p11)
    lr = max(-2 * (l0 - l1), 0.0)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def rolling_var(x: pd.Series, level=0.99, win=250, garch_win=1000, refit=63) -> pd.DataFrame:
    """One-day-ahead VaR forecasts (positive loss numbers, log-return units) made with data up to t-1."""
    x = x.dropna()
    a = 1 - level
    hist = -x.rolling(win).quantile(a).shift(1)
    mu, sd = x.rolling(win).mean().shift(1), x.rolling(win).std().shift(1)
    normal = -(mu + sd * stats.norm.ppf(a))
    # GJR-GARCH-t: re-estimated every `refit` days on a rolling window, filtered daily in between.
    y = 100 * x
    g = pd.Series(np.nan, index=x.index)
    am = arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
    for start in range(garch_win, len(y), refit):
        sub = arch_model(y.iloc[start - garch_win:start], mean="Constant", vol="GARCH",
                         p=1, o=1, q=1, dist="t").fit(disp="off")
        fixed = am.fix(sub.params)
        fc = fixed.forecast(horizon=1, start=start - 1, reindex=False)
        end = min(start + refit, len(y))
        var1 = fc.variance.iloc[: end - start + 0, 0].values[: end - start]
        nu = sub.params["nu"]
        qz = stats.t.ppf(a, nu) * np.sqrt((nu - 2) / nu)
        g.iloc[start:end] = -(sub.params["mu"] + np.sqrt(var1) * qz) / 100
    return pd.DataFrame({"ret": x, "Historical": hist, "Normal": normal, "GJR-GARCH-t": g})


def backtest(fc: pd.DataFrame, level=0.99) -> pd.DataFrame:
    fc = fc.dropna()
    rows = []
    for m in ["Historical", "Normal", "GJR-GARCH-t"]:
        hits = (fc["ret"] < -fc[m]).values
        n, k = len(hits), int(hits.sum())
        lr, pv = kupiec(k, n, 1 - level)
        lri, pvi = christoffersen(hits)
        rows.append({"method": m, "level": level, "n": n, "expected": n * (1 - level), "exceedances": k,
                     "rate_%": 100 * k / n, "kupiec_LR": lr, "kupiec_p": pv,
                     "christoffersen_LR": lri, "christoffersen_p": pvi,
                     "kupiec_verdict": "reject" if pv < 0.05 else "do not reject"})
    return pd.DataFrame(rows)
