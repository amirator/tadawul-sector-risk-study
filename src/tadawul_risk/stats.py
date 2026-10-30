"""Return statistics and drawdowns."""
import numpy as np
import pandas as pd
from scipy import stats


def summary(r: pd.DataFrame, ann: int = 250) -> pd.DataFrame:
    rows = {}
    for c in r:
        x = r[c].dropna()
        jb = stats.jarque_bera(x)
        rows[c] = {
            "start": x.index.min().date(), "n_obs": len(x),
            "ann_return_%": 100 * (np.exp(x.mean() * ann) - 1),
            "ann_vol_%": 100 * x.std() * np.sqrt(ann),
            "sharpe_rf0": x.mean() / x.std() * np.sqrt(ann),
            "skew": stats.skew(x), "excess_kurt": stats.kurtosis(x),
            "jb_pvalue": jb.pvalue, "worst_day_%": 100 * np.expm1(x.min()),
            "best_day_%": 100 * np.expm1(x.max()),
        }
    return pd.DataFrame(rows).T


def drawdown(r: pd.Series) -> pd.Series:
    w = np.exp(r.fillna(0).cumsum())
    return w / w.cummax() - 1


def drawdown_table(r: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for c in r:
        dd = drawdown(r[c])
        trough = dd.idxmin()
        peak = dd[:trough][dd[:trough] == 0].index.max()
        rec = dd[trough:][dd[trough:] == 0]
        rows[c] = {"max_drawdown_%": 100 * dd.min(), "peak": peak.date(), "trough": trough.date(),
                   "recovered": rec.index.min().date() if len(rec) else "not yet",
                   "current_drawdown_%": 100 * dd.iloc[-1]}
    return pd.DataFrame(rows).T
