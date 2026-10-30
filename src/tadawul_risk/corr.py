"""Rolling correlations and regime-level correlation summaries."""
import numpy as np
import pandas as pd
from . import config as C


def rolling_vs(r: pd.DataFrame, target: str, win: int) -> pd.DataFrame:
    return pd.DataFrame({c: r[c].rolling(win).corr(r[target]) for c in r if c != target}).dropna(how="all")


def avg_pairwise(r: pd.DataFrame) -> float:
    m = r.corr().values
    return m[np.triu_indices_from(m, 1)].mean()


def regime_table(daily: pd.DataFrame, wk: pd.DataFrame) -> pd.DataFrame:
    sectors = list(C.SECTORS)
    rows = []
    for name, (a, b) in C.REGIMES.items():
        d, w = daily.loc[a:b], wk.loc[a:b]
        row = {"regime": name, "days": len(d),
               "avg_pairwise_sector_corr": avg_pairwise(d[sectors].dropna(axis=1, thresh=20)),
               "market_ann_vol_%": 100 * d[C.MARKET].std() * np.sqrt(250),
               "market_cum_return_%": 100 * np.expm1(d[C.MARKET].sum()),
               "brent_cum_return_%": 100 * np.expm1(w["Brent"].sum()),
               "corr_market_brent_weekly": w[C.MARKET].corr(w["Brent"])}
        for s in sectors:
            row[f"brent_corr_{s}"] = w[s].corr(w["Brent"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime")
