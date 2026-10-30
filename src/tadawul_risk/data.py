"""Download, cache and assemble real public price data (Yahoo Finance via yfinance)."""
from __future__ import annotations
import datetime as dt
import json
import numpy as np
import pandas as pd
from . import config as C


def all_tickers() -> list[str]:
    return [t for v in C.SECTORS.values() for t in v] + [C.MARKET_ETF, C.BRENT]


def download(refresh: bool = False) -> pd.DataFrame:
    path = C.DATA_RAW / "prices_adjclose.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    px = yf.download(all_tickers(), start=C.START, auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.to_csv(path)
    manifest = {
        "source": "Yahoo Finance via yfinance (auto_adjust=True: split/dividend-adjusted close)",
        "downloaded_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "start": C.START, "tickers": {t: C.NAMES.get(t, t) for t in all_tickers()},
        "rows": int(len(px)),
    }
    (C.DATA_RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return px


def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff()


PRICE_LIMIT = 0.105  # Tadawul daily price limit is +/-10%; larger moves are data/corporate-action artefacts


def clean_stock_returns(px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-stock daily log returns on each stock's own trading days, with artefacts removed.

    Returns (returns, flagged) where flagged lists removed observations (|simple return| > 10.5%)."""
    stocks = [t for v in C.SECTORS.values() for t in v]
    simple = px[stocks].apply(lambda c: c.dropna().pct_change())
    flag = simple.abs() > PRICE_LIMIT
    flagged = simple.where(flag).stack().dropna().rename("simple_return").reset_index()
    flagged.columns = ["date", "ticker", "simple_return"]
    simple = simple.mask(flag)
    return np.log1p(simple), flagged


def tadawul_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Keep Tadawul sessions only: dates on which at least 10 constituents printed a price."""
    r, _ = clean_stock_returns(px)
    return r[r.notna().sum(axis=1) >= 10]


def ew_basket(r: pd.DataFrame, members: list[str]) -> pd.Series:
    """Daily-rebalanced equal-weight basket of available members (simple-return average)."""
    simple = np.expm1(r[members])
    return np.log1p(simple.mean(axis=1, skipna=True))


def build_panel(px: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of daily log returns: sector baskets + LargeCap EW market composite."""
    r = tadawul_returns(px)
    out = {s: ew_basket(r, m) for s, m in C.SECTORS.items()}
    out[C.MARKET] = ew_basket(r, [t for v in C.SECTORS.values() for t in v])
    return pd.DataFrame(out).dropna()


def weekly(px: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Weekly (Thursday-anchored) log returns of baskets and Brent to avoid calendar asynchrony."""
    wk = panel.resample("W-THU").sum(min_count=1)
    brent = np.log(px[C.BRENT].dropna()).resample("W-THU").last().diff()
    wk["Brent"] = brent
    return wk.dropna()
