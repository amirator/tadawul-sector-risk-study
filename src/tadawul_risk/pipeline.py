"""End-to-end pipeline: python -m tadawul_risk.pipeline [--refresh]"""
import argparse
import json
import numpy as np
import pandas as pd
from . import config as C, data, stats, vol, risk, corr, plots


def run(refresh: bool = False) -> dict:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    px = data.download(refresh)
    daily = data.build_panel(px)
    data.clean_stock_returns(px)[1].to_csv(C.TAB_DIR / "flagged_observations.csv", index=False)
    wk = data.weekly(px, daily)
    sectors = list(C.SECTORS)
    ksa = data.log_returns(px[C.MARKET_ETF]).dropna()

    st = stats.summary(daily)
    st.loc["KSA ETF (USD)"] = stats.summary(ksa.to_frame("KSA ETF (USD)")).iloc[0]
    st.loc["Brent (weekly)"] = stats.summary(wk[["Brent"]], ann=52).iloc[0]
    st.to_csv(C.TAB_DIR / "return_stats.csv")

    gt, cv = vol.garch_table(daily)
    gt.to_csv(C.TAB_DIR / "garch_params.csv", index=False)

    dd = daily.apply(stats.drawdown)
    ddt = stats.drawdown_table(daily); ddt.to_csv(C.TAB_DIR / "drawdowns.csv")

    rc_mkt = corr.rolling_vs(daily, C.MARKET, C.CORR_WIN)
    rc_brent = corr.rolling_vs(wk[sectors + ["Brent"]], "Brent", 52)
    reg = corr.regime_table(daily, wk); reg.to_csv(C.TAB_DIR / "regimes.csv")

    static = risk.static_table(daily, C.VAR_LEVELS); static.to_csv(C.TAB_DIR / "var_es_static.csv", index=False)

    bts, fcs = [], {}
    for c in daily:
        for lv in C.VAR_LEVELS:
            fc = risk.rolling_var(daily[c], lv, C.ROLL_WIN, C.GARCH_WIN, C.GARCH_REFIT)
            if lv == 0.99:
                fcs[c] = fc
            b = risk.backtest(fc, lv); b.insert(0, "series", c); bts.append(b)
    bt = pd.concat(bts, ignore_index=True); bt.to_csv(C.TAB_DIR / "var_backtest.csv", index=False)
    fcs[C.MARKET].to_csv(C.TAB_DIR / "var_forecasts_market_99.csv")

    figs = [plots.growth(daily, ksa), plots.cond_vol(cv),
            plots.roll_corr(rc_mkt, f"Rolling {C.CORR_WIN}-day correlation of each sector with the LargeCap EW composite",
                            "03_rolling_corr_market.png"),
            plots.roll_corr(rc_brent, "Rolling 52-week correlation of weekly sector returns with Brent crude",
                            "04_rolling_corr_brent.png"),
            plots.drawdowns(dd),
            plots.var_backtest(fcs[C.MARKET], "LargeCap EW composite: one-day 99% VaR forecasts vs realised returns",
                               "06_var_backtest_market.png"),
            plots.kupiec_heat(bt), plots.regime_heatmaps(daily, sectors)]

    covid = cv.loc["2020-02-20":"2020-06-30"]
    results = {
        "sample": {"start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
                   "n_days": int(len(daily)), "n_weeks": int(len(wk))},
        "peak_covid_gjr_vol_%": covid.max().round(1).to_dict(),
        "peak_covid_gjr_vol_date": {k: str(v.date()) for k, v in covid.idxmax().items()},
        "median_gjr_vol_%": cv.median().round(1).to_dict(),
        "latest_gjr_vol_%": cv.iloc[-1].round(1).to_dict(),
        "figures": [p.name for p in figs],
    }
    from . import thumbnail; thumbnail.make()
    (C.TAB_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--refresh", action="store_true")
    print(json.dumps(run(ap.parse_args().refresh), indent=2))
