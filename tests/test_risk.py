import numpy as np
import pandas as pd
import pytest
from scipy import stats
from tadawul_risk import risk, stats as rstats, data, config as C


def test_kupiec_zero_when_rate_matches():
    lr, p = risk.kupiec(10, 1000, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)


def test_kupiec_known_value():
    # 20 exceedances in 1000 at 1%: LR = -2[980 ln(.99/.98) + 20 ln(.5)] = 7.83 (reject at 5%)
    lr, p = risk.kupiec(20, 1000, 0.01)
    assert lr == pytest.approx(7.828, abs=0.005) and p < 0.01


def test_christoffersen_detects_clustering():
    clustered = np.r_[np.zeros(480), np.ones(10), np.zeros(510)]
    spread = np.zeros(1000); spread[::100] = 1
    assert risk.christoffersen(clustered)[1] < 0.01
    assert risk.christoffersen(spread)[1] > 0.05


def test_normal_var_es_closed_form():
    x = np.random.default_rng(0).normal(0, 0.01, 200_000)
    v, e = risk.normal_var_es(x, 0.99)
    assert v == pytest.approx(2.326 * 0.01, rel=0.01)
    assert e == pytest.approx(2.665 * 0.01, rel=0.01)
    assert e > v


def test_historical_var_es_on_t_data():
    x = stats.t.rvs(4, size=200_000, random_state=1) * 0.01
    hv, he = risk.hist_var_es(x, 0.99)
    tv, te = risk.t_var_es(x, 0.99)
    assert tv == pytest.approx(hv, rel=0.05) and te == pytest.approx(he, rel=0.07)


def test_drawdown():
    r = pd.Series(np.log([1, 1.1, 0.5, 1.2, 1.0]))  # cumulative wealth 1, 1.1, .55, .66, .66
    dd = rstats.drawdown(r)
    assert dd.min() == pytest.approx(0.55 / 1.1 - 1)
    assert dd.iloc[0] == 0


def test_rolling_var_has_no_lookahead():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=1300)
    x = pd.Series(rng.standard_t(5, 1300) * 0.01, index=idx)
    fc = risk.rolling_var(x, 0.99, win=250, garch_win=1000, refit=100)
    y = x.copy(); y.iloc[-1] = -0.5  # shock on the last day must not change that day's forecast
    fc2 = risk.rolling_var(y, 0.99, win=250, garch_win=1000, refit=100)
    for m in ["Historical", "Normal", "GJR-GARCH-t"]:
        assert fc[m].iloc[-1] == pytest.approx(fc2[m].iloc[-1])
    assert fc["GJR-GARCH-t"].iloc[1000:].notna().all()


def test_price_limit_filter_and_basket():
    idx = pd.date_range("2020-01-01", periods=4)
    tick = [t for v in C.SECTORS.values() for t in v]
    px = pd.DataFrame(100.0, index=idx, columns=tick)
    px.iloc[2, 0] = 150.0  # +50% artefact
    r, flagged = data.clean_stock_returns(px)
    assert len(flagged) == 2 and r.iloc[2, 0] != r.iloc[2, 0]  # NaN
    b = data.ew_basket(np.log(px / px.shift()).iloc[1:], ["1120.SR", "1180.SR"])
    assert b.iloc[0] == pytest.approx(0.0)


def test_pipeline_outputs_exist():
    """Integration check on the committed run outputs."""
    for f in ["return_stats.csv", "garch_params.csv", "var_backtest.csv", "regimes.csv", "results.json"]:
        assert (C.TAB_DIR / f).exists()
    bt = pd.read_csv(C.TAB_DIR / "var_backtest.csv")
    assert set(bt.method) == {"Historical", "Normal", "GJR-GARCH-t"}
    assert bt["kupiec_p"].between(0, 1).all()
    assert len(list(C.FIG_DIR.glob("*.png"))) >= 8
