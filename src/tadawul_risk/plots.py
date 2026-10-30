"""Figures (matplotlib, PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import config as C

NAVY = "#0b3d5e"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
EVENTS = {"COVID crash": ("2020-02-20", "2020-04-30"), "Oil slump": ("2015-01-01", "2016-02-29"),
          "2022 oil spike": ("2022-02-24", "2022-06-30")}


def _shade(ax):
    for lab, (a, b) in EVENTS.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=0.12)


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = C.FIG_DIR / name
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def growth(r, ksa):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for c in r:
        ax.plot(np.exp(r[c].cumsum()), lw=2.4 if c == C.MARKET else 1.2,
                color="black" if c == C.MARKET else None, label=c)
    k = np.exp(ksa.loc[r.index[0]:].cumsum())
    ax.plot(k, ls="--", color="grey", lw=1, label="iShares MSCI Saudi (KSA, USD)")
    _shade(ax); ax.set_yscale("log"); ax.set_title("Growth of 1 SAR, equal-weight sector baskets (log scale)")
    ax.legend(ncol=3, fontsize=8); return _save(fig, "01_growth.png")


def cond_vol(cv):
    cols = list(cv.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 1.6 * len(cols)), sharex=True)
    for ax, c in zip(axes, cols):
        ax.fill_between(cv.index, cv[c], color=NAVY if c == C.MARKET else "tab:blue", alpha=0.7, lw=0)
        ax.set_ylim(0, min(100, cv[c].max() * 1.05)); _shade(ax)
        ax.text(0.01, 0.8, c, transform=ax.transAxes, fontsize=9, weight="bold")
    axes[len(cols) // 2].set_ylabel("annualised volatility, %")
    fig.suptitle("GJR-GARCH(1,1)-t conditional volatility by sector (y-axis capped at 100%)")
    return _save(fig, "02_gjr_conditional_vol.png")


def roll_corr(rc, title, name, ylabel="correlation"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in rc:
        ax.plot(rc[c], lw=1.2, label=c)
    _shade(ax); ax.axhline(0, color="k", lw=0.6); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(ncol=4, fontsize=8); return _save(fig, name)


def drawdowns(dd):
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in dd:
        ax.plot(100 * dd[c], lw=2 if c == C.MARKET else 0.9, color="black" if c == C.MARKET else None, label=c)
    _shade(ax); ax.set_ylabel("% below previous peak"); ax.set_title("Drawdowns")
    ax.legend(ncol=4, fontsize=8); return _save(fig, "05_drawdowns.png")


def var_backtest(fc, title, name):
    fc = fc.dropna()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(100 * fc["ret"], color="grey", lw=0.5, label="daily return")
    for m, col in (("Historical", "tab:blue"), ("Normal", "tab:orange"), ("GJR-GARCH-t", "tab:red")):
        ax.plot(-100 * fc[m], color=col, lw=1, label=f"-VaR99 {m}")
    hits = fc["ret"] < -fc["GJR-GARCH-t"]
    ax.scatter(fc.index[hits], 100 * fc["ret"][hits], color="red", s=12, zorder=5, label="GJR exceedance")
    ax.set_ylabel("%"); ax.set_title(title); ax.legend(fontsize=8, ncol=3)
    return _save(fig, name)


def kupiec_heat(bt):
    piv = bt[bt.level == 0.99].pivot(index="series", columns="method", values="rate_%")
    pv = bt[bt.level == 0.99].pivot(index="series", columns="method", values="kupiec_p")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=0, vmax=3)
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=20, ha="right"); ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}%\np={pv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.grid(False); fig.colorbar(im, label="exceedance rate % (target 1%)")
    ax.set_title("99% VaR backtest: exceedance rate and Kupiec p-value")
    return _save(fig, "07_kupiec_heatmap.png")


def regime_heatmaps(daily, sectors):
    pre, cov = daily.loc["2016-03-01":"2020-02-19", sectors], daily.loc["2020-02-20":"2020-06-30", sectors]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, d, t in ((axes[0], pre, "Pre-COVID (Mar-2016 to Feb-2020)"), (axes[1], cov, "COVID crash (Feb-Jun 2020)")):
        m = d.corr()
        im = ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(sectors)), sectors, rotation=45, ha="right"); ax.set_yticks(range(len(sectors)), sectors)
        for i in range(len(sectors)):
            for j in range(len(sectors)):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if m.values[i, j] > 0.6 else "black")
        ax.grid(False); ax.set_title(t)
    fig.colorbar(im, ax=axes, shrink=0.8)
    p = C.FIG_DIR / "08_regime_corr_heatmaps.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p
