"""GARCH(1,1) and GJR-GARCH(1,1,1) with Student-t innovations (arch package)."""
import pandas as pd
from arch import arch_model


def fit(x: pd.Series, gjr: bool = False):
    am = arch_model(100 * x.dropna(), mean="Constant", vol="GARCH", p=1, o=1 if gjr else 0, q=1, dist="t")
    return am.fit(disp="off")


def garch_table(r: pd.DataFrame):
    rows, cvol = [], {}
    for c in r:
        for name, gjr in (("GARCH(1,1)", False), ("GJR-GARCH", True)):
            res = fit(r[c], gjr)
            p = res.params
            gamma = p.get("gamma[1]", 0.0)
            pers = p["alpha[1]"] + p["beta[1]"] + gamma / 2
            rows.append({"series": c, "model": name, "mu": p["mu"], "omega": p["omega"],
                         "alpha": p["alpha[1]"], "gamma": gamma, "beta": p["beta[1]"], "nu": p["nu"],
                         "gamma_pvalue": res.pvalues.get("gamma[1]", float("nan")),
                         "persistence": pers, "half_life_days": (pd.NA if pers >= 1 else
                                          __import__("math").log(0.5) / __import__("math").log(pers)),
                         "loglik": res.loglikelihood, "aic": res.aic, "bic": res.bic})
            if gjr:
                cvol[c] = res.conditional_volatility * (250 ** 0.5)  # annualised, in %
    return pd.DataFrame(rows), pd.DataFrame(cvol)
