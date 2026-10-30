"""Universe, sources and study parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
START = "2015-01-01"

# Sector baskets built from large-cap Tadawul constituents (Yahoo Finance .SR tickers).
# Sector labels follow the Saudi Exchange / GICS industry group of each company.
SECTORS = {
    "Banks": ["1120.SR", "1180.SR", "1010.SR", "1150.SR", "1060.SR"],
    "Materials": ["2010.SR", "1211.SR", "2020.SR", "2350.SR"],
    "Energy": ["2222.SR", "2380.SR", "4030.SR", "2030.SR"],
    "Telecom": ["7010.SR", "7020.SR", "7030.SR"],
    "Food & Staples": ["2280.SR", "2050.SR"],
    "Utilities": ["5110.SR", "2082.SR"],
}
NAMES = {
    "1120.SR": "Al Rajhi Bank", "1180.SR": "Saudi National Bank", "1010.SR": "Riyad Bank",
    "1150.SR": "Alinma Bank", "1060.SR": "Saudi Awwal Bank (SAB)", "2010.SR": "SABIC",
    "1211.SR": "Ma'aden", "2020.SR": "SABIC Agri-Nutrients", "2350.SR": "Saudi Kayan",
    "2222.SR": "Saudi Aramco", "2380.SR": "Petro Rabigh", "4030.SR": "Bahri",
    "2030.SR": "SARCO", "7010.SR": "stc", "7020.SR": "Mobily", "7030.SR": "Zain KSA",
    "2280.SR": "Almarai", "2050.SR": "Savola", "5110.SR": "Saudi Electricity",
    "2082.SR": "ACWA Power",
}
MARKET_ETF = "KSA"     # iShares MSCI Saudi Arabia ETF (USD, NYSE Arca), external benchmark
BRENT = "BZ=F"         # ICE Brent front-month futures (Yahoo Finance)
MARKET = "LargeCap EW" # equal-weight composite of all 20 constituents (Tadawul calendar)

REGIMES = {
    "Oil slump 2015-16": ("2015-01-01", "2016-02-29"),
    "Recovery 2016-19": ("2016-03-01", "2020-02-19"),
    "COVID crash 2020": ("2020-02-20", "2020-06-30"),
    "Post-COVID 2020-21": ("2020-07-01", "2021-12-31"),
    "Oil spike 2022": ("2022-01-01", "2022-12-31"),
    "2023-present": ("2023-01-01", "2100-01-01"),
}
VAR_LEVELS = (0.95, 0.99)
ROLL_WIN = 250
CORR_WIN = 126
GARCH_REFIT = 63
GARCH_WIN = 1000
