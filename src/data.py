"""Data loading: CSV prices + deterministic synthetic multi-asset generator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_csv_prices(path: str | Path, price_col: str = "Close") -> pd.Series:
    """Load a single asset's price series from a CSV with a Date index."""
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[price_col].astype(float).rename(Path(path).stem)


def load_csv_universe(folder: str | Path, price_col: str = "Close") -> pd.DataFrame:
    """Load every CSV in a folder into a wide DataFrame of close prices."""
    folder = Path(folder)
    series = [load_csv_prices(p, price_col) for p in sorted(folder.glob("*.csv"))]
    if not series:
        raise FileNotFoundError(f"No CSVs found in {folder}")
    return pd.concat(series, axis=1).ffill().dropna(how="all")


def synthetic_universe(
    n_days: int = 252 * 20,
    start: str = "2005-01-03",
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a reproducible multi-asset universe with trends, regimes, crashes.

    Builds 8 assets: a broad equity factor (trending up with drawdowns), a bond-like
    low-vol asset, a commodity-like mean-reverting asset, and 5 noisy satellites
    with heterogeneous drifts and vols. Regime shifts and crash events are injected
    so momentum/trend filters have realistic signal to extract.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    # Market factor: realistic positive drift, occasional bear regimes + crashes.
    mkt = np.zeros(n_days)
    bull_drift, bear_drift = 0.00055, -0.00040  # ~14%/yr bull, ~-10%/yr bear
    bull_vol, bear_vol = 0.010, 0.020
    regime = "bull"
    regime_left = int(rng.integers(252 * 4, 252 * 8))
    for t in range(n_days):
        if regime_left <= 0:
            if regime == "bull":
                regime = "bear"
                regime_left = int(rng.integers(int(252 * 0.5), int(252 * 1.5)))
            else:
                regime = "bull"
                regime_left = int(rng.integers(252 * 4, 252 * 8))
        mu = bull_drift if regime == "bull" else bear_drift
        sigma = bull_vol if regime == "bull" else bear_vol
        shock = rng.normal(mu, sigma)
        if regime == "bear" and rng.random() < 1 / 252:
            shock -= rng.uniform(0.02, 0.05)  # capitulation days
        mkt[t] = shock
        regime_left -= 1

    assets = {}
    # Broad equity: loads heavily on market.
    assets["EQ_BROAD"] = 1.0 * mkt + rng.normal(0, 0.004, n_days)
    # Defensive / bond-like: low vol, slight positive drift, negative beta to crashes.
    assets["BOND_AGG"] = 0.00015 + rng.normal(0, 0.0035, n_days) - 0.25 * mkt
    # Commodity: mean-reverting with occasional trends.
    com = np.zeros(n_days)
    for t in range(1, n_days):
        com[t] = -0.05 * com[t - 1] + rng.normal(0.0001, 0.014)
    assets["COMMOD"] = com
    # 5 equity satellites with varied drifts/vols and partial market loading.
    for i, (d, v, b) in enumerate(
        [(0.0006, 0.014, 1.2), (0.0003, 0.016, 0.9),
         (0.0005, 0.018, 1.1), (0.0002, 0.012, 0.7),
         (0.0004, 0.020, 1.3)]
    ):
        assets[f"SAT_{i+1}"] = d + b * mkt + rng.normal(0, v, n_days)

    returns = pd.DataFrame(assets, index=dates)
    prices = 100 * (1 + returns).cumprod()
    return prices
