"""Dual-momentum trend-following strategy with inverse-volatility sizing.

Rules (monthly rebalance on last business day):
  1. Absolute momentum / trend filter: asset is eligible only if its price is
     above its `trend_lookback`-day SMA and its `mom_lookback`-day total return
     is positive (above the risk-free proxy).
  2. Relative momentum: rank eligible assets by `mom_lookback`-day return and
     keep the top `top_n`.
  3. Inverse-volatility weighting among selected assets, scaled so that
     portfolio ex-ante volatility targets `vol_target` (annualized). Any
     unallocated weight sits in cash (risk-free).
  4. Hard per-asset cap `max_weight` to avoid concentration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    mom_lookback: int = 126        # ~6 months
    trend_lookback: int = 200      # ~10 months
    vol_lookback: int = 63         # ~3 months
    top_n: int = 3
    vol_target: float = 0.12       # 12% annualized
    max_weight: float = 0.40
    rf_annual: float = 0.02        # cash yield assumption
    rebalance: str = "BME"         # business month end


def _annualize_vol(daily_ret: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return daily_ret.rolling(lookback).std() * np.sqrt(252)


def compute_target_weights(prices: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Return a DataFrame of target weights indexed by rebalance dates."""
    if prices.isna().any().any():
        prices = prices.ffill().dropna()

    daily_ret = prices.pct_change()
    sma = prices.rolling(cfg.trend_lookback).mean()
    mom = prices.pct_change(cfg.mom_lookback)
    vol = _annualize_vol(daily_ret, cfg.vol_lookback)

    rf_period = (1 + cfg.rf_annual) ** (cfg.mom_lookback / 252) - 1
    eligible = (prices > sma) & (mom > rf_period)

    rebal_dates = prices.resample(cfg.rebalance).last().index
    rebal_dates = rebal_dates[rebal_dates >= prices.index[cfg.trend_lookback]]

    weights = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)
    for dt in rebal_dates:
        if dt not in prices.index:
            continue
        elig = eligible.loc[dt]
        if not elig.any():
            continue
        m = mom.loc[dt].where(elig)
        v = vol.loc[dt].where(elig).replace(0, np.nan)
        picks = m.dropna().nlargest(cfg.top_n).index
        if len(picks) == 0:
            continue
        inv_vol = 1.0 / v.loc[picks]
        w = inv_vol / inv_vol.sum()
        # Scale to volatility target (diagonal approximation).
        port_vol = float(np.sqrt((w**2 * v.loc[picks] ** 2).sum()))
        if port_vol > 0:
            scale = min(1.0, cfg.vol_target / port_vol)
            w = w * scale
        w = w.clip(upper=cfg.max_weight)
        weights.loc[dt, picks] = w.values
    return weights
