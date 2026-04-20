"""Deterministic intraday bar generator for framework validation.

Generates 1-minute OHLCV bars during RTH for N trading days, with realistic
features that give intraday strategies something to work with:
  - day-type regimes (trend / range / reversal) so both breakout and
    mean-reversion logic can show their strengths/weaknesses
  - opening volatility burst, midday lull, closing push
  - overnight gaps
  - realistic tick-quantized prices (snapped to contract tick_size)

This is for framework demo only. Real validation requires real market data.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .contracts import ContractSpec


def _snap(x: float, tick: float) -> float:
    return round(x / tick) * tick


def synthetic_intraday(
    spec: ContractSpec,
    n_days: int = 250,
    start_date: str = "2024-01-02",
    seed: int = 42,
    start_price: float = 4800.0,
    base_vol_per_min: float = 0.0009,
) -> pd.DataFrame:
    """Return a DataFrame of 1-min bars with MultiIndex (date, timestamp).

    Columns: open, high, low, close, volume. Index is a DatetimeIndex in
    exchange local time (naive). Only RTH bars are produced.
    """
    rng = np.random.default_rng(seed)
    o_h, o_m = spec.session_open
    c_h, c_m = spec.session_close
    open_t = datetime.strptime(f"{o_h:02d}:{o_m:02d}", "%H:%M").time()
    close_t = datetime.strptime(f"{c_h:02d}:{c_m:02d}", "%H:%M").time()

    dates = pd.bdate_range(start=start_date, periods=n_days)
    all_bars = []
    last_close = start_price

    for d in dates:
        day_type = rng.choice(["trend_up", "trend_dn", "range", "reversal"],
                              p=[0.30, 0.25, 0.30, 0.15])
        gap = rng.normal(0, 0.0015) * last_close
        o = _snap(last_close + gap, spec.tick_size)

        minutes = int(((close_t.hour * 60 + close_t.minute)
                       - (open_t.hour * 60 + open_t.minute)))
        timestamps = pd.date_range(
            start=datetime.combine(d.date(), open_t),
            periods=minutes, freq="1min",
        )

        # Build a per-minute drift profile.
        t = np.arange(minutes) / max(minutes - 1, 1)
        vol_profile = 1.3 - 1.0 * np.abs(t - 0.5) * 0.6 + 0.2 * (t > 0.9)
        vol = base_vol_per_min * vol_profile

        if day_type == "trend_up":
            drift = np.linspace(0.0002, 0.0006, minutes)
        elif day_type == "trend_dn":
            drift = np.linspace(-0.0002, -0.0006, minutes)
        elif day_type == "range":
            # Mean-reverting O-U around the day's open.
            drift = np.zeros(minutes)
        else:  # reversal: push one way, reverse midday
            sign = rng.choice([-1, 1])
            drift = np.where(t < 0.4, sign * 0.0005,
                             np.where(t < 0.7, 0.0, -sign * 0.0004))

        returns = rng.normal(drift, vol)
        if day_type == "range":
            # AR(1) mean reversion to open.
            prices = np.zeros(minutes)
            prices[0] = o
            for i in range(1, minutes):
                prices[i] = prices[i - 1] + rng.normal(0, vol[i] * o) \
                    - 0.015 * (prices[i - 1] - o)
            closes = prices
        else:
            closes = o * np.cumprod(1 + returns)

        # OHLC within each minute from intra-bar noise.
        noise_scale = vol * closes * 0.6
        highs = closes + np.abs(rng.normal(0, noise_scale))
        lows = closes - np.abs(rng.normal(0, noise_scale))
        opens = np.concatenate([[o], closes[:-1]])
        highs = np.maximum.reduce([highs, opens, closes])
        lows = np.minimum.reduce([lows, opens, closes])

        opens = np.vectorize(lambda v: _snap(v, spec.tick_size))(opens)
        highs = np.vectorize(lambda v: _snap(v, spec.tick_size))(highs)
        lows = np.vectorize(lambda v: _snap(v, spec.tick_size))(lows)
        closes = np.vectorize(lambda v: _snap(v, spec.tick_size))(closes)
        volumes = rng.integers(200, 3000, size=minutes)

        bars = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        }, index=timestamps)
        bars["session_date"] = d.date()
        all_bars.append(bars)
        last_close = closes[-1]

    out = pd.concat(all_bars)
    out.index.name = "timestamp"
    return out
