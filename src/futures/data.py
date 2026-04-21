"""Load 1-minute OHLCV CSV data for a single futures contract.

Supported CSV schemas (auto-detected):

1. Standard IB / NinjaTrader / Databento format:
       timestamp,open,high,low,close,volume
       2024-01-02 08:30:00,4783.25,4784.00,4782.50,4783.75,521
   Timestamps in exchange local time; no volume in schema → treated as zeros.

2. TradingView export format:
       time,open,high,low,close[,Volume]
       1774821600,6380,6388.25,6372,6377.25[,1234]
   ``time`` is a Unix epoch in seconds (UTC). Converted to the contract's
   exchange local time before session filtering.

A ``session_date`` column is derived from the local date of each bar.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import ContractSpec


_TV_LOCAL_TZ = {
    "America/Chicago": "America/Chicago",
    "America/New_York": "America/New_York",
}


def load_intraday_csv(path: str | Path, spec: ContractSpec,
                      rth_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize timestamp column name.
    ts_col = None
    for cand in ("timestamp", "time", "datetime", "date"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError(f"CSV has no timestamp column. Got: {list(df.columns)}")

    # Detect Unix-epoch (TradingView) vs. datetime string.
    sample = df[ts_col].iloc[0]
    is_unix = False
    try:
        int(sample)
        # plausible Unix seconds range: 1990..2100
        if 6.3e8 <= float(sample) <= 4.1e9:
            is_unix = True
    except (TypeError, ValueError):
        is_unix = False

    if is_unix:
        # TV exports are UTC seconds; convert to exchange-local naive time.
        ts_utc = pd.to_datetime(df[ts_col].astype("int64"), unit="s", utc=True)
        local_tz = _TV_LOCAL_TZ.get(spec.timezone, spec.timezone)
        ts_local = ts_utc.dt.tz_convert(local_tz).dt.tz_localize(None)
        df["timestamp"] = ts_local
    else:
        df["timestamp"] = pd.to_datetime(df[ts_col])

    df = df.set_index("timestamp").sort_index()
    if ts_col in df.columns:
        df = df.drop(columns=[ts_col])

    # Volume is optional (TV exports often omit it).
    if "volume" not in df.columns:
        df["volume"] = 0

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["session_date"] = df.index.date

    if rth_only:
        o_h, o_m = spec.session_open
        c_h, c_m = spec.session_close
        import datetime as _dt
        open_t = _dt.time(o_h, o_m)
        close_t = _dt.time(c_h, c_m)
        t = df.index.time
        mask = (t >= open_t) & (t < close_t)
        df = df[mask]
    return df
