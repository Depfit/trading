"""Load 1-minute OHLCV CSV data for a single futures contract.

Expected CSV schema (common to IB/NinjaTrader/Databento exports):
    timestamp,open,high,low,close,volume
    2024-01-02 08:30:00,4783.25,4784.00,4782.50,4783.75,521
    ...

The ``timestamp`` column must be in the contract's exchange local time. A
``session_date`` column is derived from the date part.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import ContractSpec


def load_intraday_csv(path: str | Path, spec: ContractSpec,
                      rth_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        # fall back to common name variants
        for cand in ("Datetime", "Date", "time"):
            if cand in df.columns:
                df = df.rename(columns={cand: "timestamp"})
                break
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["session_date"] = df.index.date

    if rth_only:
        o_h, o_m = spec.session_open
        c_h, c_m = spec.session_close
        t = df.index.time
        import datetime as _dt
        open_t = _dt.time(o_h, o_m)
        close_t = _dt.time(c_h, c_m)
        mask = (t >= open_t) & (t < close_t)
        df = df[mask]
    return df
