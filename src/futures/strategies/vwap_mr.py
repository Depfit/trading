"""VWAP Mean Reversion.

Logic:
  1. Compute session VWAP from cumulative (price*volume) / cumulative volume
     starting at RTH open.
  2. Compute rolling intraday standard deviation of close around VWAP (anchored
     to session start) as a band proxy.
  3. After `warmup_minutes` (so VWAP is stable), enter:
     - LONG when close <= VWAP - k_sigma * band AND close > session_low +
       `reclaim_ticks` (simple reclaim confirmation).
     - SHORT when close >= VWAP + k_sigma * band AND close < session_high -
       `reclaim_ticks`.
  4. Stop = session extreme in the signal's direction ± buffer.
  5. Target = VWAP.

This targets range/chop days which complement ORB (which needs trend days).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Optional

import pandas as pd

from ..contracts import ContractSpec
from .base import Signal, Side, Strategy


@dataclass
class VwapMRConfig:
    warmup_minutes: int = 30
    k_sigma: float = 1.8
    reclaim_ticks: int = 2
    stop_buffer_ticks: int = 3


@dataclass
class VwapMeanReversion:
    spec: ContractSpec
    config: VwapMRConfig = field(default_factory=VwapMRConfig)
    name: str = "VWAP-MR"

    def __post_init__(self) -> None:
        self._session_open: time = time(*self.spec.session_open)
        self._traded_today: bool = False

    def on_session_start(self, session_date) -> None:
        self._traded_today = False

    def on_bar(self, bar: pd.Series, history: pd.DataFrame,
               position_open: bool) -> Optional[Signal]:
        if self._traded_today or position_open:
            return None
        if len(history) < self.config.warmup_minutes:
            return None

        ts = bar.name
        session_minute = (self._session_open.hour * 60 + self._session_open.minute)
        mins_since_open = ts.hour * 60 + ts.minute - session_minute
        if mins_since_open < self.config.warmup_minutes:
            return None

        sess = history  # bars so far this session
        typical = (sess["high"] + sess["low"] + sess["close"]) / 3.0
        vol_sum = sess["volume"].cumsum().iloc[-1]
        if vol_sum <= 0:
            return None
        vwap = (typical * sess["volume"]).cumsum().iloc[-1] / vol_sum
        band = (sess["close"] - vwap).std()
        if not band or band <= 0:
            return None

        close = float(bar["close"])
        sess_high = float(sess["high"].max())
        sess_low = float(sess["low"].min())
        tick = self.spec.tick_size
        buf = self.config.stop_buffer_ticks * tick
        reclaim = self.config.reclaim_ticks * tick

        # LONG mean reversion
        if close <= vwap - self.config.k_sigma * band and close > sess_low + reclaim:
            entry = close
            stop = sess_low - buf
            risk = entry - stop
            if risk <= 0:
                return None
            target = float(vwap)
            if target <= entry:
                return None
            self._traded_today = True
            return Signal(Side.LONG, entry, stop, target,
                          reason=f"VWAP-MR long: {close:.2f}<={vwap:.2f}-{self.config.k_sigma}σ")

        # SHORT mean reversion
        if close >= vwap + self.config.k_sigma * band and close < sess_high - reclaim:
            entry = close
            stop = sess_high + buf
            risk = stop - entry
            if risk <= 0:
                return None
            target = float(vwap)
            if target >= entry:
                return None
            self._traded_today = True
            return Signal(Side.SHORT, entry, stop, target,
                          reason=f"VWAP-MR short: {close:.2f}>={vwap:.2f}+{self.config.k_sigma}σ")

        return None
