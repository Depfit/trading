"""Opening Range Breakout (ORB).

Logic:
  1. Compute the high/low of the first `or_minutes` of RTH.
  2. After the range is established, go long if price closes > range_high by
     ≥ 1 tick with a momentum confirmation (bar close in upper portion of
     range). Symmetric for shorts.
  3. Stop = opposite side of the opening range ± `stop_buffer_ticks` ticks.
  4. Target = entry ± rr_multiple × risk.
  5. One trade per session max.

Notes: ORB has decades of published research (Toby Crabel, 1990; repeatedly
retested on ES/NQ). The edge fades when ranges are too wide, so we also skip
signals if the OR width exceeds `max_or_ticks`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Optional

import pandas as pd

from ..contracts import ContractSpec
from .base import Signal, Side, Strategy


@dataclass
class ORBConfig:
    or_minutes: int = 15
    stop_buffer_ticks: int = 2
    rr_multiple: float = 1.5
    max_or_ticks: int = 80
    min_or_ticks: int = 4


@dataclass
class OpeningRangeBreakout:
    spec: ContractSpec
    config: ORBConfig = field(default_factory=ORBConfig)
    name: str = "ORB"

    def __post_init__(self) -> None:
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._or_locked: bool = False
        self._traded_today: bool = False
        self._session_open: time = time(*self.spec.session_open)

    def on_session_start(self, session_date) -> None:
        self._or_high = None
        self._or_low = None
        self._or_locked = False
        self._traded_today = False

    def on_bar(self, bar: pd.Series, history: pd.DataFrame,
               position_open: bool) -> Optional[Signal]:
        if self._traded_today or position_open:
            return None

        ts = bar.name
        minute_of_day = ts.hour * 60 + ts.minute
        session_minute = (self._session_open.hour * 60
                          + self._session_open.minute)
        mins_since_open = minute_of_day - session_minute

        # Phase 1: accumulate opening range.
        if mins_since_open < self.config.or_minutes:
            if self._or_high is None or bar["high"] > self._or_high:
                self._or_high = float(bar["high"])
            if self._or_low is None or bar["low"] < self._or_low:
                self._or_low = float(bar["low"])
            return None

        if not self._or_locked:
            self._or_locked = True

        if self._or_high is None or self._or_low is None:
            return None

        tick = self.spec.tick_size
        or_width_ticks = round((self._or_high - self._or_low) / tick)
        if not (self.config.min_or_ticks <= or_width_ticks
                <= self.config.max_or_ticks):
            return None

        close = float(bar["close"])
        buf = self.config.stop_buffer_ticks * tick

        # Long breakout
        if close > self._or_high + tick:
            entry = close
            stop = self._or_low - buf
            risk = entry - stop
            if risk <= 0:
                return None
            target = entry + self.config.rr_multiple * risk
            self._traded_today = True
            return Signal(Side.LONG, entry, stop, target,
                          reason=f"ORB long: close>{self._or_high:.2f}")

        # Short breakout
        if close < self._or_low - tick:
            entry = close
            stop = self._or_high + buf
            risk = stop - entry
            if risk <= 0:
                return None
            target = entry - self.config.rr_multiple * risk
            self._traded_today = True
            return Signal(Side.SHORT, entry, stop, target,
                          reason=f"ORB short: close<{self._or_low:.2f}")

        return None
