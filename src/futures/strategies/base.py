"""Strategy base class: bar-by-bar signal generation interface."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

import pandas as pd


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    side: Side
    entry_price: float
    stop_price: float
    target_price: float
    reason: str = ""

    @property
    def stop_distance(self) -> float:
        return abs(self.entry_price - self.stop_price)


class Strategy(Protocol):
    """Pure-function style: given the session's bars so far, return a signal
    (or None). The backtester handles entries, exits, risk sizing, and EOD
    flatten — the strategy only produces trade ideas.
    """

    name: str

    def on_session_start(self, session_date) -> None: ...

    def on_bar(self, bar: pd.Series, history: pd.DataFrame,
               position_open: bool) -> Optional[Signal]: ...
