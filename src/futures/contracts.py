"""Contract specifications for commonly traded CME/NYMEX/COMEX futures.

Values are the on-exchange specs needed for P&L and risk accounting:
  - tick_size: smallest price increment
  - tick_value: $ P&L per 1 contract per 1 tick
  - point_value: tick_value / tick_size (convenience)
  - fees_rt: round-trip commission + exchange/NFA fees estimate (USD)
  - session: regular-session open/close in exchange local time (CT for CME)
  - timezone: exchange timezone

All times are stored as (hour, minute) in 24h format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    name: str
    tick_size: float
    tick_value: float
    fees_rt: float
    session_open: Tuple[int, int]   # RTH open in exchange local time
    session_close: Tuple[int, int]  # RTH close
    timezone: str

    @property
    def point_value(self) -> float:
        return self.tick_value / self.tick_size


# CME Equity Index micros and full-size.
# Times are US Central (CME local). ES/NQ RTH is 08:30-15:00 CT.
CONTRACTS: dict[str, ContractSpec] = {
    "MES": ContractSpec("MES", "Micro E-mini S&P 500",
                        tick_size=0.25, tick_value=1.25, fees_rt=0.88,
                        session_open=(8, 30), session_close=(15, 0), timezone="America/Chicago"),
    "ES":  ContractSpec("ES",  "E-mini S&P 500",
                        tick_size=0.25, tick_value=12.50, fees_rt=4.60,
                        session_open=(8, 30), session_close=(15, 0), timezone="America/Chicago"),
    "MNQ": ContractSpec("MNQ", "Micro E-mini Nasdaq-100",
                        tick_size=0.25, tick_value=0.50, fees_rt=0.88,
                        session_open=(8, 30), session_close=(15, 0), timezone="America/Chicago"),
    "NQ":  ContractSpec("NQ",  "E-mini Nasdaq-100",
                        tick_size=0.25, tick_value=5.00, fees_rt=4.60,
                        session_open=(8, 30), session_close=(15, 0), timezone="America/Chicago"),
    "M2K": ContractSpec("M2K", "Micro E-mini Russell 2000",
                        tick_size=0.10, tick_value=0.50, fees_rt=0.88,
                        session_open=(8, 30), session_close=(15, 0), timezone="America/Chicago"),
    # NYMEX/COMEX micros
    "MCL": ContractSpec("MCL", "Micro Crude Oil",
                        tick_size=0.01, tick_value=1.00, fees_rt=1.50,
                        session_open=(8, 0), session_close=(13, 30), timezone="America/New_York"),
    "MGC": ContractSpec("MGC", "Micro Gold",
                        tick_size=0.10, tick_value=1.00, fees_rt=1.50,
                        session_open=(8, 20), session_close=(13, 30), timezone="America/New_York"),
}


def get(symbol: str) -> ContractSpec:
    key = symbol.upper()
    if key not in CONTRACTS:
        raise KeyError(f"Unknown contract {symbol!r}. Known: {list(CONTRACTS)}")
    return CONTRACTS[key]
