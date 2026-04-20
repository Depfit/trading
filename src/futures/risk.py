"""Hard mechanical risk controls for prop-firm-style accounts.

Three independent guardrails:
  1. Per-trade sizing: contracts = floor(per_trade_risk / (stop_ticks * tick_value))
  2. Daily loss limit: once cumulative daily P&L hits the limit, trading halts
     for the rest of the session.
  3. Max drawdown circuit breaker: if equity from peak exceeds `max_dd_warn`,
     force size reduction; if it exceeds `max_dd_halt`, halt all new trades.
  4. EOD flatten: any open position is closed at `eod_flatten` (exchange local
     time) regardless of strategy signal.

The engine is stateful and meant to be queried before every order submission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from math import floor

from .contracts import ContractSpec


@dataclass
class RiskConfig:
    account_equity: float = 50_000.0
    max_dd_dollars: float = 2_500.0        # hard stop; must never be breached
    max_dd_warn_frac: float = 0.60         # at 60% of max DD, halve size
    max_dd_halt_frac: float = 0.80         # at 80% of max DD, halt
    daily_loss_limit: float = 500.0        # halt trading if today's P&L <= -this
    per_trade_risk: float = 100.0          # max $ at risk per trade
    eod_flatten: time = time(14, 55)       # CT for CME index; override per contract
    max_contracts: int = 10                # absolute cap even if math allows more


@dataclass
class RiskState:
    equity: float = 0.0                    # cumulative realized + unrealized P&L
    peak_equity: float = 0.0
    today_pnl: float = 0.0
    today_date: str = ""
    halted_today: bool = False
    halted_permanently: bool = False


@dataclass
class RiskEngine:
    config: RiskConfig
    state: RiskState = field(default_factory=RiskState)

    # ---- lifecycle ---------------------------------------------------------
    def on_new_session(self, session_date: str) -> None:
        if self.state.today_date != session_date:
            self.state.today_date = session_date
            self.state.today_pnl = 0.0
            self.state.halted_today = False

    def on_fill(self, pnl_dollars: float) -> None:
        self.state.equity += pnl_dollars
        self.state.today_pnl += pnl_dollars
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

    # ---- queries -----------------------------------------------------------
    @property
    def current_drawdown(self) -> float:
        return max(0.0, self.state.peak_equity - self.state.equity)

    def dd_exceeds(self, frac: float) -> bool:
        return self.current_drawdown >= frac * self.config.max_dd_dollars

    def can_trade(self) -> tuple[bool, str]:
        if self.state.halted_permanently:
            return False, "permanent halt (max DD breached)"
        if self.dd_exceeds(self.config.max_dd_halt_frac):
            self.state.halted_permanently = True
            return False, f"max DD circuit breaker at {self.config.max_dd_halt_frac:.0%}"
        if self.state.today_pnl <= -self.config.daily_loss_limit:
            self.state.halted_today = True
            return False, f"daily loss limit hit ({self.state.today_pnl:+.2f})"
        if self.state.halted_today:
            return False, "halted for the rest of today"
        return True, ""

    def size_trade(self, stop_distance_ticks: int, spec: ContractSpec) -> int:
        """Return # contracts to trade so risk <= per_trade_risk."""
        if stop_distance_ticks <= 0:
            return 0
        risk_per_contract = stop_distance_ticks * spec.tick_value
        budget = self.config.per_trade_risk
        if self.dd_exceeds(self.config.max_dd_warn_frac):
            budget *= 0.5  # defensive sizing after drawdown warning
        n = floor(budget / risk_per_contract)
        return max(0, min(n, self.config.max_contracts))

    def should_flatten_eod(self, now_time: time) -> bool:
        return now_time >= self.config.eod_flatten
