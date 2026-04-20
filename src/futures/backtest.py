"""Bar-by-bar intraday futures backtester with realistic costs, slippage, and
prop-firm-style risk controls. Processes one session at a time, flattens EOD,
and enforces the RiskEngine's halt rules.

Assumptions:
  - Bars are 1-min OHLCV with a naive DatetimeIndex in exchange local time and
    a `session_date` column identifying each trading day.
  - Entry fill: on the bar AFTER the signal, at that bar's open + slippage.
  - Exit fill: the earlier of
      (a) next bar's range touches stop/target → filled at that level ± slip
      (b) EOD flatten at `risk.eod_flatten` → filled at bar close ± slip.
  - If a bar's range contains BOTH stop and target, the stop fills first
    (conservative).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional

import numpy as np
import pandas as pd

from .contracts import ContractSpec
from .risk import RiskConfig, RiskEngine
from .strategies.base import Side, Signal, Strategy


@dataclass
class Trade:
    session_date: str
    side: Side
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    contracts: int
    pnl_dollars: float
    reason_entry: str
    reason_exit: str

    @property
    def points(self) -> float:
        if self.side == Side.LONG:
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    metrics: dict

    def summary(self) -> str:
        m = self.metrics
        return (
            f"Trades:          {m['n_trades']:>7d}\n"
            f"Win rate:        {m['win_rate']:>7.1%}\n"
            f"Avg win ($):     {m['avg_win']:>7.2f}\n"
            f"Avg loss ($):    {m['avg_loss']:>7.2f}\n"
            f"Expectancy ($):  {m['expectancy']:>7.2f}\n"
            f"Profit factor:   {m['profit_factor']:>7.2f}\n"
            f"Total P&L ($):   {m['total_pnl']:>7.2f}\n"
            f"Max DD ($):      {m['max_dd']:>7.2f}\n"
            f"Return on DD:    {m['return_on_dd']:>7.2f}\n"
            f"Sessions:        {m['n_sessions']:>7d}"
        )


@dataclass
class IntradayBacktester:
    spec: ContractSpec
    strategy: Strategy
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    slippage_ticks: int = 1

    def _slip(self, price: float, sign: int) -> float:
        return price + sign * self.slippage_ticks * self.spec.tick_size

    def _exit_pnl(self, side: Side, entry: float, exit_: float,
                  contracts: int) -> float:
        points = (exit_ - entry) if side == Side.LONG else (entry - exit_)
        ticks = points / self.spec.tick_size
        gross = ticks * self.spec.tick_value * contracts
        fees = self.spec.fees_rt * contracts  # round-trip already
        return gross - fees

    def run(self, bars: pd.DataFrame) -> BacktestResult:
        risk = RiskEngine(self.risk_config)
        trades: list[Trade] = []

        pending_signal: Optional[Signal] = None
        open_trade: Optional[dict] = None
        session_equity = []
        cumulative = 0.0

        bars = bars.sort_index()
        for session_date, sbars in bars.groupby("session_date", sort=False):
            self.strategy.on_session_start(session_date)
            risk.on_new_session(str(session_date))
            pending_signal = None
            open_trade = None
            history = pd.DataFrame(columns=sbars.columns)

            # Pre-extract numpy arrays for hot-path speed.
            idx = sbars.index
            opens = sbars["open"].to_numpy()
            highs = sbars["high"].to_numpy()
            lows = sbars["low"].to_numpy()
            closes = sbars["close"].to_numpy()
            n_bars = len(sbars)

            for i in range(n_bars):
                ts = idx[i]
                bar_time = ts.time()
                past_eod = risk.should_flatten_eod(bar_time)
                high = highs[i]
                low = lows[i]
                close = closes[i]

                # 1) Manage open position first: stop/target/EOD.
                if open_trade is not None:
                    stop = open_trade["stop"]
                    tgt = open_trade["target"]
                    exit_price = None
                    exit_reason = ""
                    if open_trade["side"] == Side.LONG:
                        if low <= stop:
                            exit_price = self._slip(stop, -1)
                            exit_reason = "stop"
                        elif high >= tgt:
                            exit_price = self._slip(tgt, -1)
                            exit_reason = "target"
                    else:  # SHORT
                        if high >= stop:
                            exit_price = self._slip(stop, +1)
                            exit_reason = "stop"
                        elif low <= tgt:
                            exit_price = self._slip(tgt, +1)
                            exit_reason = "target"
                    if exit_price is None and past_eod:
                        exit_price = self._slip(
                            float(close),
                            -1 if open_trade["side"] == Side.LONG else +1,
                        )
                        exit_reason = "eod_flatten"
                    if exit_price is not None:
                        pnl = self._exit_pnl(open_trade["side"],
                                             open_trade["entry_price"],
                                             exit_price,
                                             open_trade["contracts"])
                        risk.on_fill(pnl)
                        cumulative += pnl
                        trades.append(Trade(
                            session_date=str(session_date),
                            side=open_trade["side"],
                            entry_time=open_trade["entry_time"],
                            entry_price=open_trade["entry_price"],
                            exit_time=ts.to_pydatetime(),
                            exit_price=exit_price,
                            contracts=open_trade["contracts"],
                            pnl_dollars=pnl,
                            reason_entry=open_trade["reason_entry"],
                            reason_exit=exit_reason,
                        ))
                        open_trade = None

                # 2) Enter if we had a pending signal from previous bar.
                if pending_signal is not None and open_trade is None:
                    ok, _ = risk.can_trade()
                    if ok and not past_eod:
                        sig = pending_signal
                        fill = self._slip(
                            float(opens[i]),
                            +1 if sig.side == Side.LONG else -1,
                        )
                        stop_ticks = int(round(
                            abs(fill - sig.stop_price) / self.spec.tick_size))
                        n = risk.size_trade(stop_ticks, self.spec)
                        if n > 0:
                            open_trade = dict(
                                side=sig.side,
                                entry_time=ts.to_pydatetime(),
                                entry_price=fill,
                                stop=sig.stop_price,
                                target=sig.target_price,
                                contracts=n,
                                reason_entry=sig.reason,
                            )
                    pending_signal = None

                # 3) Generate new signal (for next-bar entry).
                if open_trade is None and not past_eod:
                    ok, _ = risk.can_trade()
                    if ok:
                        # Build a lightweight bar Series with name=ts.
                        bar = sbars.iloc[i]
                        history = sbars.iloc[: i + 1]
                        sig = self.strategy.on_bar(bar, history,
                                                   position_open=False)
                        if sig is not None and sig.side != Side.FLAT:
                            pending_signal = sig

            session_equity.append((session_date, cumulative))

        eq = pd.Series(
            {pd.Timestamp(d): v for d, v in session_equity},
            name="equity",
        )
        return BacktestResult(trades=trades, equity_curve=eq,
                              metrics=self._metrics(trades, eq))

    @staticmethod
    def _metrics(trades: list[Trade], eq: pd.Series) -> dict:
        n = len(trades)
        if n == 0:
            return dict(n_trades=0, win_rate=0, avg_win=0, avg_loss=0,
                        expectancy=0, profit_factor=0, total_pnl=0,
                        max_dd=0, return_on_dd=0, n_sessions=len(eq))
        pnls = np.array([t.pnl_dollars for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        total = float(pnls.sum())
        gross_win = float(wins.sum()) if wins.size else 0.0
        gross_loss = float(-losses.sum()) if losses.size else 0.0
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        peak = eq.cummax()
        dd = (peak - eq).max() if len(eq) else 0.0
        return dict(
            n_trades=n,
            win_rate=float((pnls > 0).mean()),
            avg_win=float(wins.mean()) if wins.size else 0.0,
            avg_loss=float(losses.mean()) if losses.size else 0.0,
            expectancy=float(pnls.mean()),
            profit_factor=pf,
            total_pnl=total,
            max_dd=float(dd),
            return_on_dd=(total / dd) if dd > 0 else float("inf"),
            n_sessions=int(len(eq)),
        )
