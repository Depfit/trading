"""Walk-forward optimizer.

Splits the data into contiguous `n_folds` folds. For each fold i:
  - Train on fold i:   grid-search parameters, pick the one that maximizes
                       an objective subject to the max-DD constraint.
  - OOS on fold i+1:   report performance of those tuned params on unseen data.

Only OOS metrics are reported in aggregate — this is what the strategy would
have actually delivered had it been redeployed after each refit.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import pandas as pd

from .backtest import BacktestResult, IntradayBacktester
from .contracts import ContractSpec
from .risk import RiskConfig


def default_objective(m: dict, max_dd_cap: float) -> float:
    """Maximize expectancy * trade_count, penalized if DD exceeds cap."""
    if m["n_trades"] < 5:
        return -1e9
    if m["max_dd"] > max_dd_cap:
        return -1e9
    return m["expectancy"] * m["n_trades"]


@dataclass
class WalkForwardResult:
    folds: list[dict]     # per-fold {"params", "is_metrics", "oos_metrics"}
    oos_trades: int
    oos_total_pnl: float
    oos_max_dd: float
    oos_profit_factor: float
    oos_expectancy: float

    def summary(self) -> str:
        return (
            f"OOS trades:        {self.oos_trades:>6d}\n"
            f"OOS total P&L ($): {self.oos_total_pnl:>9.2f}\n"
            f"OOS max DD ($):    {self.oos_max_dd:>9.2f}\n"
            f"OOS profit factor: {self.oos_profit_factor:>6.2f}\n"
            f"OOS expectancy ($): {self.oos_expectancy:>8.2f}\n"
            f"Folds:             {len(self.folds):>6d}"
        )


def _split_folds(bars: pd.DataFrame, n_folds: int) -> list[pd.DataFrame]:
    sessions = bars["session_date"].drop_duplicates().sort_values().tolist()
    size = max(1, len(sessions) // n_folds)
    out = []
    for i in range(n_folds):
        start = i * size
        end = (i + 1) * size if i < n_folds - 1 else len(sessions)
        s = sessions[start:end]
        out.append(bars[bars["session_date"].isin(s)])
    return out


def walk_forward(
    spec: ContractSpec,
    bars: pd.DataFrame,
    strategy_factory: Callable[[dict], object],
    param_grid: dict[str, Iterable],
    *,
    n_folds: int = 5,
    risk_config: RiskConfig | None = None,
    objective: Callable[[dict, float], float] = default_objective,
    max_dd_cap: float | None = None,
) -> WalkForwardResult:
    if risk_config is None:
        risk_config = RiskConfig()
    if max_dd_cap is None:
        max_dd_cap = risk_config.max_dd_dollars

    folds = _split_folds(bars, n_folds)
    keys = list(param_grid.keys())
    grid = [dict(zip(keys, combo)) for combo in product(*param_grid.values())]

    reports: list[dict] = []
    all_oos_trades = []
    oos_equity_running = 0.0
    oos_equity_points = []

    for i in range(len(folds) - 1):
        train = folds[i]
        test = folds[i + 1]

        best_params = None
        best_score = float("-inf")
        best_is_metrics = None
        for params in grid:
            strat = strategy_factory(params)
            bt = IntradayBacktester(spec=spec, strategy=strat,
                                    risk_config=risk_config)
            r = bt.run(train)
            score = objective(r.metrics, max_dd_cap)
            if score > best_score:
                best_score = score
                best_params = params
                best_is_metrics = r.metrics

        # OOS evaluation with best params.
        strat = strategy_factory(best_params)
        bt = IntradayBacktester(spec=spec, strategy=strat,
                                risk_config=risk_config)
        oos = bt.run(test)
        all_oos_trades.extend(oos.trades)
        for t in oos.trades:
            oos_equity_running += t.pnl_dollars
            oos_equity_points.append(oos_equity_running)

        reports.append({
            "params": best_params,
            "is_metrics": best_is_metrics,
            "oos_metrics": oos.metrics,
        })

    # Aggregate OOS metrics from concatenated trades.
    if all_oos_trades:
        pnls = [t.pnl_dollars for t in all_oos_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_win = sum(wins)
        gross_loss = -sum(losses)
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        equity = pd.Series(oos_equity_points)
        dd = (equity.cummax() - equity).max()
    else:
        pnls = []
        pf = 0.0
        dd = 0.0

    return WalkForwardResult(
        folds=reports,
        oos_trades=len(all_oos_trades),
        oos_total_pnl=float(sum(pnls)) if pnls else 0.0,
        oos_max_dd=float(dd) if pnls else 0.0,
        oos_profit_factor=float(pf),
        oos_expectancy=float(sum(pnls) / len(pnls)) if pnls else 0.0,
    )
