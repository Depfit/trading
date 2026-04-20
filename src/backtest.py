"""Event-driven backtester: applies target weights next-day-open with costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    metrics: dict

    def summary(self) -> str:
        m = self.metrics
        return (
            f"CAGR:         {m['cagr']:>7.2%}\n"
            f"Vol (ann.):   {m['vol']:>7.2%}\n"
            f"Sharpe:       {m['sharpe']:>7.2f}\n"
            f"Sortino:      {m['sortino']:>7.2f}\n"
            f"Max drawdown: {m['max_dd']:>7.2%}\n"
            f"Calmar:       {m['calmar']:>7.2f}\n"
            f"Hit rate:     {m['hit_rate']:>7.2%}\n"
            f"Avg turnover: {m['avg_turnover']:>7.2%}\n"
            f"Years:        {m['years']:>7.2f}"
        )


def _performance_metrics(returns: pd.Series, rf_annual: float) -> dict:
    returns = returns.dropna()
    years = len(returns) / 252
    equity = (1 + returns).cumprod()
    total = equity.iloc[-1]
    cagr = total ** (1 / years) - 1 if years > 0 else 0.0
    vol = returns.std() * np.sqrt(252)
    excess = returns - rf_annual / 252
    sharpe = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    downside = returns[returns < 0].std()
    sortino = (excess.mean() / downside) * np.sqrt(252) if downside and downside > 0 else 0.0
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    hit_rate = (returns > 0).mean()
    return dict(
        cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino,
        max_dd=max_dd, calmar=calmar, hit_rate=hit_rate, years=years,
    )


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    cost_bps: float = 5.0,
    rf_annual: float = 0.02,
) -> BacktestResult:
    """Simulate daily P&L. Weights apply from the business day AFTER the
    rebalance date, using next-day close-to-close returns. Transaction costs
    charged as ``cost_bps`` per unit of turnover (one-way)."""
    prices = prices.ffill().dropna()
    daily_ret = prices.pct_change().fillna(0.0)

    # Forward-fill weights across daily grid, applied t+1.
    w = target_weights.reindex(prices.index, method=None).ffill().fillna(0.0)
    w_applied = w.shift(1).fillna(0.0)

    # Drift weights between rebalances to reflect actual holdings.
    # For simplicity and conservatism, we approximate gross exposure with the
    # target weights (rebalance-to-target on each rebal date).
    gross = w_applied.sum(axis=1)
    cash_w = (1.0 - gross).clip(lower=0.0)
    cash_ret = rf_annual / 252
    gross_ret = (w_applied * daily_ret).sum(axis=1) + cash_w * cash_ret

    # Turnover = sum of absolute weight changes on rebalance days.
    dw = w_applied.diff().abs().sum(axis=1).fillna(0.0)
    cost = dw * (cost_bps / 10_000.0)
    net_ret = gross_ret - cost

    equity = (1 + net_ret).cumprod()
    metrics = _performance_metrics(net_ret, rf_annual)
    metrics["avg_turnover"] = dw[dw > 0].mean() if (dw > 0).any() else 0.0
    return BacktestResult(equity=equity, returns=net_ret, weights=w,
                          turnover=dw, metrics=metrics)


def buy_and_hold(prices: pd.DataFrame, asset: str, rf_annual: float = 0.02) -> BacktestResult:
    r = prices[asset].pct_change().fillna(0.0)
    eq = (1 + r).cumprod()
    m = _performance_metrics(r, rf_annual)
    m["avg_turnover"] = 0.0
    w = pd.DataFrame({asset: 1.0}, index=prices.index)
    return BacktestResult(equity=eq, returns=r, weights=w,
                          turnover=pd.Series(0.0, index=prices.index), metrics=m)
