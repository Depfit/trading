"""Smoke tests for the futures framework.

Not an exhaustive suite — these verify the critical invariants:
  - Risk engine halts when DD cap is breached and enforces daily limits.
  - Backtester always closes positions by EOD (no overnight holds).
  - Strategy signals produce valid stop/target prices.
  - Walk-forward keeps train and OOS partitions disjoint.
"""
from __future__ import annotations

from datetime import date, datetime, time as dtime

import pandas as pd

from src.futures import (
    IntradayBacktester,
    OpeningRangeBreakout,
    RiskConfig,
    RiskEngine,
    VwapMeanReversion,
    get,
    synthetic_intraday,
)
from src.futures.strategies.base import Signal, Side
from src.futures.walk_forward import _split_folds, walk_forward
from run_futures import build_strategy, default_grid


def test_risk_halts_at_dd_cap():
    rc = RiskConfig(max_dd_dollars=1000, max_dd_halt_frac=0.8)
    re_ = RiskEngine(rc)
    re_.on_new_session("2024-01-02")
    re_.on_fill(+500)       # new high
    re_.on_fill(-900)       # DD = 900 (90% of cap)
    ok, reason = re_.can_trade()
    assert not ok, reason
    assert "max DD" in reason


def test_risk_daily_loss_limit():
    rc = RiskConfig(daily_loss_limit=300)
    re_ = RiskEngine(rc)
    re_.on_new_session("2024-01-02")
    re_.on_fill(-310)
    ok, reason = re_.can_trade()
    assert not ok
    assert "daily" in reason


def test_risk_sizing_respects_budget():
    rc = RiskConfig(per_trade_risk=100)
    re_ = RiskEngine(rc)
    spec = get("MES")
    # stop 20 ticks → $25/contract risk → floor(100/25) = 4
    assert re_.size_trade(20, spec) == 4
    # stop 1 tick → $1.25/contract → would be 80, but capped at max_contracts
    assert re_.size_trade(1, spec) == rc.max_contracts


def test_eod_flatten_is_enforced():
    """Every trade's exit must be at or before the EOD flatten time."""
    spec = get("MES")
    bars = synthetic_intraday(spec, n_days=30, seed=1, start_price=4800)
    rc = RiskConfig(eod_flatten=dtime(14, 55))
    bt = IntradayBacktester(spec=spec, strategy=OpeningRangeBreakout(spec=spec),
                            risk_config=rc)
    result = bt.run(bars)
    for t in result.trades:
        assert t.exit_time.time() <= dtime(15, 0), \
            f"trade exited after 15:00: {t.exit_time}"


def test_backtester_never_breaches_dd_cap():
    """A correctly configured engine must never realize DD > max_dd_dollars."""
    spec = get("MES")
    bars = synthetic_intraday(spec, n_days=60, seed=2, start_price=4800)
    rc = RiskConfig(max_dd_dollars=2500, per_trade_risk=100)
    bt = IntradayBacktester(spec=spec, strategy=OpeningRangeBreakout(spec=spec),
                            risk_config=rc)
    result = bt.run(bars)
    assert result.metrics["max_dd"] <= rc.max_dd_dollars + 1e-6


def test_strategy_signals_are_well_formed():
    spec = get("MES")
    bars = synthetic_intraday(spec, n_days=10, seed=3, start_price=4800)
    for strat_cls in (OpeningRangeBreakout, VwapMeanReversion):
        strat = strat_cls(spec=spec)
        strat.on_session_start(date(2024, 1, 2))
        day = bars[bars["session_date"] == sorted(bars["session_date"].unique())[0]]
        for i in range(len(day)):
            sig = strat.on_bar(day.iloc[i], day.iloc[: i + 1],
                               position_open=False)
            if sig is None:
                continue
            assert sig.entry_price > 0
            if sig.side == Side.LONG:
                assert sig.stop_price < sig.entry_price < sig.target_price
            elif sig.side == Side.SHORT:
                assert sig.stop_price > sig.entry_price > sig.target_price


def test_walk_forward_fold_disjointness():
    spec = get("MES")
    bars = synthetic_intraday(spec, n_days=60, seed=4, start_price=4800)
    folds = _split_folds(bars, 4)
    session_sets = [set(f["session_date"].unique()) for f in folds]
    for i in range(len(session_sets)):
        for j in range(i + 1, len(session_sets)):
            assert session_sets[i].isdisjoint(session_sets[j])


def test_walk_forward_smoke():
    spec = get("MES")
    bars = synthetic_intraday(spec, n_days=100, seed=5, start_price=4800)
    rc = RiskConfig()
    result = walk_forward(
        spec=spec, bars=bars,
        strategy_factory=lambda p: build_strategy("orb", spec, p),
        param_grid={"or_minutes": [15, 30], "rr_multiple": [1.5, 2.0]},
        n_folds=3, risk_config=rc,
    )
    assert len(result.folds) == 2           # (n_folds-1) train/oos splits
    for fold in result.folds:
        assert "params" in fold
        assert "oos_metrics" in fold


if __name__ == "__main__":
    import traceback, sys
    failures = 0
    for name in [n for n in dir() if n.startswith("test_")]:
        try:
            globals()[name]()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}: {e}")
        except Exception:
            failures += 1
            print(f"  ERROR {name}:")
            traceback.print_exc()
    print(f"\n{len(dir())} names checked. {failures} failure(s).")
    sys.exit(1 if failures else 0)
