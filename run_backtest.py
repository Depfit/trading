"""Demo runner: backtests the dual-momentum strategy vs buy & hold."""
from __future__ import annotations

import argparse
from pathlib import Path

from src import (
    StrategyConfig,
    buy_and_hold,
    compute_target_weights,
    load_csv_universe,
    run_backtest,
    synthetic_universe,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="Folder of CSVs with Date,Close columns (one per asset). "
                         "If omitted, a reproducible synthetic universe is used.")
    ap.add_argument("--cost-bps", type=float, default=5.0,
                    help="One-way transaction cost in basis points.")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--mom-lookback", type=int, default=126)
    ap.add_argument("--trend-lookback", type=int, default=200)
    ap.add_argument("--vol-target", type=float, default=0.12)
    args = ap.parse_args()

    if args.data:
        prices = load_csv_universe(Path(args.data))
        label = f"CSV universe ({len(prices.columns)} assets)"
    else:
        prices = synthetic_universe()
        label = "Synthetic universe (reproducible, seed=7)"

    cfg = StrategyConfig(
        mom_lookback=args.mom_lookback,
        trend_lookback=args.trend_lookback,
        top_n=args.top_n,
        vol_target=args.vol_target,
    )
    weights = compute_target_weights(prices, cfg)
    result = run_backtest(prices, weights, cost_bps=args.cost_bps)

    print(f"=== Dual-Momentum Strategy — {label} ===")
    print(f"Period: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Assets: {', '.join(prices.columns)}")
    print(f"Config: {cfg}")
    print()
    print(result.summary())

    print("\n--- Benchmarks (buy & hold) ---")
    for col in prices.columns:
        m = buy_and_hold(prices, col).metrics
        print(f"  {col:<10} CAGR {m['cagr']:>6.2%}  Sharpe {m['sharpe']:>5.2f}  "
              f"MaxDD {m['max_dd']:>7.2%}")


if __name__ == "__main__":
    main()
