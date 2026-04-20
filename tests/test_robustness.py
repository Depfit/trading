"""Robustness check: run the strategy on many random synthetic universes and
confirm it produces a positive risk-adjusted edge on average."""
from __future__ import annotations

import statistics

from src import (
    StrategyConfig,
    buy_and_hold,
    compute_target_weights,
    run_backtest,
    synthetic_universe,
)

SEEDS = list(range(1, 31))


def run_one(seed: int) -> dict:
    prices = synthetic_universe(seed=seed)
    cfg = StrategyConfig()
    weights = compute_target_weights(prices, cfg)
    strat = run_backtest(prices, weights, cost_bps=5.0).metrics
    bh = buy_and_hold(prices, "EQ_BROAD").metrics
    return {
        "seed": seed,
        "strat_cagr": strat["cagr"],
        "strat_sharpe": strat["sharpe"],
        "strat_dd": strat["max_dd"],
        "bh_cagr": bh["cagr"],
        "bh_sharpe": bh["sharpe"],
        "bh_dd": bh["max_dd"],
    }


def main() -> None:
    rows = [run_one(s) for s in SEEDS]
    for r in rows:
        print(f"seed={r['seed']:3d}  "
              f"strat CAGR {r['strat_cagr']:>7.2%} Sharpe {r['strat_sharpe']:>5.2f} DD {r['strat_dd']:>7.2%}  |  "
              f"BH CAGR {r['bh_cagr']:>7.2%} Sharpe {r['bh_sharpe']:>5.2f} DD {r['bh_dd']:>7.2%}")
    print("\n=== Aggregate over", len(rows), "seeds ===")
    for key in ("strat_cagr", "strat_sharpe", "strat_dd",
                "bh_cagr", "bh_sharpe", "bh_dd"):
        vals = [r[key] for r in rows]
        print(f"  {key:<13} mean {statistics.mean(vals):>7.3f}  median {statistics.median(vals):>7.3f}")
    wins_cagr = sum(1 for r in rows if r["strat_cagr"] > r["bh_cagr"])
    wins_sharpe = sum(1 for r in rows if r["strat_sharpe"] > r["bh_sharpe"])
    better_dd = sum(1 for r in rows if r["strat_dd"] > r["bh_dd"])  # less negative
    print(f"  Strategy beats BH on CAGR:   {wins_cagr}/{len(rows)}")
    print(f"  Strategy beats BH on Sharpe: {wins_sharpe}/{len(rows)}")
    print(f"  Strategy has shallower DD:   {better_dd}/{len(rows)}")


if __name__ == "__main__":
    main()
