"""CLI: iterate profitable intraday futures strategies contract-by-contract.

Examples:
  # Demo on reproducible synthetic data (no external market feed required):
  PYTHONPATH=. python3 run_futures.py --contract MES --strategy orb --synthetic

  # Walk-forward tune on real 1-min CSV:
  PYTHONPATH=. python3 run_futures.py --contract MES --strategy orb \
      --csv ./data/MES_1min_2022_2024.csv --walk-forward

  # Same strategy, next contract (pipeline step 2):
  PYTHONPATH=. python3 run_futures.py --contract MNQ --strategy orb \
      --csv ./data/MNQ_1min_2022_2024.csv --walk-forward
"""
from __future__ import annotations

import argparse
from datetime import time as dtime

from src.futures import (
    IntradayBacktester,
    OpeningRangeBreakout,
    RiskConfig,
    VwapMeanReversion,
    get,
    synthetic_intraday,
)
from src.futures.data import load_intraday_csv
from src.futures.strategies.orb import ORBConfig
from src.futures.strategies.vwap_mr import VwapMRConfig
from src.futures.walk_forward import walk_forward


STRATEGIES = {"orb", "vwap_mr"}


def build_strategy(name: str, spec, params: dict | None = None):
    params = params or {}
    if name == "orb":
        return OpeningRangeBreakout(spec=spec, config=ORBConfig(**params))
    if name == "vwap_mr":
        return VwapMeanReversion(spec=spec, config=VwapMRConfig(**params))
    raise ValueError(name)


def default_grid(name: str) -> dict:
    """Small default grids that run in reasonable time on a single CPU.
    Users with real data can widen these and run overnight."""
    if name == "orb":
        # Grid is contract-agnostic by being broad. Walk-forward picks the
        # subset that fits the contract's typical OR width and tick value.
        return {
            "or_minutes": [15, 30],
            "stop_buffer_ticks": [2, 4],
            "rr_multiple": [1.5, 2.0],
            "max_or_ticks": [80, 200, 500, 1200],
        }
    if name == "vwap_mr":
        return {
            "warmup_minutes": [30, 60],
            "k_sigma": [2.0, 2.5],
            "stop_buffer_ticks": [4, 8],
        }
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True,
                   help="Symbol: MES, MNQ, M2K, MCL, MGC, ES, NQ")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    p.add_argument("--csv", type=str, default=None,
                   help="1-min OHLCV CSV. Omit to use synthetic data.")
    p.add_argument("--synthetic", action="store_true",
                   help="Force synthetic data (reproducible, seed=42).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-days", type=int, default=500,
                   help="Synthetic data length.")
    p.add_argument("--walk-forward", action="store_true",
                   help="Run walk-forward tuning + OOS report.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--max-dd", type=float, default=2500.0)
    p.add_argument("--daily-loss-limit", type=float, default=500.0)
    p.add_argument("--per-trade-risk", type=float, default=100.0)
    p.add_argument("--eod-flatten", type=str, default="14:55",
                   help="Exchange local HH:MM to flatten all positions.")
    args = p.parse_args()

    spec = get(args.contract)
    eod_h, eod_m = map(int, args.eod_flatten.split(":"))
    rc = RiskConfig(
        max_dd_dollars=args.max_dd,
        daily_loss_limit=args.daily_loss_limit,
        per_trade_risk=args.per_trade_risk,
        eod_flatten=dtime(eod_h, eod_m),
    )

    # Data
    if args.csv and not args.synthetic:
        bars = load_intraday_csv(args.csv, spec)
        label = f"CSV {args.csv}"
    else:
        start_price = {"MES": 4800, "ES": 4800, "MNQ": 16500, "NQ": 16500,
                       "M2K": 2000, "MCL": 75.0, "MGC": 2050.0}.get(
            spec.symbol, 100.0)
        bars = synthetic_intraday(spec, n_days=args.n_days, seed=args.seed,
                                  start_price=start_price)
        label = f"synthetic (seed={args.seed}, {args.n_days} sessions)"

    print(f"=== {spec.symbol} ({spec.name}) — {label} ===")
    print(f"Risk: max DD ${rc.max_dd_dollars:.0f}, daily limit "
          f"${rc.daily_loss_limit:.0f}, per-trade ${rc.per_trade_risk:.0f}, "
          f"EOD flat {rc.eod_flatten.strftime('%H:%M')} local")
    print(f"Bars: {len(bars):,}   Sessions: {bars['session_date'].nunique()}")
    print()

    if args.walk_forward:
        grid = default_grid(args.strategy)
        result = walk_forward(
            spec=spec, bars=bars,
            strategy_factory=lambda params: build_strategy(
                args.strategy, spec, params),
            param_grid=grid,
            n_folds=args.n_folds,
            risk_config=rc,
            max_dd_cap=args.max_dd,
        )
        print(f"--- Walk-forward ({args.n_folds} folds) ---")
        print(result.summary())
        print("\nPer-fold best params (trained on fold i, OOS on i+1):")
        for i, r in enumerate(result.folds):
            oos = r["oos_metrics"]
            print(f"  fold {i}: params={r['params']}")
            print(f"          OOS trades={oos['n_trades']} "
                  f"PF={oos['profit_factor']:.2f} "
                  f"P&L=${oos['total_pnl']:.2f} "
                  f"DD=${oos['max_dd']:.2f}")
    else:
        strat = build_strategy(args.strategy, spec)
        bt = IntradayBacktester(spec=spec, strategy=strat, risk_config=rc)
        r = bt.run(bars)
        print(f"--- Single run: {args.strategy} (default params) ---")
        print(r.summary())


if __name__ == "__main__":
    main()
