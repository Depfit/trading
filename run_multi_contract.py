"""Iterate a strategy across multiple futures contracts.

For each contract in the list:
  1. Walk-forward tune on in-sample folds
  2. Report OOS metrics
  3. Verify the OOS drawdown respects the account's max-DD budget
  4. Keep the strategy+params only if it passes the DD gate AND has positive
     OOS expectancy

At the end, print a ranked table of survivors. These are the candidates worth
paper-trading / forward-testing before going live.

Usage:
  # All synthetic:
  PYTHONPATH=. python3 run_multi_contract.py --strategy orb \
      --contracts MES,MNQ,M2K

  # Mixed real+synthetic (real CSVs for contracts you have):
  PYTHONPATH=. python3 run_multi_contract.py --strategy orb \
      --contracts MES,MNQ \
      --csv-template "./data/{contract}_1min.csv"
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import time as dtime
from pathlib import Path

from src.futures import (
    RiskConfig,
    get,
    synthetic_intraday,
)
from src.futures.data import load_intraday_csv
from src.futures.walk_forward import walk_forward
from run_futures import build_strategy, default_grid


@dataclass
class ContractReport:
    contract: str
    params: dict
    oos_trades: int
    oos_total_pnl: float
    oos_max_dd: float
    oos_profit_factor: float
    oos_expectancy: float
    passed: bool
    reason: str


def run_one(contract: str, strategy_name: str, rc: RiskConfig, *,
            csv_template: str | None, n_days: int, seed: int,
            n_folds: int, per_trade_risks: list[float]) -> ContractReport:
    spec = get(contract)

    # Data.
    csv_path = None
    if csv_template:
        p = Path(csv_template.format(contract=contract))
        if p.exists():
            csv_path = p
    if csv_path:
        bars = load_intraday_csv(csv_path, spec)
        src_label = f"CSV {csv_path}"
    else:
        start_price = {"MES": 4800, "ES": 4800, "MNQ": 16500, "NQ": 16500,
                       "M2K": 2000, "MCL": 75.0, "MGC": 2050.0}.get(
            spec.symbol, 100.0)
        bars = synthetic_intraday(spec, n_days=n_days, seed=seed,
                                  start_price=start_price)
        src_label = f"synthetic ({n_days} sessions, seed={seed})"

    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {spec.symbol} ({spec.name}) — {src_label}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Try escalating per-trade-risk values to find the smallest budget that
    # produces trades + positive OOS expectancy while honoring max-DD cap.
    best: ContractReport | None = None
    for ptr in per_trade_risks:
        rc_try = RiskConfig(**{**rc.__dict__, "per_trade_risk": ptr})
        grid = default_grid(strategy_name)
        result = walk_forward(
            spec=spec, bars=bars,
            strategy_factory=lambda params, s=spec: build_strategy(
                strategy_name, s, params),
            param_grid=grid,
            n_folds=n_folds,
            risk_config=rc_try,
            max_dd_cap=rc_try.max_dd_dollars,
        )
        # Pick most common param set across folds.
        from collections import Counter
        pc = Counter(tuple(sorted(r["params"].items())) for r in result.folds)
        common_params = dict(pc.most_common(1)[0][0]) if pc else {}

        passed = (result.oos_trades >= max(5, n_folds)
                  and result.oos_expectancy > 0
                  and result.oos_max_dd <= rc_try.max_dd_dollars)
        reason = []
        if result.oos_trades < max(5, n_folds):
            reason.append("too few OOS trades")
        if result.oos_expectancy <= 0:
            reason.append("non-positive OOS expectancy")
        if result.oos_max_dd > rc_try.max_dd_dollars:
            reason.append(f"OOS DD ${result.oos_max_dd:.0f} > cap "
                          f"${rc_try.max_dd_dollars:.0f}")

        report = ContractReport(
            contract=contract, params={**common_params, "per_trade_risk": ptr},
            oos_trades=result.oos_trades,
            oos_total_pnl=result.oos_total_pnl,
            oos_max_dd=result.oos_max_dd,
            oos_profit_factor=result.oos_profit_factor,
            oos_expectancy=result.oos_expectancy,
            passed=passed,
            reason="ok" if passed else "; ".join(reason),
        )
        print(f"  per_trade_risk=${ptr:>4.0f}  OOS trades={report.oos_trades:>3d}  "
              f"PF={report.oos_profit_factor:>5.2f}  "
              f"P&L=${report.oos_total_pnl:>7.2f}  "
              f"DD=${report.oos_max_dd:>6.2f}  "
              f"→ {'PASS' if passed else 'skip (' + report.reason + ')'}")
        if passed and (best is None or report.oos_total_pnl > best.oos_total_pnl):
            best = report
    return best or ContractReport(contract, {}, 0, 0.0, 0.0, 0.0, 0.0,
                                  False, "no param set passed all gates")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["orb", "vwap_mr"], default="orb")
    p.add_argument("--contracts", type=str, default="MES,MNQ,M2K",
                   help="Comma-separated list of contract symbols.")
    p.add_argument("--csv-template", type=str, default=None,
                   help="Optional path template, e.g. './data/{contract}.csv'.")
    p.add_argument("--n-days", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--max-dd", type=float, default=2500.0)
    p.add_argument("--daily-loss-limit", type=float, default=500.0)
    p.add_argument("--eod-flatten", type=str, default="14:55")
    p.add_argument("--per-trade-risks", type=str, default="250",
                   help="Comma-separated $ per-trade-risk values to try. "
                        "Default: 250 (one value, fast). Try '100,250,500' "
                        "for a fuller sweep.")
    args = p.parse_args()

    eod_h, eod_m = map(int, args.eod_flatten.split(":"))
    rc = RiskConfig(
        max_dd_dollars=args.max_dd,
        daily_loss_limit=args.daily_loss_limit,
        eod_flatten=dtime(eod_h, eod_m),
    )
    per_trade_risks = [float(x) for x in args.per_trade_risks.split(",")]

    print(f"Strategy: {args.strategy}   Max DD cap: ${args.max_dd:.0f}   "
          f"Daily limit: ${args.daily_loss_limit:.0f}")
    reports: list[ContractReport] = []
    for c in [s.strip().upper() for s in args.contracts.split(",")]:
        reports.append(run_one(
            c, args.strategy, rc,
            csv_template=args.csv_template,
            n_days=args.n_days, seed=args.seed,
            n_folds=args.n_folds, per_trade_risks=per_trade_risks,
        ))

    survivors = sorted([r for r in reports if r.passed],
                       key=lambda r: r.oos_total_pnl, reverse=True)
    print("\n\n═══════════════════════════════════════════════════════════")
    print(f"  SURVIVORS ({len(survivors)} / {len(reports)})")
    print("═══════════════════════════════════════════════════════════")
    if not survivors:
        print("  No contract passed every gate. Iterate:")
        print("  - provide more data (≥500 sessions recommended)")
        print("  - try the other strategy (vwap_mr complements orb)")
        print("  - loosen per-trade-risk or widen the grid in default_grid()")
    else:
        print(f"  {'contract':<8} {'OOS PF':>7} {'OOS P&L':>10} {'OOS DD':>9} "
              f"{'trades':>7}  params")
        for r in survivors:
            print(f"  {r.contract:<8} {r.oos_profit_factor:>7.2f} "
                  f"${r.oos_total_pnl:>8.2f}  ${r.oos_max_dd:>7.2f} "
                  f"{r.oos_trades:>7d}  {r.params}")
        print("\n  Next step: paper-trade the survivor(s) for 1-2 months before")
        print("  committing real capital. DD cap is mechanical — it WILL halt")
        print("  new entries once the circuit breaker fires.")


if __name__ == "__main__":
    main()
