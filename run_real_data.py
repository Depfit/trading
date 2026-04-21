"""Exploratory backtest of ORB + VWAP-MR on the 4 user-provided CSVs.

Calibrates per-contract defaults from observed OR widths in the data:
  - MES: OR median ~92 ticks, full-size stops OK under $100/trade budget.
  - MNQ: OR median ~432 ticks; needs wider max_or_ticks AND bigger per-trade
    risk for the sizer to round to ≥ 1 contract.
  - ES (full-size): stops commonly run > $1,000/contract → incompatible with
    a $2,500 account-level DD cap. Included for visibility only; will report
    0 trades when the sizer correctly refuses.
  - GC (full-size): same problem; stops routinely > $2,000/contract.

With only ~16 sessions per contract this is a FIRST READ, not evidence of a
live edge. Widen the data window before committing to any result here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime

from src.futures import (
    IntradayBacktester,
    OpeningRangeBreakout,
    RiskConfig,
    VwapMeanReversion,
    get,
)
from src.futures.data import load_intraday_csv
from src.futures.strategies.orb import ORBConfig
from src.futures.strategies.vwap_mr import VwapMRConfig


@dataclass
class ContractCfg:
    sym: str
    per_trade_risk: float
    eod_flatten: dtime
    orb: ORBConfig
    vwap: VwapMRConfig
    note: str = ""


CONFIGS = [
    ContractCfg(
        sym="MES", per_trade_risk=100.0, eod_flatten=dtime(14, 55),
        orb=ORBConfig(or_minutes=15, stop_buffer_ticks=2, rr_multiple=1.5,
                      max_or_ticks=200),
        vwap=VwapMRConfig(warmup_minutes=30, k_sigma=1.5,
                          stop_buffer_ticks=4),
    ),
    ContractCfg(
        sym="MNQ", per_trade_risk=250.0, eod_flatten=dtime(14, 55),
        orb=ORBConfig(or_minutes=15, stop_buffer_ticks=4, rr_multiple=1.5,
                      max_or_ticks=800),
        vwap=VwapMRConfig(warmup_minutes=30, k_sigma=1.5,
                          stop_buffer_ticks=8),
    ),
    ContractCfg(
        sym="ES", per_trade_risk=250.0, eod_flatten=dtime(14, 55),
        orb=ORBConfig(or_minutes=15, stop_buffer_ticks=2, rr_multiple=1.5,
                      max_or_ticks=200),
        vwap=VwapMRConfig(warmup_minutes=30, k_sigma=1.5,
                          stop_buffer_ticks=4),
        note="Full-size ES: typical stop = $1000+/contract. Use MES instead.",
    ),
    ContractCfg(
        sym="GC", per_trade_risk=250.0, eod_flatten=dtime(13, 25),
        orb=ORBConfig(or_minutes=15, stop_buffer_ticks=2, rr_multiple=1.5,
                      max_or_ticks=300),
        vwap=VwapMRConfig(warmup_minutes=30, k_sigma=1.5,
                          stop_buffer_ticks=4),
        note="Full-size GC: typical stop = $1600+/contract. Use MGC instead.",
    ),
]


def run_one(cfg: ContractCfg, strategy: str) -> dict:
    spec = get(cfg.sym)
    bars = load_intraday_csv(f"data/{cfg.sym}_1min.csv", spec)
    rc = RiskConfig(
        max_dd_dollars=2500,
        daily_loss_limit=500,
        per_trade_risk=cfg.per_trade_risk,
        eod_flatten=cfg.eod_flatten,
    )
    if strategy == "orb":
        strat = OpeningRangeBreakout(spec=spec, config=cfg.orb)
    else:
        strat = VwapMeanReversion(spec=spec, config=cfg.vwap)
    bt = IntradayBacktester(spec=spec, strategy=strat, risk_config=rc)
    result = bt.run(bars)
    return {
        "contract": cfg.sym,
        "strategy": strategy,
        "sessions": int(bars["session_date"].nunique()),
        "per_trade_risk": cfg.per_trade_risk,
        "note": cfg.note,
        **result.metrics,
    }


def main() -> None:
    print("Period: 2026-03-30 → 2026-04-21 (≈16 sessions/contract)")
    print("Risk:   $2500 max DD, $500 daily limit, contract-specific stops")
    print()
    header = (f"{'contract':<4} {'strategy':<9} {'ptr$':>5} {'trades':>6} "
              f"{'win%':>5} {'PF':>6} {'exp$':>7} {'P&L$':>8} {'DD$':>7}")
    print(header)
    print("-" * len(header))

    rows = []
    for cfg in CONFIGS:
        for strat in ("orb", "vwap_mr"):
            r = run_one(cfg, strat)
            rows.append(r)
            pf = r["profit_factor"]
            pf_s = f"{pf:6.2f}" if pf not in (float("inf"),) else "   inf"
            print(f"{r['contract']:<4} {r['strategy']:<9} "
                  f"{r['per_trade_risk']:>5.0f} {r['n_trades']:>6d} "
                  f"{r['win_rate']*100:>5.1f} {pf_s} "
                  f"{r['expectancy']:>7.2f} {r['total_pnl']:>8.2f} "
                  f"{r['max_dd']:>7.2f}")
        if CONFIGS[[c.sym for c in CONFIGS].index(r['contract'])].note:
            print(f"     └─ {CONFIGS[[c.sym for c in CONFIGS].index(r['contract'])].note}")

    print("\n--- Interpretation ---")
    survivors = [r for r in rows if r["n_trades"] >= 5 and r["total_pnl"] > 0
                 and r["max_dd"] <= 2500]
    failed = [r for r in rows if r["n_trades"] < 5]
    losing = [r for r in rows if r["n_trades"] >= 5 and r["total_pnl"] <= 0]
    print(f"  {len(survivors)} contract/strategy combos show preliminary signal")
    print(f"  {len(losing)} show negative expectancy on this window")
    print(f"  {len(failed)} produced too few trades (<5) to be meaningful")
    if survivors:
        print("\nPreliminary survivors (NOT enough data to commit capital):")
        for r in sorted(survivors, key=lambda x: x["total_pnl"], reverse=True):
            print(f"  {r['contract']:<4} {r['strategy']:<9}  "
                  f"PF={r['profit_factor']:.2f}  P&L=${r['total_pnl']:.2f}  "
                  f"DD=${r['max_dd']:.2f}  trades={r['n_trades']}")


if __name__ == "__main__":
    main()
