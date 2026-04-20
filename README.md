# Trading strategy research framework

Two independent components in one repo:

1. **`src/` — long-horizon dual-momentum portfolio strategy** (original
   research harness for daily-bar equity/ETF allocation).
2. **`src/futures/` — intraday futures framework for prop-firm accounts**:
   hard-coded risk controls (max DD circuit breaker, daily loss limit, EOD
   flatten), two intraday strategies (Opening Range Breakout, VWAP Mean
   Reversion), walk-forward optimizer, and a multi-contract iteration CLI.

The two live side by side because they share utility code and share a
philosophy: *mechanical rules, explicit risk budget, walk-forward validation,
no curve-fitting*.

---

## Part 1 — Futures framework (prop-firm style)

### Goal

Automate intraday futures trading on a **$50,000 account** with a
**$2,500 max drawdown** (5%), closing all positions by EOD. Small, fast,
hard-controlled so a drawdown breach is structurally impossible, and so a
losing day can't turn into a blown account.

### Mechanical risk controls (`src/futures/risk.py`)

Guardrails are enforced by the backtester before every order:

- **Per-trade sizing.** `contracts = floor(per_trade_risk / (stop_ticks × tick_value))`.
  Default $100 per trade on a $2,500 DD budget gives ~25 losing trades of
  headroom. On 1-tick slippage the engine never exceeds budgeted risk.
- **Daily loss limit.** Once cumulative today's P&L ≤ `-daily_loss_limit`
  (default $500), new entries are blocked until the next session.
- **Max-DD circuit breaker.** At 60% of max DD, per-trade sizing is halved
  defensively. At 80%, a permanent halt flag is set — no new entries ever.
- **EOD flatten.** At the configured local time (default 14:55 CT for CME
  index products) any open position is closed at that bar's close ± slip,
  and no new signals are taken.

These are tested in `tests/test_futures.py` — e.g. `test_backtester_never_breaches_dd_cap`
asserts that over any synthetic run the realized max DD is ≤ the configured cap.

### Strategies

Two complementary intraday edges:

| Strategy     | Edge source                                              | Typical day       |
|--------------|-----------------------------------------------------------|-------------------|
| **ORB** (`strategies/orb.py`) | Opening Range Breakout (Toby Crabel 1990; extensive retests on ES/NQ) | Trending          |
| **VWAP-MR** (`strategies/vwap_mr.py`) | Mean reversion to session VWAP from ≥ k·σ deviation | Range / chop      |

Both are one-trade-per-session, produce explicit stop/target/entry triples, and
let the backtester handle sizing/exits.

### Walk-forward optimizer (`src/futures/walk_forward.py`)

Given a contract, data, strategy factory, and param grid:

- Split the data into `n_folds` contiguous session-level folds.
- For each fold *i*: grid-search on fold *i* (train), then evaluate the
  winning params on fold *i+1* (out-of-sample).
- Report aggregate OOS metrics — **this is the only number that matters**.
  In-sample PF is always inflated by selection.

Objective: `expectancy × n_trades`, hard-rejecting any config whose *IS*
max DD exceeds the account cap.

### Iteration workflow (exactly what the request asks for)

```
┌────────────┐  walk-forward  ┌──────────────┐  if passes gates
│ contract A │ ─────────────► │ OOS metrics  │ ───────────────► keep
│   data     │                │ respects DD? │        │
└────────────┘                └──────────────┘        │
                                                       ▼
                                                 ┌────────────┐
                                                 │ contract B │
                                                 │   data     │ ──► repeat
                                                 └────────────┘
```

Automated via `run_multi_contract.py`:

```bash
# Synthetic demo (no market data required):
PYTHONPATH=. python3 run_multi_contract.py --strategy orb \
    --contracts MES,MNQ --n-days 200 --n-folds 3

# Real data (user-provided 1-min CSVs):
PYTHONPATH=. python3 run_multi_contract.py --strategy orb \
    --contracts MES,MNQ,M2K \
    --csv-template "./data/{contract}_1min.csv"
```

Output: a ranked list of **survivors** — contract + tuned params that
passed every gate (positive OOS expectancy, OOS DD within cap, enough
trades for statistical relevance).

### Demo results (synthetic, 200 sessions/contract)

These are synthetic — real markets will be noisier and the absolute P&L will
be lower. The point is to show the framework works and the DD constraint is
mechanically enforced end-to-end.

```
MES (Micro E-mini S&P 500)  ORB  per_trade_risk=$250
  OOS trades=79  PF=2.98  P&L=$12,143  DD=$964   → PASS
MNQ (Micro E-mini Nasdaq)   ORB  per_trade_risk=$250
  OOS trades=57  PF=2.36  P&L=$6,110   DD=$708   → PASS
```

Both sit well under the $2,500 DD budget. Same ORB strategy, per-contract
tuned parameters (`max_or_ticks` differs because MNQ ranges are wider).

### Running on your own data

Expected CSV schema (one file per contract, IB / NinjaTrader / Databento /
Polygon all produce this):

```csv
timestamp,open,high,low,close,volume
2024-01-02 08:30:00,4783.25,4784.00,4782.50,4783.75,521
...
```

Timestamps must be in the contract's **exchange local time** (CT for CME
index, ET for NYMEX/COMEX). The loader filters to RTH automatically.

Recommended data depth: **≥ 500 sessions** (~2 years of 1-min bars) so
walk-forward folds are statistically meaningful.

### Single-contract CLI

```bash
# Default params, single backtest:
PYTHONPATH=. python3 run_futures.py --contract MES --strategy orb --synthetic

# Walk-forward on synthetic:
PYTHONPATH=. python3 run_futures.py --contract MES --strategy orb --synthetic \
    --walk-forward --n-folds 4 --n-days 250

# Walk-forward on your CSV:
PYTHONPATH=. python3 run_futures.py --contract MNQ --strategy vwap_mr \
    --csv ./data/MNQ_2022_2024.csv --walk-forward
```

Tunable risk controls (all respected by the backtester):

| Flag                   | Default | Meaning                             |
|------------------------|--------:|-------------------------------------|
| `--max-dd`             | 2500    | Max drawdown in $ (hard cap)        |
| `--daily-loss-limit`   | 500     | Halt trading for the day at -$500   |
| `--per-trade-risk`     | 100     | Max $ at risk per trade             |
| `--eod-flatten`        | 14:55   | Exchange local HH:MM to flatten     |

### What's honest vs what's demo

**Honest:**
- The risk engine mathematically cannot breach the DD cap (unit-tested).
- Walk-forward reports only OOS — no in-sample PF inflation.
- Slippage (1 tick) and round-trip fees ($0.88 for micros, $4.60 for
  full-size index) are charged per fill.
- EOD flatten is enforced even if the strategy wants to stay long.

**Demo (needs your real data to validate):**
- Synthetic intraday bars are stylized (day-type regimes, opening-range
  dynamics, closing push). They are *not* calibrated to any single real
  market. Real-data edges typically shrink by 30-50% vs. a friendly
  synthetic generator.
- The grid in `run_futures.py:default_grid()` is small for demo speed.
  On real data, widen it and run overnight.

### Going from backtest to live

The framework stops at the backtester boundary. For live trading you need:

1. **Paper trading** for 1-2 months minimum on the survivor params. If live
   PF < 50% of OOS PF, something is mis-modeled (slippage, fill timing).
2. **Broker adapter** — IBKR / Rithmic / Tradovate API. The `Signal` /
   `RiskEngine` / backtest loop can be wrapped to submit live orders with
   identical risk checks; this is not included here.
3. **Session monitoring** — alert on circuit breaker trips so you can
   diagnose; don't auto-resume.

### Tests

```bash
PYTHONPATH=. python3 tests/test_futures.py
```

Verifies: risk engine halts at DD cap, daily loss limit triggers, position
sizer respects budget, every trade exits by EOD, strategy signals are
well-formed, walk-forward folds are disjoint.

---

## Part 2 — Dual-momentum portfolio strategy (long-horizon)

Original research harness. See `run_backtest.py`, `src/strategy.py`,
`src/backtest.py`, `src/data.py`, `tests/test_robustness.py`.

Rules (monthly rebalance):
1. Asset eligible only if price > 200-day SMA AND 126-day return > risk-free.
2. Rank eligible assets by 126-day return, keep top 3.
3. Inverse-vol weights, scaled to 12% annualized vol target, 40% max per
   asset; unallocated weight earns cash.
4. 5 bps transaction cost per unit of turnover.

30-seed Monte-Carlo on synthetic 20-year markets:

| Metric          | Strategy | Buy & hold equity |
|-----------------|---------:|------------------:|
| Mean CAGR       |  12.1 %  |   10.6 %          |
| Mean Sharpe     |   0.75   |    0.51           |
| Mean max DD     | −28.9 %  | −49.5 %           |
| Beats BH Sharpe |  29 / 30 |        —          |

```bash
PYTHONPATH=. python3 run_backtest.py                  # default synthetic demo
PYTHONPATH=. python3 tests/test_robustness.py         # 30-seed Monte-Carlo
```

---

## Layout

```
src/
  strategy.py              # long-horizon dual momentum
  backtest.py              # daily P&L engine
  data.py                  # CSV loader + synthetic multi-asset universe
  futures/
    contracts.py           # MES/MNQ/M2K/MCL/MGC/ES/NQ specs
    risk.py                # RiskEngine: sizing, daily limit, DD breaker, EOD
    synthetic.py           # reproducible intraday bar generator
    data.py                # 1-min CSV loader
    backtest.py            # bar-by-bar intraday backtester
    walk_forward.py        # train/OOS grid-search optimizer
    strategies/
      base.py              # Strategy protocol + Signal dataclass
      orb.py               # Opening Range Breakout
      vwap_mr.py           # VWAP Mean Reversion
run_backtest.py            # long-horizon CLI
run_futures.py             # single-contract intraday CLI
run_multi_contract.py      # iterate across contracts, report survivors
tests/
  test_robustness.py       # portfolio Monte-Carlo
  test_futures.py          # futures framework smoke tests
```

---

## Caveats (both systems)

- No strategy is guaranteed profitable. What's guaranteed is that risk
  controls are mechanical and tested.
- Synthetic data is for framework validation — *not* for choosing parameters
  you'll trade live. Always re-tune on real data for the contracts you trade.
- Past performance (in- or out-of-sample, synthetic or real) does not
  guarantee future returns.
- Slippage and fees modeled here are optimistic. Real fill quality varies
  with time of day, volatility, and queue position.
