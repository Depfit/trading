# Intraday futures data

Drop your 1-min OHLCV CSVs here, one file per contract, named with the
contract symbol.

## Filename convention

```
data/MES_1min.csv
data/MNQ_1min.csv
data/M2K_1min.csv
data/MCL_1min.csv
data/MGC_1min.csv
```

Any file that matches `data/{SYMBOL}_1min.csv` is auto-detected by
`run_multi_contract.py --csv-template "data/{contract}_1min.csv"`.

## Required CSV format

```csv
timestamp,open,high,low,close,volume
2024-01-02 08:30:00,4783.25,4784.00,4782.50,4783.75,521
2024-01-02 08:31:00,4783.75,4784.25,4783.25,4783.50,312
...
```

### Column rules

| Column      | Requirement                                                                |
|-------------|-----------------------------------------------------------------------------|
| `timestamp` | Local exchange time (CT for MES/MNQ/M2K/ES/NQ, ET for MCL/MGC). No timezone suffix needed — the loader assumes exchange-local. |
| `open/high/low/close` | Float prices, snapped to the contract's tick size.                |
| `volume`    | Integer. If your source doesn't report volume at 1-min, zeros are fine.    |

Column names are case-insensitive. `Datetime`, `Date`, or `time` are accepted
aliases for `timestamp`. Extra columns are ignored.

### Contract type

- Prefer **single-contract, front-month, non-back-adjusted** series when
  possible. Continuous back-adjusted series create fake gaps on rolls that
  distort intraday backtests.
- If you only have continuous series (typical for TradingView exports),
  still usable for a directional-edge first read — just be aware the absolute
  P&L numbers are optimistic near roll dates.

### Session filter

The loader automatically keeps only RTH bars (e.g. 08:30-15:00 CT for MES).
Include overnight/Globex bars in your CSV if you have them; the loader will
drop them on import.

## Data gotchas the loader tolerates

- Duplicate rows at the same timestamp (kept as-is; pandas sorts).
- Missing minutes (no forward-fill applied — gaps are real).
- Non-trading days present (filtered by `session_date` grouping).

## Data gotchas the loader does NOT tolerate

- Missing any of the OHLCV columns → ValueError.
- Timestamps the loader can't parse via `pd.to_datetime` → ValueError.
- Timestamps in UTC without conversion → *silently wrong* results. Always
  export in exchange local time, or pre-convert before saving.
