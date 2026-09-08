# Backtest Platform — Backend

A full-stack strategy backtesting platform, built in phases, production-style.
This README explains what's built, how a request actually flows through the
system, and what's still ahead.

## What this is

You write a trading strategy as a Python class. You give it a symbol, a
timeframe, and a date range. The system fetches (and caches) real price
data, resamples it, runs your strategy against it bar by bar with proper
risk management (trailing stops), and hands back a full trade ledger plus
performance metrics. There's also an optimizer that can search across many
parameter combinations to find the best-performing setup for a strategy.

## Run it

```bash
pip install -r requirements.txt
python3 test_run.py          # sanity check, no server needed
uvicorn main:app --reload    # then visit http://127.0.0.1:8000/docs
```

## The full request flow, end to end

This is the actual path a `POST /backtest` request takes through the code:

```
1. Client sends: { strategy, params, symbol, timeframe, start, end, trailing_stop_pct }
                            |
2. main.py: get_price_data(symbol, timeframe, start, end)
                            |
3. data/ingestion.py: ensure_data()
     - checks the local parquet cache for this symbol
     - CACHE HIT  -> reads straight from disk, no external call
     - CACHE MISS -> calls data/vendors/fyers.py (rate-limited, chunked),
                     saves the result to disk for next time
                            |
4. data/loader.py: resample()
     - converts the raw 1-minute data into whatever timeframe was requested
     - (5min, 15min, 1h, 1d - OHLC aggregation: open=first, high=max,
        low=min, close=last, volume=sum)
                            |
5. strategies/<your_strategy>.py: generate_signals(df)
     - your strategy's own logic, e.g. moving average crossover
     - returns the TARGET POSITION per bar: 1.0 long, 0.0 flat, -1.0 short
                            |
6. backtest/ledger_engine.py: run_backtest(signals, df, trailing_stop_pct)
     - walks the data bar by bar (not vectorized - trailing stops need
       per-trade memory of the best price seen since entry)
     - each bar: check the trailing stop FIRST (using that bar's low/high,
       not just the close), then check if the strategy's signal changed
     - a reversal (long -> short) closes the current trade and opens the
       opposite one in the same bar - they never coexist as two positions
     - produces: closed_trades (a ledger, one row per completed trade),
       active_order (still open at the end, if any), equity_curve
                            |
7. backtest/analytics.py: compute_metrics()
     - reads the closed_trades ledger and equity_curve
     - derives win rate, avg win/loss, profit factor, Sharpe ratio,
       max drawdown, and a breakdown of WHY each trade closed
       (sl_hit / signal_reversal / signal_flat)
                            |
8. main.py: converts equity_curve/trade dates into plain JSON, returns it
                            |
9. Client gets back: { closed_trades: [...], active_order, equity_curve,
                        metrics: {...} }
```

`POST /optimize` shares steps 2-4 (same data fetching/caching), but instead
of steps 5-7 running once, it runs many times - once per parameter
combination in the search grid - using the faster `backtest/engine.py`
(vectorized, no per-trade stop-loss precision, but fast enough to run
hundreds of times per request). See `optimizers/grid_search.py`.

## Folder structure

```
backend/
|-- strategies/
|   |-- base.py                # Strategy interface every strategy inherits from.
|   |                             generate_signals(df) -> target position (1/0/-1)
|   |-- moving_average.py      # first concrete strategy: MA crossover, optional shorting
|
|-- optimizers/
|   |-- base.py                 # Optimizer interface: takes a strategy CLASS + param_space
|   |-- grid_search.py          # exhaustive search over every param combination
|
|-- backtest/
|   |-- engine.py                # FAST vectorized backtest, no stops. Used by the optimizer,
|   |                              where speed matters more than per-trade precision.
|   |-- ledger_engine.py         # REALISTIC bar-by-bar backtest: trailing stops, a proper
|   |                              trade ledger, immediate reversal handling. Used by /backtest.
|   |-- analytics.py             # turns a trade ledger into win rate, Sharpe, profit factor, etc.
|                                  Deliberately separate from execution - works on any ledger
|                                  shaped the same way, regardless of which engine produced it.
|
|-- data/
|   |-- loader.py                 # resample() - 1min data to any coarser timeframe
|   |-- ingestion.py              # ensure_data() - cache-aside: only fetches what's
|   |                               genuinely missing from local storage, never re-fetches
|   |                               data you already have
|   |-- vendors/
|       |-- fyers.py               # pluggable data source: rate-limited, chunked into
|                                    <=100 day requests, matches the FetchFn signature
|                                    ensure_data() expects. Swappable for any other vendor
|                                    without touching ingestion.py or anything upstream.
|
|-- main.py                      # FastAPI app - wires all of the above into
|                                   POST /backtest, POST /optimize, GET /strategies
|-- test_run.py                   # end-to-end sanity check using synthetic data, no server
|-- requirements.txt
|-- README.md                     # this file
```

## Core design decisions, and why

**Strategies are a swappable interface, not hardcoded logic.** Every strategy
implements `generate_signals(df) -> pd.Series`. The backtest engine, the
optimizer, and the API never know or care what's inside a specific
strategy - they only call this one method. This is what will let an AI-generated
strategy plug in later without changing anything else: it just needs to be
registered in `STRATEGY_REGISTRY` in `main.py`.

**Position is a single number (1 / 0 / -1), never two things at once.**
A single instrument can't be long and short simultaneously - that nets to
zero. A reversal is always sequential: close, then open, in that order,
recorded as two separate trades in the ledger sharing one price point.

**Two backtest engines, on purpose.** The vectorized one (`engine.py`) is
fast because it computes the whole equity curve in one pass, but can't
express anything path-dependent like a trailing stop. The ledger engine
(`ledger_engine.py`) walks bar by bar so it CAN track "the best price since
this specific trade was entered" - at the cost of being slower. The
optimizer uses the fast one (it needs to run hundreds of times per
request); real backtests use the accurate one.

**Data fetching is cache-aside and vendor-agnostic.** `ensure_data()`
doesn't know or care that `fetch_fyers_history` happens to call Fyers -
it just calls whatever function matches the `FetchFn` signature
`(symbol, start, end) -> DataFrame`. Swapping vendors later means writing
one new file in `data/vendors/`, not touching the caching logic.

**Analytics is separate from execution.** `compute_metrics()` only reads
the `closed_trades` ledger and `equity_curve` - it has no idea how those
were produced. This means the exact same analytics code will work on
results from a completely different execution engine later (e.g. a
multi-leg options engine), as long as it produces a ledger in the same shape.

## What's built and tested so far

- Strategy + Optimizer base classes, with `MovingAverageCrossover` and
  `GridSearchOptimizer` as first concrete implementations
- Vectorized backtest engine (fast, used by the optimizer)
- Ledger-based backtest engine (trailing stops, trade ledger, tested against
  a long-to-short reversal scenario and a trailing-stop-hit scenario)
- Analytics layer (win rate, profit factor, Sharpe, drawdown, exit-reason breakdown)
- Data layer: parquet-based 1-min storage, resampling to any timeframe,
  cache-aside fetching that only pulls missing date ranges
- Fyers fetcher skeleton: rate limiting, date-range chunking, pluggable
  into the cache layer (exact API endpoint/params still need verifying
  against current Fyers docs, and OAuth token refresh isn't implemented yet)
- Full FastAPI wiring: `/backtest`, `/optimize`, `/strategies`, tested
  end-to-end with `TestClient` against seeded cache data

## What's designed but not yet built

- **Postgres**: no database yet. Nothing persists between requests - every
  backtest recomputes from scratch. Tables needed: users, strategies,
  backtest_runs, optimization_runs, trades.
- **Auth**: no user accounts or JWT yet. Every request is anonymous.
- **Job queue (Celery + Redis)**: optimization runs currently block the
  HTTP request until finished. Fine for a handful of param combos, not
  fine for a large grid search with real users waiting.
- **Frontend**: none yet. Everything so far is API-only.
- **AI layer**: the part that turns a plain-English prompt into a new
  `Strategy` subclass, sandboxed before execution. Not started.
- **Fyers OAuth refresh flow**: `refresh_access_token()` in `fyers.py` is
  a stub - needs the actual daily token refresh wired up once the OAuth
  app is registered.

## API reference (current)

**`POST /backtest`**
```json
{
  "strategy": "moving_average_crossover",
  "params": {"short_window": 20, "long_window": 50, "allow_short": true},
  "symbol": "NIFTYFUT",
  "timeframe": "15min",
  "start": "2024-01-01",
  "end": "2024-06-01",
  "trailing_stop_pct": 0.02,
  "initial_capital": 100000
}
```
Returns `closed_trades`, `active_order`, `equity_curve`, and `metrics`.

**`POST /optimize`**
```json
{
  "strategy": "moving_average_crossover",
  "param_space": {"short_window": [10, 20, 50], "long_window": [50, 100, 200]},
  "objective": "sharpe_ratio",
  "symbol": "NIFTYFUT",
  "timeframe": "15min",
  "start": "2024-01-01",
  "end": "2024-06-01"
}
```
Returns `best_params`, `best_score`, and `all_results` (every combo tried).

**`GET /strategies`** — lists registered strategy names.