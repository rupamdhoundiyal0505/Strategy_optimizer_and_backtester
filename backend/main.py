from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from strategies.moving_average import MovingAverageCrossover
from optimizers.grid_search import GridSearchOptimizer
from backtest.engine import run_backtest as run_backtest_vectorized
from backtest.ledger_engine import run_backtest as run_backtest_ledger
from backtest.analytics import compute_metrics
from data.ingestion import ensure_data
from data.loader import resample
from data.vendors.fyers import fetch_fyers_history

app = FastAPI(title="Backtest Platform API")

STRATEGY_REGISTRY = {
    "moving_average_crossover": MovingAverageCrossover,
}


class BacktestRequest(BaseModel):
    strategy: str
    params: dict
    symbol: str
    timeframe: str = "1min"
    start: str
    end: str
    trailing_stop_pct: float | None = None
    initial_capital: float = 10000.0


class OptimizeRequest(BaseModel):
    strategy: str
    param_space: dict  # e.g. {"short_window": [10, 20, 50], "long_window": [50, 100, 200]}
    objective: str = "sharpe_ratio"
    symbol: str
    timeframe: str = "1min"
    start: str
    end: str


def get_price_data(symbol: str, timeframe: str, start: str, end: str):
    """Shared by both endpoints: cache-aside fetch, then resample to the requested timeframe."""
    try:
        raw_1min = ensure_data(symbol, start, end, fetch_fn=fetch_fyers_history)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data for {symbol}: {e}")

    if raw_1min.empty:
        raise HTTPException(status_code=404, detail=f"No data available for {symbol} in that range")

    return raw_1min if timeframe == "1min" else resample(raw_1min, timeframe)


def jsonable(result: dict) -> dict:
    """Equity curves and trade dates need converting before they can go into a JSON response."""
    out = dict(result)
    if "equity_curve" in out:
        out["equity_curve"] = out["equity_curve"].tolist()
    if "closed_trades" in out:
        out["closed_trades"] = [
            {**t, "entry_date": str(t["entry_date"]), "exit_date": str(t["exit_date"])}
            for t in out["closed_trades"]
        ]
    if "active_order" in out and out["active_order"] is not None:
        out["active_order"] = {**out["active_order"], "entry_date": str(out["active_order"]["entry_date"])}
    return out


@app.post("/backtest")
def backtest(req: BacktestRequest):
    strategy_cls = STRATEGY_REGISTRY.get(req.strategy)
    if strategy_cls is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy '{req.strategy}'")

    df = get_price_data(req.symbol, req.timeframe, req.start, req.end)
    strategy = strategy_cls(**req.params)
    signals = strategy.generate_signals(df)

    result = run_backtest_ledger(signals, df, trailing_stop_pct=req.trailing_stop_pct,
                                  initial_capital=req.initial_capital)
    metrics = compute_metrics(result, initial_capital=req.initial_capital)

    return {**jsonable(result), "metrics": metrics}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    strategy_cls = STRATEGY_REGISTRY.get(req.strategy)
    if strategy_cls is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy '{req.strategy}'")

    df = get_price_data(req.symbol, req.timeframe, req.start, req.end)
    # optimizer still uses the fast vectorized engine - it runs the backtest
    # once per param combo, and doesn't need per-trade stop-loss precision
    # to compare hundreds of combos quickly.
    optimizer = GridSearchOptimizer(strategy_cls, req.param_space, req.objective)
    return optimizer.optimize(df)


@app.get("/strategies")
def list_strategies():
    return {"available": list(STRATEGY_REGISTRY.keys())}