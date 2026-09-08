import numpy as np
import pandas as pd


def run_backtest(signals: pd.Series, df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """
    Long/short backtest. signals is now the TARGET POSITION directly
    (1.0 long, 0.0 flat, -1.0 short) - see strategies/base.py.
    No transaction costs or slippage yet - add those once this baseline
    is working correctly.

    df: must contain a 'close' column, same index as signals
    """
    position = signals.astype(float)

    daily_returns = df["close"].pct_change().fillna(0.0)
    # shift(1): today's return depends on YESTERDAY's position, not today's -
    # you can't act on a signal before it exists. Skipping this is the most
    # common backtesting bug (lookahead bias).
    # Note this same line handles shorting for free: when position is -1,
    # multiplying by daily_returns FLIPS the sign - a price drop becomes a
    # positive strategy return, which is exactly what being short means.
    strategy_returns = position.shift(1).fillna(0.0) * daily_returns

    equity_curve = (1 + strategy_returns).cumprod() * initial_capital

    total_return = equity_curve.iloc[-1] / initial_capital - 1
    sharpe_ratio = (
        (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        if strategy_returns.std() > 0 else 0.0
    )
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "final_equity": round(equity_curve.iloc[-1], 2),
        "equity_curve": equity_curve,
    }