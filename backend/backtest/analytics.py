import numpy as np
import pandas as pd


def compute_metrics(result: dict, initial_capital: float = 10000.0) -> dict:
    """
    Takes the output of ledger_engine.run_backtest() and derives every stat
    from the closed_trades ledger + equity_curve. This is deliberately
    separate from execution logic - the same function works no matter
    what engine produced the trades, as long as the shape matches.
    """
    trades = result["closed_trades"]
    equity_curve = result["equity_curve"]

    if not trades:
        return {"num_trades": 0, "message": "no closed trades"}

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    long_trades = [t for t in trades if t["dir"] == "long"]
    short_trades = [t for t in trades if t["dir"] == "short"]

    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = -sum(t["pnl_pct"] for t in losses)  # make positive for the ratio

    daily_returns = equity_curve.pct_change().fillna(0.0)
    sharpe_ratio = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0 else 0.0
    )
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max

    return {
        "num_trades": len(trades),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "avg_win_pct": round(gross_profit / len(wins), 2) if wins else 0.0,
        "avg_loss_pct": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "total_return_pct": round((result["final_equity"] / initial_capital - 1) * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown_pct": round(drawdown.min() * 100, 2),
        "final_equity": result["final_equity"],
        "exit_reason_breakdown": pd.Series([t["exit_reason"] for t in trades]).value_counts().to_dict(),
    }