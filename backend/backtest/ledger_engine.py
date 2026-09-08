import pandas as pd


def run_backtest(signals: pd.Series, df: pd.DataFrame,
                  trailing_stop_pct: float = None,
                  initial_capital: float = 10000.0) -> dict:
    """
    Single-instrument, single-position-at-a-time execution loop.

    signals: target position per bar (1.0 long, 0.0 flat, -1.0 short)
    trailing_stop_pct: e.g. 0.05 = stop trails 5% behind the best price
                        since entry. None disables stops entirely.

    Priority per bar:
      1. Update and check the trailing stop first, using that bar's low
         (for longs) / high (for shorts) - not just the close.
      2. If the stop didn't already close the trade, compare the strategy's
         signal to the current position. A reversal closes the current
         trade AND opens the opposite one immediately, in the same bar -
         they never coexist, it's always close-then-open in sequence.

    Returns:
      active_order:   the still-open position at the end of the run, or None
      closed_trades:  list of completed trades, one dict per trade
      equity_curve:   pd.Series of account equity, one value per bar
      final_equity:   the last value of the equity curve
    """
    active_order = None   # at most ONE dict at any time: {dir, entry_price, entry_date, sl}
    closed_trades = []
    equity = initial_capital
    equity_curve = []

    for date, row in df.iterrows():
        close = row["close"]
        low = row.get("low", close)
        high = row.get("high", close)
        target = signals.loc[date]
        exited_via_stop = False

        # --- 1. trailing stop: update, then check ---
        if active_order is not None and trailing_stop_pct is not None:
            if active_order["dir"] == "long":
                new_stop = close * (1 - trailing_stop_pct)
                active_order["sl"] = max(active_order["sl"], new_stop)  # trails up only
                if low <= active_order["sl"]:
                    exit_price = active_order["sl"]
                    pnl_pct = (exit_price - active_order["entry_price"]) / active_order["entry_price"]
                    equity *= (1 + pnl_pct)
                    closed_trades.append({**active_order, "exit_price": exit_price, "exit_date": date,
                                           "exit_reason": "sl_hit", "pnl_pct": round(pnl_pct * 100, 2)})
                    active_order = None
                    exited_via_stop = True
            else:  # short
                new_stop = close * (1 + trailing_stop_pct)
                active_order["sl"] = min(active_order["sl"], new_stop)  # trails down only
                if high >= active_order["sl"]:
                    exit_price = active_order["sl"]
                    pnl_pct = (active_order["entry_price"] - exit_price) / active_order["entry_price"]
                    equity *= (1 + pnl_pct)
                    closed_trades.append({**active_order, "exit_price": exit_price, "exit_date": date,
                                           "exit_reason": "sl_hit", "pnl_pct": round(pnl_pct * 100, 2)})
                    active_order = None
                    exited_via_stop = True

        # --- 2. signal-driven exit/entry (only if the stop didn't already act) ---
        current_position = 0.0 if active_order is None else (1.0 if active_order["dir"] == "long" else -1.0)

        if not exited_via_stop and target != current_position:
            if active_order is not None:
                dir_ = active_order["dir"]
                pnl_pct = ((close - active_order["entry_price"]) / active_order["entry_price"] if dir_ == "long"
                           else (active_order["entry_price"] - close) / active_order["entry_price"])
                equity *= (1 + pnl_pct)
                reason = "signal_flat" if target == 0.0 else "signal_reversal"
                closed_trades.append({**active_order, "exit_price": close, "exit_date": date,
                                       "exit_reason": reason, "pnl_pct": round(pnl_pct * 100, 2)})
                active_order = None

            if target != 0.0:
                dir_ = "long" if target == 1.0 else "short"
                sl = None
                if trailing_stop_pct is not None:
                    sl = close * (1 - trailing_stop_pct) if dir_ == "long" else close * (1 + trailing_stop_pct)
                active_order = {"dir": dir_, "entry_price": close, "entry_date": date, "sl": sl}

        equity_curve.append(equity)

    return {
        "active_order": active_order,
        "closed_trades": closed_trades,
        "equity_curve": pd.Series(equity_curve, index=df.index),
        "final_equity": round(equity, 2),
    }