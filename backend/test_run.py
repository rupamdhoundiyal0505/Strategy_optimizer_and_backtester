import numpy as np
import pandas as pd

from strategies.moving_average import MovingAverageCrossover
from backtest.engine import run_backtest
from optimizers.grid_search import GridSearchOptimizer

# --- fake price data so we don't need a live data source yet ---
np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=500)
prices = 100 + np.cumsum(np.random.randn(500)) + np.linspace(0, 20, 500)
df = pd.DataFrame({"close": prices}, index=dates)

# --- 1. run a single strategy once ---
strategy = MovingAverageCrossover(short_window=20, long_window=50)
signals = strategy.generate_signals(df)
metrics = run_backtest(signals, df)
print("Single backtest (20/50):")
for k, v in metrics.items():
    if k != "equity_curve":
        print(f"  {k}: {v}")

# --- 2. run the optimizer over several combos ---
optimizer = GridSearchOptimizer(
    strategy_cls=MovingAverageCrossover,
    param_space={"short_window": [10, 20, 50], "long_window": [50, 100, 200]},
    objective="sharpe_ratio",
)
result = optimizer.optimize(df)
print("\nOptimizer result:")
print(f"  best_params: {result['best_params']}")
print(f"  best_score (sharpe): {result['best_score']}")
print(f"  combos tried: {len(result['all_results'])}")