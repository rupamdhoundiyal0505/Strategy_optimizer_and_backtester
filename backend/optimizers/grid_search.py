import itertools
from optimizers.base import Optimizer
from backtest.engine import run_backtest


class GridSearchOptimizer(Optimizer):
    """Exhaustively tries every combination in param_space. Simple, slow, correct."""

    def _param_grid(self):
        keys = list(self.param_space.keys())
        value_lists = [self.param_space[k] for k in keys]
        for combo in itertools.product(*value_lists):
            yield dict(zip(keys, combo))

    def optimize(self, price_data) -> dict:
        results = []
        for params in self._param_grid():
            strategy = self.strategy_cls(**params)          # fresh object per combo, no shared state
            signals = strategy.generate_signals(price_data)
            metrics = run_backtest(signals, price_data)
            results.append({
                "params": params,
                "score": metrics[self.objective],
                "metrics": {k: v for k, v in metrics.items() if k != "equity_curve"},
            })

        best = max(results, key=lambda r: r["score"])
        return {
            "best_params": best["params"],
            "best_score": best["score"],
            "all_results": results,
        }