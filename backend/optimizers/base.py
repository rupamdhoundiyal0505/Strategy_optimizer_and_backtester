from abc import ABC, abstractmethod
 
 
class Optimizer(ABC):
    """Base class every parameter-search algorithm must inherit from."""
 
    def __init__(self, strategy_cls, param_space: dict, objective: str = "sharpe_ratio"):
        self.strategy_cls = strategy_cls   # the CLASS, not an instance - optimizer creates many objects from it
        self.param_space = param_space     # e.g. {"short_window": [10, 20, 50], "long_window": [50, 100, 200]}
        self.objective = objective         # which backtest metric to maximize
 
    @abstractmethod
    def optimize(self, price_data) -> dict:
        """returns: {'best_params': dict, 'best_score': float, 'all_results': list}"""
        raise NotImplementedError
 