from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """Base class every trading strategy must inherit from."""

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df: OHLCV dataframe indexed by date, must contain a 'close' column.
        returns: pd.Series aligned to df.index, the TARGET POSITION at each row:
                  1.0  = fully long
                  0.0  = flat, no position
                 -1.0  = fully short
        A position can never be long and short at once on one instrument -
        it's a single number that flips between these states over time.
        """
        raise NotImplementedError

    def validate_params(self) -> bool:
        return True