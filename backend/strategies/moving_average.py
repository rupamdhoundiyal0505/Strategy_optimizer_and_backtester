import pandas as pd
from strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    """
    Go long when short MA is above long MA, go short when it's below.
    allow_short=False (default) makes it behave like the original long-only
    version - short signals just become flat instead of -1.
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        short_window = self.params["short_window"]
        long_window = self.params["long_window"]
        allow_short = self.params.get("allow_short", False)

        short_ma = df["close"].rolling(short_window).mean()
        long_ma = df["close"].rolling(long_window).mean()

        position = pd.Series(0.0, index=df.index)
        position[short_ma > long_ma] = 1.0
        position[short_ma < long_ma] = -1.0 if allow_short else 0.0
        return position