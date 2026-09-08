import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

# pandas resample rule strings for common timeframes
TIMEFRAME_RULES = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "1h": "1h",
    "1d": "1D",
}


def load_raw(symbol: str) -> pd.DataFrame:
    """Load the raw 1-min data for one symbol from its parquet file."""
    path = RAW_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data stored for symbol '{symbol}'")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    return df.sort_index()


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Turn 1-min OHLCV data into any coarser timeframe.
    OHLC aggregation rule: open=first price in the window, high=max,
    low=min, close=last price in the window, volume=summed.
    """
    if timeframe not in TIMEFRAME_RULES:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from {list(TIMEFRAME_RULES)}")

    rule = TIMEFRAME_RULES[timeframe]
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df.resample(rule).agg(agg)
    return resampled.dropna(subset=["open"])  # drop windows with no trades (e.g. non-trading minutes)


def get_stock_data(symbol: str, timeframe: str = "1min",
                    start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Main entry point: load, filter by date range, resample to requested timeframe."""
    df = load_raw(symbol)

    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    if timeframe != "1min":
        df = resample(df, timeframe)

    return df