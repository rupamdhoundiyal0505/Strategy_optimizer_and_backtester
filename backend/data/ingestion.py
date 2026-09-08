import pandas as pd
from pathlib import Path
from typing import Callable

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Pluggable fetcher: swap this for a real vendor call (Kite Connect, Upstox, etc.)
# without touching any caching logic below. Signature: (symbol, start, end) -> DataFrame
FetchFn = Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame]


def _path(symbol: str) -> Path:
    return RAW_DIR / f"{symbol}.parquet"


def _load_cached(symbol: str) -> pd.DataFrame | None:
    path = _path(symbol)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df.set_index("timestamp").sort_index() if "timestamp" in df.columns else df.sort_index()


def _save(symbol: str, df: pd.DataFrame):
    df.reset_index().to_parquet(_path(symbol), index=False)


def ensure_data(symbol: str, start: str, end: str, fetch_fn: FetchFn) -> pd.DataFrame:
    """
    Cache-aside: return locally cached data for [start, end], fetching only
    whatever's missing from the vendor. This is the function every API route
    should call instead of loader.load_raw() directly.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    cached = _load_cached(symbol)

    if cached is None:
        # nothing cached at all - fetch the full requested range
        print(f"[cache MISS] {symbol}: no local data, fetching {start} to {end}")
        fresh = fetch_fn(symbol, start_ts, end_ts)
        _save(symbol, fresh)
        return fresh

    cached_start, cached_end = cached.index.min(), cached.index.max()
    missing_chunks = []

    if start_ts < cached_start:
        print(f"[cache PARTIAL] {symbol}: fetching missing head {start_ts} to {cached_start}")
        missing_chunks.append(fetch_fn(symbol, start_ts, cached_start))

    if end_ts > cached_end:
        print(f"[cache PARTIAL] {symbol}: fetching missing tail {cached_end} to {end_ts}")
        missing_chunks.append(fetch_fn(symbol, cached_end, end_ts))

    if missing_chunks:
        combined = pd.concat([cached] + missing_chunks)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        _save(symbol, combined)
        cached = combined
    else:
        print(f"[cache HIT] {symbol}: fully served from local cache, no fetch needed")

    return cached[(cached.index >= start_ts) & (cached.index <= end_ts)]