"""
Fyers historical data fetcher - implements the FetchFn signature that
data/ingestion.py's ensure_data() expects: (symbol, start, end) -> DataFrame.

IMPORTANT: the exact request URL/params below are placeholders. Fyers'
API details (endpoint paths, auth header format, response schema) should
be verified against their current official docs before this hits real
users - broker APIs change these without much notice. What's real and
worth keeping regardless of the exact endpoint: the rate limiting, the
100-day chunking, and the pluggable fetch_fn shape.
"""
import os
import time
import requests
import pandas as pd
from collections import deque

FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
FYERS_ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")  # refreshed daily, see refresh_access_token()

MAX_REQUESTS_PER_SECOND = 10
MAX_DAYS_PER_REQUEST = 100  # Fyers' cap for intraday resolutions


class RateLimiter:
    """Simple sliding-window limiter: blocks until it's safe to make another call."""

    def __init__(self, max_per_second: int = MAX_REQUESTS_PER_SECOND):
        self.max_per_second = max_per_second
        self.call_times = deque()

    def wait_if_needed(self):
        now = time.time()
        while self.call_times and now - self.call_times[0] > 1.0:
            self.call_times.popleft()
        if len(self.call_times) >= self.max_per_second:
            sleep_for = 1.0 - (now - self.call_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.call_times.append(time.time())


_rate_limiter = RateLimiter()


def _chunk_date_range(start: pd.Timestamp, end: pd.Timestamp, max_days: int = MAX_DAYS_PER_REQUEST):
    """Split a date range into <=100 day pieces, since Fyers caps request size."""
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=max_days), end)
        yield chunk_start, chunk_end
        chunk_start = chunk_end


def _fetch_chunk(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One rate-limited API call for a single <=100 day chunk."""
    _rate_limiter.wait_if_needed()

    # TODO: confirm this against current Fyers API docs before going live -
    # symbol format (e.g. "NSE:RELIANCE-EQ"), exact param names, and the
    # response JSON shape may differ from what's sketched here.
    response = requests.get(
        "https://api-t1.fyers.in/data/history",
        headers={"Authorization": f"{FYERS_CLIENT_ID}:{FYERS_ACCESS_TOKEN}"},
        params={
            "symbol": symbol,
            "resolution": "1",  # 1-minute candles
            "date_format": "1",
            "range_from": start.strftime("%Y-%m-%d"),
            "range_to": end.strftime("%Y-%m-%d"),
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()

    candles = payload.get("candles", [])  # expected: [[epoch, o, h, l, c, v], ...]
    df = pd.DataFrame(candles, columns=["epoch", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["epoch"], unit="s")
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]


def fetch_fyers_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Public entry point - matches ensure_data()'s FetchFn signature exactly,
    so it can be passed straight in: ensure_data(symbol, start, end, fetch_fn=fetch_fyers_history)
    """
    chunks = [_fetch_chunk(symbol, s, e) for s, e in _chunk_date_range(start, end)]
    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(chunks).sort_index()


def refresh_access_token():
    """
    Fyers access tokens expire daily. The real OAuth login flow needs a
    one-time interactive browser login to get a refresh token; after that,
    this function (run once a day via a scheduled job, e.g. Celery beat
    before market open) exchanges the refresh token for a new access token
    and updates FYERS_ACCESS_TOKEN / wherever it's persisted (env, DB, secret store).
    Left unimplemented here - wire this up once the OAuth app is registered.
    """
    raise NotImplementedError("Wire up Fyers OAuth refresh flow once app credentials are registered")