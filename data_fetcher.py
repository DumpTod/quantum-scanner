# data_fetcher.py — Fyers SDK data fetch with caching

import logging
import pandas as pd
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from config import FYERS_CLIENT_ID, DATA_PERIOD_DAYS, BACKTEST_MIN_CANDLES, to_fyers_symbol
from token_manager import get_access_token

logger = logging.getLogger(__name__)

# Simple in-process cache: {symbol: (df, fetched_at)}
_data_cache: dict = {}
CACHE_TTL_HOURS = 4


def _get_fyers():
    token = get_access_token()
    if not token:
        raise RuntimeError("No access token available")
    return fyersModel.FyersModel(
        client_id=FYERS_CLIENT_ID,
        token=token,
        is_async=False,
        log_path=""
    )


def fetch_ohlcv(symbol: str, force: bool = False) -> pd.DataFrame | None:
    """
    Fetch daily OHLCV for NSE symbol.
    Returns DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
    or None on failure.
    """
    now = datetime.utcnow()

    # Cache hit?
    if not force and symbol in _data_cache:
        df, fetched_at = _data_cache[symbol]
        if (now - fetched_at).total_seconds() < CACHE_TTL_HOURS * 3600:
            return df

    fyers_sym = to_fyers_symbol(symbol)
    date_to   = now.strftime("%Y-%m-%d")
    date_from = (now - timedelta(days=DATA_PERIOD_DAYS)).strftime("%Y-%m-%d")

    try:
        fyers = _get_fyers()
        response = fyers.history(data={
            "symbol":      fyers_sym,
            "resolution":  "D",
            "date_format": "1",
            "range_from":  date_from,
            "range_to":    date_to,
            "cont_flag":   "1",
        })

        if response.get("s") != "ok":
            logger.error(f"{symbol}: Fyers error — {response}")
            return None

        candles = response.get("candles", [])
        if len(candles) < BACKTEST_MIN_CANDLES:
            logger.warning(f"{symbol}: Only {len(candles)} candles, need {BACKTEST_MIN_CANDLES}")
            return None

        df = pd.DataFrame(candles, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        _data_cache[symbol] = (df, now)
        return df

    except Exception as e:
        logger.error(f"{symbol}: fetch_ohlcv exception — {e}")
        return None


def clear_cache():
    _data_cache.clear()
