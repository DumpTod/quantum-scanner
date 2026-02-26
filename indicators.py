# indicators.py — Custom indicators (no pandas_ta dependency)

import numpy as np
import pandas as pd


# ── Moving averages ──────────────────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ── RSI ──────────────────────────────────────────────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ─────────────────────────────────────────────────────────────────────
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    e_fast   = ema(series, fast)
    e_slow   = ema(series, slow)
    macd_line = e_fast - e_slow
    sig_line  = ema(macd_line, signal)
    histogram = macd_line - sig_line
    return macd_line, sig_line, histogram


# ── ATR ──────────────────────────────────────────────────────────────────────
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ── Bollinger Bands ──────────────────────────────────────────────────────────
def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0):
    mid   = sma(series, period)
    sigma = series.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    return upper, mid, lower


# ── Keltner Channel ──────────────────────────────────────────────────────────
def keltner_channel(high, low, close, ema_period=20, atr_period=10, mult=1.5):
    mid   = ema(close, ema_period)
    _atr  = atr(high, low, close, atr_period)
    upper = mid + mult * _atr
    lower = mid - mult * _atr
    return upper, mid, lower


# ── Z-Score ──────────────────────────────────────────────────────────────────
def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period).mean()
    std  = series.rolling(period).std()
    return (series - mean) / std.replace(0, np.nan)


# ── OBV ──────────────────────────────────────────────────────────────────────
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ── VWAP (rolling daily proxy using all available bars) ──────────────────────
def vwap(high, low, close, volume, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    tp_vol  = typical * volume
    rolling_tpvol = tp_vol.rolling(period).sum()
    rolling_vol   = volume.rolling(period).sum()
    return rolling_tpvol / rolling_vol.replace(0, np.nan)


# ── Volume Delta Ratio (proxy: up-close bars vs down-close bars vol ratio) ───
def volume_delta_ratio(close: pd.Series, volume: pd.Series, period: int = 10) -> pd.Series:
    up_vol   = volume.where(close > close.shift(1), 0.0)
    down_vol = volume.where(close < close.shift(1), 0.0)
    up_sum   = up_vol.rolling(period).sum()
    down_sum = down_vol.rolling(period).sum()
    return up_sum / down_sum.replace(0, np.nan)


# ── ATR Percentile ───────────────────────────────────────────────────────────
def atr_percentile(atr_series: pd.Series, period: int = 100) -> pd.Series:
    def _pct(x):
        valid = x[~np.isnan(x)]
        if len(valid) < 2:
            return np.nan
        return (np.sum(valid[:-1] < valid[-1]) / (len(valid) - 1)) * 100
    return atr_series.rolling(period).apply(_pct, raw=True)


# ── Fibonacci levels from last N bars ────────────────────────────────────────
def fibonacci_levels(high: pd.Series, low: pd.Series, lookback: int = 60):
    h = high.iloc[-lookback:].max()
    l = low.iloc[-lookback:].min()
    rng = h - l
    return {
        "0":    l,
        "23.6": l + 0.236 * rng,
        "38.2": l + 0.382 * rng,
        "50":   l + 0.500 * rng,
        "61.8": l + 0.618 * rng,
        "78.6": l + 0.786 * rng,
        "100":  h,
    }


# ── Historical Volatility (20-day) ───────────────────────────────────────────
def hv20(close: pd.Series) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(20).std() * np.sqrt(252) * 100


# ── Master compute function ───────────────────────────────────────────────────
def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators and attach as new columns.
    Input df must have: Open, High, Low, Close, Volume
    """
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # EMAs
    df["ema9"]  = ema(c, 9)
    df["ema21"] = ema(c, 21)
    # Weekly/Monthly proxies
    df["ema45"]  = ema(c, 45)   # ~weekly EMA9
    df["ema105"] = ema(c, 105)  # ~weekly EMA21
    df["ema200"] = ema(c, 200)  # ~monthly EMA

    # RSI
    df["rsi"] = rsi(c, 14)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(c)
    df["macd_accel"] = df["macd_hist"].diff()

    # ATR
    df["atr14"] = atr(h, l, c, 14)
    df["atr_pct"] = atr_percentile(df["atr14"], 100)

    # Bollinger Bands
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(c, 20, 2.0)

    # Keltner Channel
    df["kc_upper"], df["kc_mid"], df["kc_lower"] = keltner_channel(h, l, c)

    # Z-Score
    df["zscore"] = zscore(c, 20)

    # OBV
    df["obv"]     = obv(c, v)
    df["obv_sma"] = sma(df["obv"], 20)

    # VWAP
    df["vwap"] = vwap(h, l, c, v, 20)

    # Volume SMA
    df["vol_sma20"] = sma(v, 20)

    # Volume delta ratio
    df["vol_delta_ratio"] = volume_delta_ratio(c, v, 10)

    # HV20
    df["hv20"] = hv20(c)

    return df
