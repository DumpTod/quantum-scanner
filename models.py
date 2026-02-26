# models.py — 5 Quantitative Models

import numpy as np
import pandas as pd
from indicators import fibonacci_levels


# ── Helpers ──────────────────────────────────────────────────────────────────
def _near(price: float, level: float, pct: float = 1.5) -> bool:
    return abs(price - level) / max(level, 1e-9) * 100 <= pct


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Mean Reversion (MRB)
# ─────────────────────────────────────────────────────────────────────────────
def model_mean_reversion(row: pd.Series, prev: pd.Series) -> str:
    """Return 'BUY', 'SELL', or 'NEUTRAL' (need 3/5 conditions)."""
    c = row["Close"]

    buy_conds = [
        row["zscore"]    < -1,
        c <= row["bb_lower"] * 1.02,
        c <  row["kc_lower"] * 1.03,
        row["rsi"]       < 40,
        c > prev["Close"],
    ]
    sell_conds = [
        row["zscore"]    > 1,
        c >= row["bb_upper"] * 0.98,
        c >  row["kc_upper"] * 0.97,
        row["rsi"]       > 60,
        c < prev["Close"],
    ]

    buy_score  = sum(bool(x) for x in buy_conds)
    sell_score = sum(bool(x) for x in sell_conds)

    if buy_score >= 3:
        return "BUY", buy_score
    if sell_score >= 3:
        return "SELL", sell_score
    return "NEUTRAL", max(buy_score, sell_score)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Momentum Decay (MDD)
# ─────────────────────────────────────────────────────────────────────────────
def model_momentum_decay(row: pd.Series, prev: pd.Series) -> tuple:
    c = row["Close"]

    buy_conds = [
        row["macd_hist"]  > prev["macd_hist"],
        row["macd_accel"] > 0,
        row["rsi"]        > prev["rsi"],
        40 < row["rsi"] < 65,
        c > prev["Close"],
    ]
    sell_conds = [
        row["macd_hist"]  < prev["macd_hist"],
        row["macd_accel"] < 0,
        row["rsi"]        < prev["rsi"],
        35 < row["rsi"] < 60,
        c < prev["Close"],
    ]

    buy_score  = sum(bool(x) for x in buy_conds)
    sell_score = sum(bool(x) for x in sell_conds)

    if buy_score >= 3:
        return "BUY", buy_score
    if sell_score >= 3:
        return "SELL", sell_score
    return "NEUTRAL", max(buy_score, sell_score)


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — Volatility Regime (VRC)
# ─────────────────────────────────────────────────────────────────────────────
def model_volatility_regime(row: pd.Series, prev: pd.Series) -> tuple:
    atr_pct = row.get("atr_pct", 50)
    c       = row["Close"]

    if atr_pct < 30:
        regime = "LOW_VOL"
    elif atr_pct < 70:
        regime = "NORMAL_VOL"
    else:
        regime = "HIGH_VOL"

    if regime == "LOW_VOL":
        buy_conds = [
            row["zscore"]   < -0.5,
            c < row["bb_lower"] * 1.03,
            row["rsi"]      < 45,
            c > prev["Close"],
            row["vol_delta_ratio"] > 1.1,
        ]
        sell_conds = [
            row["zscore"]   > 0.5,
            c > row["bb_upper"] * 0.97,
            row["rsi"]      > 55,
            c < prev["Close"],
            row["vol_delta_ratio"] < 0.9,
        ]
    elif regime == "NORMAL_VOL":
        buy_conds = [
            row["ema9"]     > row["ema21"],
            row["macd_hist"] > 0,
            row["rsi"]      > 45,
            c > row["vwap"],
            row["obv"]      > row["obv_sma"],
        ]
        sell_conds = [
            row["ema9"]     < row["ema21"],
            row["macd_hist"] < 0,
            row["rsi"]      < 55,
            c < row["vwap"],
            row["obv"]      < row["obv_sma"],
        ]
    else:  # HIGH_VOL
        buy_conds = [
            row["zscore"]   < -1.5,
            c < row["bb_lower"],
            row["rsi"]      < 35,
            row["vol_delta_ratio"] > 1.5,
            c > prev["Close"],
        ]
        sell_conds = [
            row["zscore"]   > 1.5,
            c > row["bb_upper"],
            row["rsi"]      > 65,
            row["vol_delta_ratio"] < 0.65,
            c < prev["Close"],
        ]

    buy_score  = sum(bool(x) for x in buy_conds)
    sell_score = sum(bool(x) for x in sell_conds)

    if buy_score >= 3:
        return "BUY", buy_score, regime
    if sell_score >= 3:
        return "SELL", sell_score, regime
    return "NEUTRAL", max(buy_score, sell_score), regime


# ─────────────────────────────────────────────────────────────────────────────
# Model 4 — Order Flow (OFI)
# ─────────────────────────────────────────────────────────────────────────────
def model_order_flow(row: pd.Series, prev: pd.Series) -> tuple:
    c = row["Close"]

    buy_conds = [
        row["vol_delta_ratio"] > 1.3,
        row["obv"]             > row["obv_sma"],
        c                      > row["vwap"],
        row["Volume"]          > row["vol_sma20"] * 0.9,
        c                      > row["Open"],
    ]
    sell_conds = [
        row["vol_delta_ratio"] < 0.75,
        row["obv"]             < row["obv_sma"],
        c                      < row["vwap"],
        row["Volume"]          > row["vol_sma20"] * 0.9,
        c                      < row["Open"],
    ]

    buy_score  = sum(bool(x) for x in buy_conds)
    sell_score = sum(bool(x) for x in sell_conds)

    if buy_score >= 3:
        return "BUY", buy_score
    if sell_score >= 3:
        return "SELL", sell_score
    return "NEUTRAL", max(buy_score, sell_score)


# ─────────────────────────────────────────────────────────────────────────────
# Model 5 — Multi-Timeframe (MTC)
# ─────────────────────────────────────────────────────────────────────────────
def model_multitimeframe(row: pd.Series, prev: pd.Series, df_slice: pd.DataFrame) -> tuple:
    c    = row["Close"]
    fibs = fibonacci_levels(df_slice["High"], df_slice["Low"], lookback=60)

    near_support    = any(_near(c, fibs[k]) for k in ["23.6", "38.2", "50"])
    near_resistance = any(_near(c, fibs[k]) for k in ["50", "61.8", "78.6", "100"])

    buy_conds = [
        row["ema9"]   > row["ema21"],
        row["ema45"]  > row["ema105"],   # weekly EMA9 > weekly EMA21
        c             > row["ema200"],   # above monthly EMA
        near_support,
        35 < row["rsi"] < 70,
    ]
    sell_conds = [
        row["ema9"]   < row["ema21"],
        row["ema45"]  < row["ema105"],
        c             < row["ema200"],
        near_resistance,
        30 < row["rsi"] < 65,
    ]

    buy_score  = sum(bool(x) for x in buy_conds)
    sell_score = sum(bool(x) for x in sell_conds)

    if buy_score >= 3:
        return "BUY", buy_score
    if sell_score >= 3:
        return "SELL", sell_score
    return "NEUTRAL", max(buy_score, sell_score)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate signal from all 5 models
# ─────────────────────────────────────────────────────────────────────────────
def run_all_models(df: pd.DataFrame, idx: int) -> dict:
    """
    Run all 5 models on df up to and including bar `idx`.
    Returns dict with direction, model votes, scores, point counts.
    """
    if idx < 1:
        return {"direction": "NEUTRAL", "models_buy": 0, "models_sell": 0}

    row      = df.iloc[idx]
    prev     = df.iloc[idx - 1]
    df_slice = df.iloc[: idx + 1]

    results = {}

    # Model 1
    sig1, s1 = model_mean_reversion(row, prev)
    results["mrb"] = {"signal": sig1, "score": s1}

    # Model 2
    sig2, s2 = model_momentum_decay(row, prev)
    results["mdd"] = {"signal": sig2, "score": s2}

    # Model 3
    sig3, s3, regime = model_volatility_regime(row, prev)
    results["vrc"] = {"signal": sig3, "score": s3, "regime": regime}

    # Model 4
    sig4, s4 = model_order_flow(row, prev)
    results["ofi"] = {"signal": sig4, "score": s4}

    # Model 5
    sig5, s5 = model_multitimeframe(row, prev, df_slice)
    results["mtc"] = {"signal": sig5, "score": s5}

    all_sigs = [r["signal"] for r in results.values()]
    buy_count  = all_sigs.count("BUY")
    sell_count = all_sigs.count("SELL")

    if buy_count >= 3:
        direction = "BUY"
    elif sell_count >= 3:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    # Model score = (total points across all models) / (5 models × 5 max pts) × 100
    total_points    = sum(r["score"] for r in results.values())
    max_points      = 25  # 5 models × 5 conditions
    model_score_pct = round(total_points / max_points * 100, 1)

    return {
        "direction":    direction,
        "models_buy":   buy_count,
        "models_sell":  sell_count,
        "models":       results,
        "model_score":  model_score_pct,
    }
