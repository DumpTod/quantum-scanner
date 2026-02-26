# risk_manager.py — Stop Loss, Target, RR calculation

from config import (
    ATR_MULT_SL, MIN_SL_PCT, MAX_SL_PCT,
    MIN_RR, RR_LOW_VOL, RR_NORMAL_VOL, RR_HIGH_VOL
)


def calc_risk_params(entry_price: float, atr: float, atr_pct: float,
                     direction: str) -> dict | None:
    """
    Calculate stop_loss, target, rr for a given entry.
    Returns None if RR is below minimum.
    """
    if entry_price <= 0 or atr <= 0:
        return None

    # SL based on ATR, clamped between min/max pct
    atr_sl_pct   = (atr * ATR_MULT_SL) / entry_price * 100
    sl_pct       = max(MIN_SL_PCT, min(MAX_SL_PCT, atr_sl_pct))
    sl_distance  = entry_price * sl_pct / 100

    # R:R ratio varies by volatility regime
    if atr_pct < 30:
        rr = RR_LOW_VOL
    elif atr_pct < 70:
        rr = RR_NORMAL_VOL
    else:
        rr = RR_HIGH_VOL

    rr = max(rr, MIN_RR)

    if direction == "BUY":
        stop_loss = entry_price - sl_distance
        target    = entry_price + sl_distance * rr
    elif direction == "SELL":
        stop_loss = entry_price + sl_distance
        target    = entry_price - sl_distance * rr
    else:
        return None

    actual_rr = abs(target - entry_price) / sl_distance

    return {
        "entry_price": round(entry_price, 2),
        "stop_loss":   round(stop_loss, 2),
        "target":      round(target, 2),
        "sl_pct":      round(sl_pct, 2),
        "rr":          round(actual_rr, 2),
    }
