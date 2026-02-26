# scanner.py — Orchestrates the full scan pipeline

import logging
from datetime import datetime
from data_fetcher import fetch_ohlcv
from indicators import compute_all
from models import run_all_models
from risk_manager import calc_risk_params
from backtest_engine import get_or_run_backtest
from scorer import compute_score
from symbols import ALL_STOCKS, FNO_SET

logger = logging.getLogger(__name__)

# ── Shared state for background scan ────────────────────────────────────────
scan_state = {
    "running":    False,
    "started_at": None,
    "completed":  False,
    "total":      0,
    "scanned":    0,
    "results":    [],
    "errors":     [],
}


def scan_symbol(symbol: str) -> dict | None:
    """
    Full pipeline for one symbol.
    Returns signal dict or None if no A+ grade signal.
    """
    is_fno = symbol in FNO_SET

    df = fetch_ohlcv(symbol)
    if df is None:
        return None

    df = compute_all(df)

    idx    = len(df) - 1
    result = run_all_models(df, idx)

    direction = result["direction"]
    if direction == "NEUTRAL":
        return None

    # SELL only for FNO stocks
    if direction == "SELL" and not is_fno:
        return None

    row      = df.iloc[idx]
    entry    = row["Close"]
    atr_val  = row.get("atr14", 0)
    atr_pct  = row.get("atr_pct", 50)

    risk = calc_risk_params(entry, atr_val, atr_pct, direction)
    if risk is None:
        return None

    # Backtest (cached if available)
    backtest = get_or_run_backtest(df, symbol, direction)

    score_info = compute_score(result, backtest)

    if not score_info["include"]:
        return None

    return {
        "symbol":      symbol,
        "direction":   direction,
        "is_fno":      is_fno,
        "entry_price": risk["entry_price"],
        "stop_loss":   risk["stop_loss"],
        "target":      risk["target"],
        "sl_pct":      risk["sl_pct"],
        "rr":          risk["rr"],
        "grade":       score_info["grade"],
        "total_score": score_info["total_score"],
        "model_score": score_info["model_score"],
        "models_buy":  result["models_buy"],
        "models_sell": result["models_sell"],
        "win_rate":    backtest.get("win_rate", 0),
        "total_trades":backtest.get("total_trades", 0),
        "expectancy":  backtest.get("expectancy", 0),
        "models":      result.get("models", {}),
        "scanned_at":  datetime.utcnow().isoformat(),
    }


def run_scan(max_stocks: int | None = None):
    """
    Main scan loop — intended to run in a background thread.
    Updates scan_state in place.
    """
    global scan_state

    stocks = ALL_STOCKS if max_stocks is None else ALL_STOCKS[:max_stocks]

    scan_state.update({
        "running":    True,
        "started_at": datetime.utcnow().isoformat(),
        "completed":  False,
        "total":      len(stocks),
        "scanned":    0,
        "results":    [],
        "errors":     [],
    })

    for symbol in stocks:
        try:
            signal = scan_symbol(symbol)
            if signal:
                scan_state["results"].append(signal)
                logger.info(f"✅ {symbol} → {signal['direction']} {signal['grade']} ({signal['total_score']})")
        except Exception as e:
            scan_state["errors"].append({"symbol": symbol, "error": str(e)})
            logger.error(f"❌ {symbol}: {e}")
        finally:
            scan_state["scanned"] += 1

    # Sort by score descending
    scan_state["results"].sort(key=lambda x: x["total_score"], reverse=True)
    scan_state["running"]   = False
    scan_state["completed"] = True
    logger.info(
        f"Scan complete. {len(scan_state['results'])} signals from {scan_state['scanned']} stocks."
    )
