# backtest_engine.py — Walk-forward backtester (no look-ahead bias)

import logging
import requests
from datetime import datetime
from config import (
    SUPABASE_URL, SUPABASE_API_KEY,
    BACKTEST_STEP, BACKTEST_MAX_BARS_HELD, BACKTEST_MAX_TRADES, BACKTEST_MIN_CANDLES
)
from models import run_all_models
from risk_manager import calc_risk_params

logger = logging.getLogger(__name__)


# ── Supabase helpers ─────────────────────────────────────────────────────────
def _sb_headers():
    return {
        "apikey":        SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

def _sb_url(path):
    return f"{SUPABASE_URL}/rest/v1/{path}"


def get_cached_backtest(symbol: str, direction: str) -> dict | None:
    try:
        r = requests.get(
            _sb_url(f"backtest_cache?symbol=eq.{symbol}&direction=eq.{direction}&select=*"),
            headers=_sb_headers(), timeout=10
        )
        rows = r.json()
        return rows[0] if rows else None
    except Exception as e:
        logger.error(f"get_cached_backtest: {e}")
        return None


def save_backtest_summary(symbol: str, direction: str, summary: dict):
    payload = {
        "symbol":         symbol,
        "direction":      direction,
        "win_rate":       summary["win_rate"],
        "total_trades":   summary["total_trades"],
        "wins":           summary["wins"],
        "losses":         summary["losses"],
        "avg_profit_pct": summary["avg_profit_pct"],
        "avg_loss_pct":   summary["avg_loss_pct"],
        "expectancy":     summary["expectancy"],
        "updated_at":     datetime.utcnow().isoformat(),
    }
    try:
        r = requests.post(
            _sb_url("backtest_cache"),
            headers={**_sb_headers(), "Prefer": "resolution=merge-duplicates,return=representation"},
            json=payload, timeout=10
        )
        logger.info(f"save_backtest_summary {symbol}/{direction}: {r.status_code}")
    except Exception as e:
        logger.error(f"save_backtest_summary: {e}")


def save_backtest_trades(symbol: str, direction: str, trades: list):
    if not trades:
        return
    # Delete old trades first
    try:
        requests.delete(
            _sb_url(f"backtest_trades?symbol=eq.{symbol}&direction=eq.{direction}"),
            headers=_sb_headers(), timeout=10
        )
    except Exception:
        pass

    for t in trades:
        t["symbol"]    = symbol
        t["direction"] = direction

    try:
        r = requests.post(
            _sb_url("backtest_trades"),
            headers=_sb_headers(),
            json=trades, timeout=15
        )
        logger.info(f"save_backtest_trades {symbol}: {r.status_code}")
    except Exception as e:
        logger.error(f"save_backtest_trades: {e}")


def get_backtest_trades(symbol: str, direction: str) -> list:
    try:
        r = requests.get(
            _sb_url(f"backtest_trades?symbol=eq.{symbol}&direction=eq.{direction}&select=*&order=entry_bar.asc"),
            headers=_sb_headers(), timeout=10
        )
        return r.json() if r.ok else []
    except Exception as e:
        logger.error(f"get_backtest_trades: {e}")
        return []


# ── Walk-forward engine ──────────────────────────────────────────────────────
def run_backtest(df, symbol: str, direction: str) -> dict:
    """
    Walk-forward backtest on pre-computed indicator DataFrame.
    direction: 'BUY' or 'SELL'
    Returns summary dict.
    """
    n      = len(df)
    trades = []

    bar = BACKTEST_MIN_CANDLES
    while bar < n and len(trades) < BACKTEST_MAX_TRADES:
        result = run_all_models(df, bar)

        if result["direction"] != direction:
            bar += BACKTEST_STEP
            continue

        entry_price = df.iloc[bar]["Close"]
        atr_val     = df.iloc[bar].get("atr14", 0)
        atr_pct     = df.iloc[bar].get("atr_pct", 50)

        risk = calc_risk_params(entry_price, atr_val, atr_pct, direction)
        if risk is None:
            bar += BACKTEST_STEP
            continue

        sl     = risk["stop_loss"]
        target = risk["target"]

        # Simulate forward
        exit_bar   = None
        exit_price = None
        outcome    = "timeout"

        for fwd in range(bar + 1, min(bar + BACKTEST_MAX_BARS_HELD + 1, n)):
            h = df.iloc[fwd]["High"]
            l = df.iloc[fwd]["Low"]
            c = df.iloc[fwd]["Close"]

            if direction == "BUY":
                if l <= sl:
                    exit_bar, exit_price, outcome = fwd, sl, "loss"
                    break
                if h >= target:
                    exit_bar, exit_price, outcome = fwd, target, "win"
                    break
            else:  # SELL
                if h >= sl:
                    exit_bar, exit_price, outcome = fwd, sl, "loss"
                    break
                if l <= target:
                    exit_bar, exit_price, outcome = fwd, target, "win"
                    break

        if exit_bar is None:
            exit_bar   = min(bar + BACKTEST_MAX_BARS_HELD, n - 1)
            exit_price = df.iloc[exit_bar]["Close"]
            outcome    = "timeout"

        if direction == "BUY":
            profit_pct = (exit_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - exit_price) / entry_price * 100

        entry_ts = df.iloc[bar]["timestamp"]
        exit_ts  = df.iloc[exit_bar]["timestamp"]

        trades.append({
            "entry_price": round(entry_price, 2),
            "stop_loss":   round(sl, 2),
            "target":      round(target, 2),
            "exit_price":  round(exit_price, 2),
            "outcome":     outcome,
            "profit_pct":  round(profit_pct, 2),
            "entry_bar":   bar,
            "exit_bar":    exit_bar,
            "bars_held":   exit_bar - bar,
            "entry_date":  entry_ts.strftime("%d/%m/%Y") if hasattr(entry_ts, "strftime") else str(entry_ts),
            "exit_date":   exit_ts.strftime("%d/%m/%Y")  if hasattr(exit_ts, "strftime")  else str(exit_ts),
        })

        bar = exit_bar + 1

    # Summary
    wins    = [t for t in trades if t["outcome"] == "win"]
    losses  = [t for t in trades if t["outcome"] in ("loss", "timeout") and t["profit_pct"] < 0]
    n_total = len(trades)
    n_wins  = len(wins)
    n_loss  = n_total - n_wins

    win_rate       = round(n_wins / n_total * 100, 1) if n_total else 0
    avg_profit_pct = round(sum(t["profit_pct"] for t in wins) / len(wins), 2) if wins else 0
    avg_loss_pct   = round(sum(t["profit_pct"] for t in losses) / len(losses), 2) if losses else 0
    expectancy     = round(
        (win_rate / 100) * avg_profit_pct + ((1 - win_rate / 100) * avg_loss_pct), 2
    )

    summary = {
        "win_rate":       win_rate,
        "total_trades":   n_total,
        "wins":           n_wins,
        "losses":         n_loss,
        "avg_profit_pct": avg_profit_pct,
        "avg_loss_pct":   avg_loss_pct,
        "expectancy":     expectancy,
    }
    return summary, trades


def get_or_run_backtest(df, symbol: str, direction: str) -> dict:
    """
    Check Supabase cache first; run and cache if missing.
    Returns summary dict (never None).
    """
    cached = get_cached_backtest(symbol, direction)
    if cached:
        logger.info(f"Backtest cache hit: {symbol}/{direction}")
        return cached

    logger.info(f"Running backtest: {symbol}/{direction}")
    summary, trades = run_backtest(df, symbol, direction)
    save_backtest_summary(symbol, direction, summary)
    save_backtest_trades(symbol, direction, trades)
    return summary
