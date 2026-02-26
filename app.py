# app.py — Quantum Scanner Flask Application

import logging
import threading
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

from config import FYERS_CLIENT_ID
from token_manager import get_access_token, read_tokens_from_db
from data_fetcher import fetch_ohlcv
from scanner import run_scan, scan_state
from backtest_engine import get_backtest_trades
from symbols import ALL_STOCKS, FNO_STOCKS


# ── Health & Info ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({
        "name":      "Quantum Scanner",
        "version":   "2.0",
        "universe":  f"{len(ALL_STOCKS)} stocks | {len(FNO_STOCKS)} FNO",
        "endpoints": [
            "GET /api/health",
            "GET /api/token",
            "GET /api/debug",
            "GET /api/scan",
            "GET /api/scan?max=N",
            "GET /api/results",
            "GET /api/backtest/<symbol>/<direction>",
        ],
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# ── Token management ─────────────────────────────────────────────────────────
@app.route("/api/token")
def token_refresh():
    """Force refresh access token and return masked result."""
    try:
        token = get_access_token(force=True)
        if token:
            return jsonify({
                "status":  "ok",
                "token":   token[:10] + "..." + token[-5:],
                "message": "Token refreshed successfully",
            })
        return jsonify({"status": "error", "message": "Token refresh failed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Debug ─────────────────────────────────────────────────────────────────────
@app.route("/api/debug")
def debug():
    """Test token, data fetch for RELIANCE, and Supabase connectivity."""
    results = {}

    # Token check
    token = None
    try:
        token = get_access_token()
        results["token"] = "ok" if token else "missing"
        results["token_preview"] = token[:20] + "..." if token else None
    except Exception as e:
        results["token"] = f"error: {e}"

    # Raw Fyers SDK test
    try:
        from fyers_apiv3 import fyersModel
        from datetime import datetime, timedelta
        fyers = fyersModel.FyersModel(
            client_id=FYERS_CLIENT_ID,
            token=token,
            is_async=False,
            log_path=""
        )
        now = datetime.utcnow()
        response = fyers.history(data={
            "symbol":      "NSE:RELIANCE-EQ",
            "resolution":  "D",
            "date_format": "1",
            "range_from":  (now - timedelta(days=30)).strftime("%Y-%m-%d"),
            "range_to":    now.strftime("%Y-%m-%d"),
            "cont_flag":   "1",
        })
        results["fyers_raw"] = {
            "status":   response.get("s"),
            "message":  response.get("message", ""),
            "code":     response.get("code", ""),
            "candles":  len(response.get("candles", [])),
        }
    except Exception as e:
        results["fyers_raw"] = f"error: {e}"

    # Supabase check
    try:
        refresh_token, access_token_db = read_tokens_from_db()
        results["supabase"] = {
            "status":      "ok",
            "has_refresh": bool(refresh_token),
            "has_access":  bool(access_token_db),
        }
    except Exception as e:
        results["supabase"] = f"error: {e}"

    return jsonify(results)


# ── Scanner ──────────────────────────────────────────────────────────────────
@app.route("/api/scan")
def start_scan():
    """Start scan in background thread. Returns immediately."""
    if scan_state["running"]:
        return jsonify({
            "status":  "already_running",
            "scanned": scan_state["scanned"],
            "total":   scan_state["total"],
        })

    max_stocks = request.args.get("max", type=int)

    t = threading.Thread(
        target=run_scan,
        args=(max_stocks,),
        daemon=True
    )
    t.start()

    return jsonify({
        "status":  "started",
        "message": f"Scanning {'all' if max_stocks is None else max_stocks} stocks in background",
        "poll":    "/api/results",
    })


@app.route("/api/results")
def get_results():
    """Return current scan state and results."""
    return jsonify({
        "running":    scan_state["running"],
        "completed":  scan_state["completed"],
        "started_at": scan_state["started_at"],
        "scanned":    scan_state["scanned"],
        "total":      scan_state["total"],
        "signals":    len(scan_state["results"]),
        "errors":     len(scan_state["errors"]),
        "results":    scan_state["results"],
    })


# ── Backtest trades ──────────────────────────────────────────────────────────
@app.route("/api/backtest/<symbol>/<direction>")
def backtest_detail(symbol: str, direction: str):
    """Return all backtest trades from Supabase for a symbol/direction."""
    direction = direction.upper()
    if direction not in ("BUY", "SELL"):
        return jsonify({"error": "direction must be BUY or SELL"}), 400

    trades = get_backtest_trades(symbol.upper(), direction)
    return jsonify({
        "symbol":    symbol.upper(),
        "direction": direction,
        "count":     len(trades),
        "trades":    trades,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
