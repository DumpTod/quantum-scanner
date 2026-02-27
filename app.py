# app.py — Quantum Scanner Flask Application

import logging
import threading
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

from config import FYERS_CLIENT_ID, SUPABASE_URL, SUPABASE_API_KEY
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

# -- Debug single stock -------------------------------------------------------
@app.route("/api/debug_scan/<symbol>")
def debug_scan(symbol: str):
    """Show exactly what happens when scanning a single stock."""
    from indicators import compute_all
    from models import run_all_models
    from risk_manager import calc_risk_params
    from backtest_engine import get_or_run_backtest
    from scorer import compute_score

    symbol = symbol.upper()
    try:
        df = fetch_ohlcv(symbol)
        if df is None:
            return jsonify({"error": "data fetch failed", "symbol": symbol})

        df = compute_all(df)
        idx    = len(df) - 1
        result = run_all_models(df, idx)
        row    = df.iloc[idx]

        risk = None
        backtest   = {}
        score_info = {}

        if result["direction"] != "NEUTRAL":
            risk = calc_risk_params(
                float(row["Close"]),
                float(row.get("atr14", 0)),
                float(row.get("atr_pct", 50)),
                result["direction"]
            )
            backtest   = get_or_run_backtest(df, symbol, result["direction"])
            score_info = compute_score(result, backtest)

        return jsonify({
            "symbol":      symbol,
            "candles":     len(df),
            "direction":   result["direction"],
            "models_buy":  result["models_buy"],
            "models_sell": result["models_sell"],
            "model_score": result.get("model_score"),
            "models":      {k: v["signal"] + f" ({v['score']}/5)" for k, v in result.get("models", {}).items()},
            "risk":        risk,
            "backtest":    backtest,
            "score_info":  score_info,
            "last_close":  round(float(row["Close"]), 2),
            "rsi":         round(float(row.get("rsi", 0)), 1),
            "atr_pct":     round(float(row.get("atr_pct", 0)), 1),
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

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



# -- Clear duplicate backtest trades -----------------------------------------
@app.route("/api/cleanup_dupes")
def cleanup_dupes():
    """Remove duplicate rows from backtest_trades keeping only one per symbol/direction/entry_bar."""
    import requests as req
    headers = {
        "apikey":        SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type":  "application/json",
    }
    try:
        # Fetch all trades
        r = req.get(
            f"{SUPABASE_URL}/rest/v1/backtest_trades?select=id,symbol,direction,entry_bar&order=id.asc&limit=2000",
            headers=headers, timeout=15
        )
        rows = r.json()
        if not isinstance(rows, list):
            return jsonify({"status": "error", "response": rows})

        # Find duplicate IDs (keep first occurrence)
        seen = {}
        dupe_ids = []
        for row in rows:
            key = (row["symbol"], row["direction"], row["entry_bar"])
            if key in seen:
                dupe_ids.append(str(row["id"]))
            else:
                seen[key] = row["id"]

        if not dupe_ids:
            return jsonify({"status": "ok", "total_rows": len(rows), "duplicates_removed": 0})

        # Bulk delete all dupes in one request using IN filter
        ids_param = ",".join(dupe_ids)
        del_r = req.delete(
            f"{SUPABASE_URL}/rest/v1/backtest_trades?id=in.({ids_param})",
            headers=headers, timeout=20
        )
        return jsonify({
            "status":            "ok",
            "total_rows":        len(rows),
            "duplicates_removed": len(dupe_ids),
            "delete_status":     del_r.status_code,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


# -- Live prices for rescan ---------------------------------------------------
@app.route("/api/prices", methods=["POST"])
def get_prices():
    """
    Accept list of {symbol, from_date} and return daily OHLCV
    candles after from_date for each symbol.
    Used by history.html Rescan feature.
    Body: {"requests": [{"symbol": "HDFCBANK", "from_date": "2026-02-25"}]}
    """
    import requests as req
    from datetime import datetime, timedelta

    body = request.get_json()
    if not body or "requests" not in body:
        return jsonify({"error": "Missing requests array"}), 400

    results = {}
    for item in body["requests"]:
        symbol    = item.get("symbol", "").upper()
        from_date = item.get("from_date", "")
        if not symbol or not from_date:
            continue
        try:
            df = fetch_ohlcv(symbol)
            if df is None:
                results[symbol] = {"error": "fetch failed"}
                continue

            # Filter to only candles after from_date
            cutoff = pd.Timestamp(from_date)
            df_filtered = df[df["timestamp"] > cutoff].copy()

            if df_filtered.empty:
                results[symbol] = {"candles": [], "latest_close": None}
                continue

            candles = []
            for _, row in df_filtered.iterrows():
                candles.append({
                    "date":  row["timestamp"].strftime("%Y-%m-%d"),
                    "open":  round(float(row["Open"]),  2),
                    "high":  round(float(row["High"]),  2),
                    "low":   round(float(row["Low"]),   2),
                    "close": round(float(row["Close"]), 2),
                })

            results[symbol] = {
                "candles":      candles,
                "latest_close": candles[-1]["close"] if candles else None,
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}

    return jsonify({"results": results})

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
