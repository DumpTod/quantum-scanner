# token_manager.py — Fyers Token Management (daily refresh from Supabase)

import hashlib
import requests
import logging
from datetime import date, datetime
from config import (
    FYERS_CLIENT_ID, FYERS_SECRET_KEY, FYERS_PIN,
    SUPABASE_URL, SUPABASE_API_KEY
)

logger = logging.getLogger(__name__)

# ── In-memory cache ──────────────────────────────────────────────────────────
_cache = {"access_token": None, "date": None}


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


def read_tokens_from_db():
    """Read refresh_token and access_token from Supabase."""
    try:
        r = requests.get(
            _sb_url("fyers_tokens?id=eq.1&select=refresh_token,access_token"),
            headers=_sb_headers(), timeout=10
        )
        rows = r.json()
        if rows:
            return rows[0].get("refresh_token"), rows[0].get("access_token")
    except Exception as e:
        logger.error(f"read_tokens_from_db: {e}")
    return None, None


def save_tokens_to_db(refresh_token=None, access_token=None):
    """Upsert tokens into Supabase."""
    payload = {"id": 1, "updated_at": datetime.utcnow().isoformat()}
    if refresh_token:
        payload["refresh_token"] = refresh_token
    if access_token:
        payload["access_token"] = access_token
    try:
        r = requests.post(
            _sb_url("fyers_tokens"),
            headers={**_sb_headers(), "Prefer": "resolution=merge-duplicates,return=representation"},
            json=payload, timeout=10
        )
        logger.info(f"save_tokens_to_db: {r.status_code}")
        return r.status_code in (200, 201)
    except Exception as e:
        logger.error(f"save_tokens_to_db: {e}")
        return False


# ── Token refresh ────────────────────────────────────────────────────────────
def _sha256_appid_secret():
    raw = f"{FYERS_CLIENT_ID}:{FYERS_SECRET_KEY}"
    return hashlib.sha256(raw.encode()).hexdigest()


def refresh_access_token(refresh_token: str) -> str | None:
    """Call Fyers v3 token refresh endpoint."""
    url  = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
    data = {
        "grant_type":    "refresh_token",
        "appIdHash":     _sha256_appid_secret(),
        "refresh_token": refresh_token,
        "pin":           FYERS_PIN,
    }
    try:
        r = requests.post(url, json=data, timeout=15)
        res = r.json()
        logger.info(f"Token refresh response: {res.get('s')} | code={res.get('code')}")
        if res.get("s") == "ok":
            return res.get("access_token")
        logger.error(f"Token refresh failed: {res}")
    except Exception as e:
        logger.error(f"refresh_access_token error: {e}")
    return None


def get_access_token(force: bool = False) -> str | None:
    """
    Return a valid access token.
    Uses in-memory cache (refreshed once per calendar day).
    """
    today = date.today().isoformat()

    if not force and _cache["access_token"] and _cache["date"] == today:
        return _cache["access_token"]

    refresh_token, stored_access = read_tokens_from_db()

    if not refresh_token:
        logger.error("No refresh_token in DB — run Colab auth first.")
        return stored_access  # fallback to whatever is stored

    new_access = refresh_access_token(refresh_token)

    if new_access:
        save_tokens_to_db(access_token=new_access)
        _cache["access_token"] = new_access
        _cache["date"]         = today
        logger.info("Access token refreshed and cached.")
        return new_access

    # Fallback: use stored token
    if stored_access:
        logger.warning("Refresh failed — using stored access token as fallback.")
        _cache["access_token"] = stored_access
        _cache["date"]         = today
    return stored_access
