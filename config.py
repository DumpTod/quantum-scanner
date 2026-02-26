# config.py — Quantum Scanner Configuration

FYERS_CLIENT_ID  = "VS55VDHYCW-100"
FYERS_SECRET_KEY = "724FOKKSFS"
FYERS_PIN        = "2504"          # Fill this in before running
FYERS_REDIRECT   = "https://trade.fyers.in/api-login/redirect-uri/index.html"

SUPABASE_URL     = "https://ntxkqmjnmaowvwduswea.supabase.co"
SUPABASE_API_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im50eGtxbWpubWFvd3Z3ZHVzd2VhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE5ODg0OTMsImV4cCI6MjA4NzU2NDQ5M30."
    "7NV0yDkMHRVpiYpoUXbcz3LIm9t__ocKlDGJV0HRIVE"
)

# Backtest settings
BACKTEST_STEP           = 5       # bars between each entry check
BACKTEST_MAX_BARS_HELD  = 30      # forced exit after N bars
BACKTEST_MAX_TRADES     = 15      # max trades stored per symbol
BACKTEST_MIN_CANDLES    = 60      # skip symbol if fewer bars
DATA_PERIOD_DAYS        = 540     # ~18 months of daily data

# Risk management
ATR_MULT_SL             = 1.5
MIN_SL_PCT              = 1.5
MAX_SL_PCT              = 5.0
MIN_RR                  = 1.2
RR_LOW_VOL              = 2.5
RR_NORMAL_VOL           = 2.0
RR_HIGH_VOL             = 1.5

# Grade thresholds
GRADE_A_PLUS_HIGH_SCORE     = 85
GRADE_A_PLUS_MEDIUM_SCORE   = 80
GRADE_MIN_TRADES            = 3
GRADE_HIGH_WR               = 60.0
GRADE_MED_WR                = 50.0
GRADE_MIN_EXPECTANCY        = 0.0

# Symbol corrections for Fyers
SYMBOL_MAP = {
    "M&M":    "M&M",
    "M&MFIN": "M&MFIN",
    "L&TFH":  "LTF",
    "ARE&M":  "ARE&M",
}

def to_fyers_symbol(symbol: str) -> str:
    mapped = SYMBOL_MAP.get(symbol, symbol)
    return f"NSE:{mapped}-EQ"
