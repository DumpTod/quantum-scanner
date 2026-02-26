# Quantum Scanner v2.0

A Flask-based NSE stock scanner using 5 quantitative models for swing trading signals.

## Architecture

```
app.py              — Flask endpoints
scanner.py          — Scan orchestration (background thread)
models.py           — 5 quantitative models (MRB, MDD, VRC, OFI, MTC)
indicators.py       — All custom indicators (no pandas_ta)
data_fetcher.py     — Fyers SDK OHLCV fetch with in-memory cache
backtest_engine.py  — Walk-forward backtester + Supabase cache
scorer.py           — Score & grade signals
risk_manager.py     — SL/Target/RR calculation
token_manager.py    — Daily token refresh from Supabase
config.py           — All credentials & constants
symbols.py          — ~300 ALL_STOCKS, ~150 FNO_STOCKS
colab_auth.ipynb    — Run every 15 days in Google Colab
```

## Setup

### 1. Fill in your Fyers PIN
In `config.py` set `FYERS_PIN = "your_4_digit_pin"`
Or set it as an environment variable on Render: `FYERS_PIN`

### 2. Run Colab auth (every 15 days)
Open `colab_auth.ipynb` in Google Colab.
Follow the prompts to authorize Fyers and save tokens to Supabase.

### 3. Deploy to Render
- Push this repo to GitHub
- Create new Web Service on Render pointing to your repo
- Set env var `FYERS_PIN` in Render dashboard
- Render auto-deploys on push

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Scanner info + endpoint list |
| `GET /api/health` | Health check |
| `GET /api/token` | Force token refresh |
| `GET /api/debug` | Test token + data + Supabase |
| `GET /api/scan` | Start background scan |
| `GET /api/scan?max=20` | Scan first 20 stocks |
| `GET /api/results` | Poll scan results |
| `GET /api/backtest/RELIANCE/BUY` | Backtest trade history |

## Signal Grades

| Grade | Score | Min Trades | Min WR | Expectancy |
|---|---|---|---|---|
| A+ HIGH | ≥85 | 3 | 60% | >0 |
| A+ MEDIUM | ≥80 | 3 | 50% | >0 |

## Models

1. **MRB** — Mean Reversion: Z-Score, Bollinger Bands, Keltner, RSI
2. **MDD** — Momentum Decay: MACD histogram acceleration, RSI momentum
3. **VRC** — Volatility Regime: ATR percentile determines regime + conditions
4. **OFI** — Order Flow: Volume delta, OBV, VWAP
5. **MTC** — Multi-Timeframe: EMA alignment across daily/weekly/monthly
