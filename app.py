import pandas as pd
import numpy as np
import warnings
import requests
import hashlib
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
warnings.filterwarnings('ignore')

# ============================================================
# FYERS CONFIG
# ============================================================

FYERS_CLIENT_ID  = "VS55VDHYCW-100"
FYERS_SECRET_KEY = "724FOKKSFS"
FYERS_PIN        = "2504"   # ← Fill this in

FYERS_APP_HASH = hashlib.sha256(
    f"{FYERS_CLIENT_ID}:{FYERS_SECRET_KEY}".encode()
).hexdigest()

# ============================================================
# SUPABASE CONFIG
# ============================================================

SUPABASE_URL     = "https://ntxkqmjnmaowvwduswea.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im50eGtxbWpubWFvd3Z3ZHVzd2VhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE5ODg0OTMsImV4cCI6MjA4NzU2NDQ5M30.7NV0yDkMHRVpiYpoUXbcz3LIm9t__ocKlDGJV0HRIVE"

SUPA_HEADERS = {
    "apikey":        SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal"
}

# ============================================================
# SYMBOL FORMAT MAP — special cases for Fyers
# ============================================================

SYMBOL_MAP = {
    "M&M":    "MM",
    "M&MFIN": "MMFIN",
    "L&TFH":  "LTFH",
    "ARE&M":  "AREM",
}

def to_fyers_symbol(symbol):
    mapped = SYMBOL_MAP.get(symbol, symbol)
    return f"NSE:{mapped}-EQ"

# ============================================================
# TOKEN MANAGER
# ============================================================

_access_token = None
_token_date   = None

def get_supabase_tokens():
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/fyers_tokens",
            headers=SUPA_HEADERS,
            params={"id": "eq.1", "select": "*"},
            timeout=10
        )
        if r.status_code == 200 and r.json():
            return r.json()[0]
    except Exception as e:
        print(f"[Supabase] get_tokens error: {e}")
    return None

def save_access_token(access_token):
    try:
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/fyers_tokens",
            headers=SUPA_HEADERS,
            params={"id": "eq.1"},
            json={"access_token": access_token,
                  "updated_at": datetime.utcnow().isoformat()},
            timeout=10
        )
    except Exception as e:
        print(f"[Supabase] save_access_token error: {e}")

def refresh_access_token(refresh_token):
    try:
        r = requests.post(
            "https://api-t1.fyers.in/api/v3/validate-refresh-token",
            headers={"Content-Type": "application/json"},
            json={"grant_type":    "refresh_token",
                  "appIdHash":     FYERS_APP_HASH,
                  "refresh_token": refresh_token,
                  "pin":           FYERS_PIN},
            timeout=15
        )
        if r.status_code == 200:
            token = r.json().get("access_token")
            if token:
                print(f"[Fyers] Token refreshed at {datetime.now().strftime('%H:%M:%S')}")
                return token
            print(f"[Fyers] Refresh response: {r.json()}")
    except Exception as e:
        print(f"[Fyers] refresh error: {e}")
    return None

def get_token():
    global _access_token, _token_date
    today = datetime.now().date()
    if _access_token and _token_date == today:
        return _access_token
    tokens = get_supabase_tokens()
    if not tokens:
        raise Exception("No tokens in Supabase. Run Colab notebook first.")
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise Exception("refresh_token missing. Run Colab notebook.")
    new_token = refresh_access_token(refresh_token)
    if new_token:
        _access_token = new_token
        _token_date   = today
        save_access_token(new_token)
        return _access_token
    stored = tokens.get("access_token")
    if stored:
        print("[Fyers] Using stored access token.")
        _access_token = stored
        _token_date   = today
        return _access_token
    raise Exception("All token methods failed. Run Colab notebook again.")

def fyers_headers():
    return {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type":  "application/json"
    }

# ============================================================
# DATE HELPERS — DD/MM/YYYY format
# ============================================================

def fmt_date(dt_str):
    """Convert any date string to DD/MM/YYYY for storage."""
    try:
        dt = pd.to_datetime(dt_str)
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return dt_str

# ============================================================
# SUPABASE — Backtest helpers
# ============================================================

def supa_get_cache(symbol, direction):
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/backtest_cache",
            headers=SUPA_HEADERS,
            params={"symbol":    f"eq.{symbol}",
                    "direction": f"eq.{direction}",
                    "select":    "*"},
            timeout=10
        )
        if r.status_code == 200 and r.json():
            return r.json()[0]
    except Exception as e:
        print(f"[Supabase] get_cache error: {e}")
    return None

def supa_save_cache(symbol, direction, bt):
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/backtest_cache",
            headers={**SUPA_HEADERS, "Prefer": "resolution=merge-duplicates"},
            json={"symbol": symbol, "direction": direction,
                  "win_rate": bt["win_rate"], "total_trades": bt["total_trades"],
                  "wins": bt["wins"], "losses": bt["losses"],
                  "avg_profit_pct": bt["avg_profit_pct"],
                  "avg_loss_pct":   bt["avg_loss_pct"],
                  "expectancy":     bt["expectancy"],
                  "updated_at":     datetime.utcnow().isoformat()},
            timeout=10
        )
    except Exception as e:
        print(f"[Supabase] save_cache error: {e}")

def supa_save_trades(symbol, direction, trades):
    if not trades:
        return
    try:
        requests.delete(
            f"{SUPABASE_URL}/rest/v1/backtest_trades",
            headers=SUPA_HEADERS,
            params={"symbol": f"eq.{symbol}", "direction": f"eq.{direction}"},
            timeout=10
        )
        requests.post(
            f"{SUPABASE_URL}/rest/v1/backtest_trades",
            headers=SUPA_HEADERS,
            json=[{"symbol":      symbol,
                   "direction":   direction,
                   "entry_price": t["entry_price"],
                   "stop_loss":   t["stop_loss"],
                   "target":      t["target"],
                   "exit_price":  t["exit_price"],
                   "outcome":     t["outcome"],
                   "profit_pct":  t["profit_pct"],
                   "entry_bar":   t["entry_bar"],
                   "exit_bar":    t["exit_bar"],
                   "bars_held":   t["bars_held"],
                   "entry_date":  t["entry_date"],
                   "exit_date":   t["exit_date"]} for t in trades],
            timeout=15
        )
    except Exception as e:
        print(f"[Supabase] save_trades error: {e}")

# ============================================================
# FYERS DATA FETCH
# ============================================================

def fetch_stock_data(symbol, period="18mo"):
    try:
        days      = {"18mo": 540, "1y": 365, "6mo": 180}.get(period, 540)
        date_to   = datetime.now()
        date_from = date_to - timedelta(days=days)
        fyers_sym = to_fyers_symbol(symbol)
        r = requests.get(
            ""https://api-t1.fyers.in/data/v3/history"",
            headers=fyers_headers(),
            params={"symbol":      fyers_sym,
                    "resolution":  "D",
                    "date_format": "1",
                    "range_from":  date_from.strftime("%Y-%m-%d"),
                    "range_to":    date_to.strftime("%Y-%m-%d"),
                    "cont_flag":   "1"},
            timeout=15
        )
        if r.status_code != 200:
            print(f"[Fyers] HTTP {r.status_code} for {fyers_sym}")
            return None
        data = r.json()
        if data.get("s") != "ok" or not data.get("candles"):
            print(f"[Fyers] Bad response for {fyers_sym}: {data.get('message','')}")
            return None
        df = pd.DataFrame(data["candles"],
                          columns=["timestamp","Open","High","Low","Close","Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df if len(df) >= 60 else None
    except Exception as e:
        print(f"[Fyers] fetch error {symbol}: {e}")
        return None

# ============================================================
# HELPERS
# ============================================================

def percentileofscore(arr, score):
    arr = np.array(arr)
    return (np.sum(arr <= score) / len(arr)) * 100

# ============================================================
# FNO + ALL STOCKS
# ============================================================

FNO_STOCKS = {
    "AARTIIND","ABB","ABBOTINDIA","ABCAPITAL","ABFRL","ACC","ADANIENT","ADANIENSOL",
    "ADANIGREEN","ADANIPORTS","ADANIPOWER","ALKEM","AMBUJACEM","ANGELONE","APLAPOLLO",
    "APOLLOHOSP","APOLLOTYRE","ASHOKLEY","ASIANPAINT","ASTRAL","ATGL","ATUL","AUBANK",
    "AUROPHARMA","AWL","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BALKRISIND",
    "BALRAMCHIN","BANDHANBNK","BANKBARODA","BATAINDIA","BDL","BEL","BERGEPAINT",
    "BHARATFORG","BHARTIARTL","BHEL","BIOCON","BOSCHLTD","BPCL","BRIGADE","BRITANNIA",
    "BSE","BSOFT","CANBK","CANFINHOME","CDSL","CHAMBLFERT","CHOLAFIN","CIPLA","CLEAN",
    "COALINDIA","COCHINSHIP","COFORGE","COLPAL","CONCOR","COROMANDEL","CREDITACC",
    "CRISIL","CROMPTON","CUB","CUMMINSIND","CYIENT","DABUR","DALBHARAT","DEEPAKNTR",
    "DELHIVERY","DELTACORP","DEVYANI","DIVISLAB","DIXON","DLF","DRREDDY","EICHERMOT",
    "EMCURE","ESCORTS","ETERNAL","EXIDEIND","FACT","FEDERALBNK","GAIL","GLENMARK",
    "GMRAIRPORT","GNFC","GODREJCP","GODREJPROP","GRANULES","GRASIM","GRSE","GSPL",
    "GUJGASLTD","HAL","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE","HEROMOTOCO",
    "HINDALCO","HINDCOPPER","HINDPETRO","HINDUNILVR","HONAUT","HUDCO","ICICIBANK",
    "ICICIGI","ICICIPRULI","IDEA","IDFCFIRSTB","IEX","IGL","INDHOTEL","INDIACEM",
    "INDIAMART","INDIANB","INDIGO","INDUSINDBK","INDUSTOWER","INFY","IOC","IPCALAB",
    "IRCTC","IREDA","IRFC","ITC","ITCHOTELS","JINDALSTEL","JIOFIN","JKCEMENT",
    "JSWENERGY","JSWINFRA","JSWSTEEL","JUBLFOOD","KALYANKJIL","KEI","KFINTECH",
    "KOTAKBANK","KPITTECH","LALPATHLAB","LAURUSLABS","LICHSGFIN","LICI","LODHA",
    "LT","LTIM","LTTS","LUPIN","M&M","M&MFIN","MANAPPURAM","MANKIND","MAPMYINDIA",
    "MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX","METROPOLIS","MFSL","MGL","MOTHERSON",
    "MPHASIS","MRF","MUTHOOTFIN","NATIONALUM","NAUKRI","NAVINFLUOR","NESTLEIND","NHPC",
    "NMDC","NTPC","OBEROIRLTY","OFSS","ONGC","PAGEIND","PATANJALI","PAYTM","PERSISTENT",
    "PETRONET","PFC","PIDILITIND","PIIND","PNB","POLICYBZR","POLYCAB","POONAWALLA",
    "POWERGRID","PRESTIGE","PVRINOX","RAMCOCEM","RBLBANK","RECLTD","RELIANCE","RVNL",
    "SAIL","SBICARD","SBILIFE","SBIN","SHREECEM","SHRIRAMFIN","SIEMENS","SJVN",
    "SONACOMS","SRF","SUNPHARMA","SUNTV","SUPREMEIND","SUZLON","SYNGENE","SYRMA",
    "TATACHEM","TATACOMM","TATACONSUM","TATAELXSI","TATAMOTORS","TATAPOWER","TATASTEEL",
    "TATATECH","TCS","TECHM","TIINDIA","TITAN","TORNTPHARM","TORNTPOWER","TRENT",
    "TVSMOTOR","UBL","ULTRACEMCO","UNIONBANK","UNITDSPR","UPL","VBL","VEDL","VOLTAS",
    "WIPRO","YESBANK","ZEEL","ZOMATO","ZYDUSLIFE"
}

ALL_STOCKS = list(set([
    "360ONE","3MINDIA","5PAISA","AARTIDRUGS","AARTIIND","AAVAS","ABB","ABBOTINDIA",
    "ABCAPITAL","ABFRL","ACC","ADANIENT","ADANIENSOL","ADANIGREEN","ADANIPORTS",
    "ADANIPOWER","ADVENZYMES","AFFLE","AIAENG","AJANTPHARM","AKZOINDIA","ALEMBICLTD",
    "ALKEM","ALKYLAMINE","ALLCARGO","ALOKINDS","AMBER","AMBUJACEM","ANANDRATHI",
    "ANGELONE","APLAPOLLO","APOLLOHOSP","APOLLOTYRE","APTUS","ASHOKLEY","ASIANPAINT",
    "ASTERDM","ASTRAL","ATGL","ATUL","AUBANK","AUROPHARMA","AVANTIFEED","AWL","AXISBANK",
    "BAJAJ-AUTO","BAJAJCON","BAJAJFINSV","BAJAJHLDNG","BAJFINANCE","BALAMINES",
    "BALKRISIND","BALMLAWRIE","BALRAMCHIN","BANDHANBNK","BANKBARODA","BANKINDIA","BASF",
    "BATAINDIA","BAYERCROP","BDL","BEL","BEML","BERGEPAINT","BHARATFORG","BHARTIARTL",
    "BHEL","BIKAJI","BIOCON","BIRLACORPN","BLUESTARCO","BOSCHLTD","BPCL","BRIGADE",
    "BRITANNIA","BSE","BSOFT","CAMPUS","CAMS","CANBK","CANFINHOME","CAPLIPOINT",
    "CARBORUNIV","CASTROLIND","CDSL","CEATLTD","CENTRALBK","CENTURYPLY","CERA","CGCL",
    "CGPOWER","CHALET","CHAMBLFERT","CHEMPLASTS","CHOLAFIN","CIPLA","CLEAN","COALINDIA",
    "COCHINSHIP","COFORGE","COLPAL","CONCOR","CONCORDBIO","COROMANDEL","CRAFTSMAN",
    "CREDITACC","CRISIL","CROMPTON","CSBBANK","CUB","CUMMINSIND","CYIENT","DABUR",
    "DALBHARAT","DATAPATTNS","DCMSHRIRAM","DEEPAKFERT","DEEPAKNTR","DELHIVERY",
    "DELTACORP","DEVYANI","DIVISLAB","DIXON","DLF","DMART","DRREDDY","EASEMYTRIP",
    "ECLERX","EIDPARRY","EICHERMOT","ELECON","ELGIEQUIP","EMAMILTD","EMCURE","ENDURANCE",
    "ENGINERSIN","EPL","EQUITASBNK","ERIS","ESCORTS","ETERNAL","EXIDEIND","FACT",
    "FEDERALBNK","FINCABLES","FINEORG","FINPIPE","FLUOROCHEM","FORTIS","FSL","GAIL",
    "GALAXYSURF","GARFIBRES","GESHIP","GICRE","GILLETTE","GLAND","GLAXO","GLENMARK",
    "GMRAIRPORT","GNFC","GODREJCP","GODREJIND","GODREJPROP","GPPL","GRANULES","GRAPHITE",
    "GRASIM","GRINDWELL","GRINFRA","GRSE","GSPL","GUJALKALI","GUJGASLTD","HAL",
    "HAPPSTMNDS","HATSUN","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE","HEMIPROP",
    "HEROMOTOCO","HFCL","HIKAL","HINDALCO","HINDCOPPER","HINDPETRO","HINDUNILVR",
    "HINDWAREAP","HOMEFIRST","HONAUT","HSCL","HUDCO","IBREALEST","ICICIBANK","ICICIGI",
    "ICICIPRULI","IDEA","IDFCFIRSTB","IEX","IGL","IIFL","INDHOTEL","INDIACEM","INDIAMART",
    "INDIANB","INDIGO","INDUSINDBK","INDUSTOWER","INFY","INGERRAND","INTELLECT","IOC",
    "IPCALAB","IRCTC","IREDA","IRFC","ITC","ITCHOTELS","ITI","JBCHEPHARM","JBMA",
    "JINDALSTEL","JIOFIN","JKCEMENT","JKLAKSHMI","JKPAPER","JMFINANCIL","JSL","JSWENERGY",
    "JSWINFRA","JSWSTEEL","JTEKTINDIA","JUBLFOOD","JUSTDIAL","JYOTHYLAB","KAJARIACER",
    "KALPATARU","KALYANKJIL","KANSAINER","KAYNES","KEC","KEI","KFINTECH","KIRLOSENG",
    "KNRCON","KPITTECH","KSB","KOTAKBANK","KRBL","KSCL","LAOPALA","LALPATHLAB",
    "LATENTVIEW","LAURUSLABS","LEMONTREE","LICHSGFIN","LICI","LINDEINDIA","LLOYDSME",
    "LODHA","LT","LTIM","LTTS","LUPIN","LUXIND","LXCHEM","M&M","M&MFIN","MAHABANK",
    "MAHLIFE","MAHLOG","MANAPPURAM","MANKIND","MAPMYINDIA","MARICO","MARKSANS","MARUTI",
    "MASTEK","MAXHEALTH","MAZDOCK","MCX","METROBRAND","METROPOLIS","MFSL","MGL","MIDHANI",
    "MINDACORP","MOTHERSON","MPHASIS","MRF","MSUMI","MTARTECH","MUTHOOTFIN","NATCOPHARM",
    "NATIONALUM","NAUKRI","NAVINFLUOR","NAVNETEDUL","NCC","NESTLEIND","NETWORK18","NEWGEN",
    "NHPC","NIACL","NLCINDIA","NMDC","NOCIL","NTPC","NUVOCO","OBEROIRLTY","OFSS","OIL",
    "OLECTRA","ONGC","ORIENTELEC","PAGEIND","PAISALO","PATANJALI","PAYTM","PCBL",
    "PERSISTENT","PETRONET","PFC","PFIZER","PGHH","PIDILITIND","PIIND","PNB","PNBHOUSING",
    "POLICYBZR","POLYCAB","POONAWALLA","POWERGRID","POWERINDIA","PPLPHARMA","PRESTIGE",
    "PRINCEPIPE","PRSMJOHNSN","PVRINOX","QUESS","RADICO","RAIN","RAILTEL","RAJESHEXPO",
    "RALLIS","RAMCOCEM","RATNAMANI","RAYMOND","RBA","RBLBANK","RCF","RECLTD","REDINGTON",
    "RELAXO","RELIANCE","RELIGARE","RITES","RKFORGE","ROUTE","RVNL","SAREGAMA","SAIL",
    "SBICARD","SBILIFE","SBIN","SCHAEFFLER","SCHNEIDER","SHILPAMED","SHREECEM","SHRIRAMFIN",
    "SIEMENS","SJVN","SKFINDIA","SOBHA","SOLARINDS","SONACOMS","SONATSOFTW","SOUTHBANK",
    "SPARC","SRF","STAR","STARHEALTH","STEL","STLTECH","SUDARSCHEM","SUMICHEM","SUNDARMFIN",
    "SUNDRMFAST","SUNPHARMA","SUNTV","SUPREMEIND","SUVEN","SUZLON","SWSOLAR","SYNGENE",
    "SYRMA","TANLA","TARSONS","TATACHEM","TATACOMM","TATACONSUM","TATAELXSI","TATAINVEST",
    "TATAPOWER","TATASTEEL","TATATECH","TCS","TECHM","THERMAX","TIINDIA","TIMKEN","TITAN",
    "TORNTPHARM","TORNTPOWER","TRENT","TRIDENT","TRITURBINE","TRIVENI","TTML","TVSMOTOR",
    "UBL","UJJIVANSFB","ULTRACEMCO","UNIONBANK","UNITDSPR","UPL","UTIAMC","VAKRANGEE",
    "VARROC","VBL","VEDL","VENKEYS","VGUARD","VINATIORGA","VIPIND","VOLTAS","VSTIND",
    "WELCORP","WELSPUNLIV","WHIRLPOOL","WIPRO","YESBANK","ZEEL","ZENSARTECH","ZFCVINDIA",
    "ZYDUSLIFE","ZYDUSWELL"
]))

# ============================================================
# CUSTOM INDICATORS
# ============================================================

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def sma(series, length):
    return series.rolling(window=length).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=length).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=length).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ml = ema(series, fast) - ema(series, slow)
    sl = ema(ml, signal)
    return ml, sl, ml - sl

def atr(high, low, close, length=14):
    pc = close.shift(1)
    tr = pd.concat([high-low, abs(high-pc), abs(low-pc)], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()

def bbands(series, length=20, std=2):
    mid = sma(series, length)
    s   = series.rolling(window=length).std()
    return mid+std*s, mid, mid-std*s

def obv(close, volume):
    d = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    return (volume * d).cumsum()

def vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()

# ============================================================
# INDICATORS
# ============================================================

def calculate_all_indicators(df):
    df['EMA9']  = ema(df['Close'], 9)
    df['EMA21'] = ema(df['Close'], 21)
    df['EMA50'] = ema(df['Close'], 50)
    df['SMA20'] = sma(df['Close'], 20)
    df['SMA50'] = sma(df['Close'], 50)
    df['SMA200']= sma(df['Close'], 200)
    df['BB_upper'], df['BB_mid'], df['BB_lower'] = bbands(df['Close'], 20, 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
    kc_atr = atr(df['High'], df['Low'], df['Close'], 20)
    df['KC_upper'] = df['EMA21'] + 2 * kc_atr
    df['KC_lower'] = df['EMA21'] - 2 * kc_atr
    df['ZScore']     = (df['Close'] - df['SMA20']) / df['Close'].rolling(20).std()
    df['RSI']        = rsi(df['Close'], 14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df['Close'], 12, 26, 9)
    df['RSI_ROC']    = df['RSI'].diff(3)
    df['MACD_accel'] = df['MACD_hist'].diff(1)
    df['ATR']            = atr(df['High'], df['Low'], df['Close'], 14)
    df['ATR_pct']        = (df['ATR'] / df['Close']) * 100
    df['ATR_percentile'] = df['ATR'].rolling(100).apply(
        lambda x: percentileofscore(x, x.iloc[-1]), raw=False)
    df['Returns']         = df['Close'].pct_change()
    df['HV20']            = df['Returns'].rolling(20).std() * np.sqrt(252) * 100
    df['Vol_SMA20']       = sma(df['Volume'], 20)
    df['VWAP']            = vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Up_Vol']          = np.where(df['Close'] > df['Open'], df['Volume'], 0)
    df['Down_Vol']        = np.where(df['Close'] < df['Open'], df['Volume'], 0)
    df['Vol_Delta']       = pd.Series(df['Up_Vol']).rolling(5).sum() - pd.Series(df['Down_Vol']).rolling(5).sum()
    df['Vol_Delta_Ratio'] = pd.Series(df['Up_Vol']).rolling(5).sum() / (pd.Series(df['Down_Vol']).rolling(5).sum() + 1)
    df['OBV']             = pd.Series(obv(df['Close'], df['Volume']), index=df.index)
    df['OBV_SMA']         = sma(df['OBV'], 20)
    df['Weekly_EMA9']  = ema(df['Close'], 45)
    df['Weekly_EMA21'] = ema(df['Close'], 105)
    df['Monthly_EMA']  = ema(df['Close'], 200)
    df['High_52w'] = df['High'].rolling(window=min(252,len(df)), min_periods=50).max()
    df['Low_52w']  = df['Low'].rolling(window=min(252,len(df)),  min_periods=50).min()
    rh = df['High'].iloc[-60:].max(); rl = df['Low'].iloc[-60:].min(); fr = rh - rl
    df['Fib_236'] = rh - fr*0.236
    df['Fib_382'] = rh - fr*0.382
    df['Fib_500'] = rh - fr*0.500
    df['Fib_618'] = rh - fr*0.618
    return df.dropna()

# ============================================================
# 5 MODELS
# ============================================================

def model_1_mean_reversion(df):
    l=df.iloc[-1]; p=df.iloc[-2]
    buy  = [l['ZScore']<-1.0, l['Close']<=l['BB_lower']*1.02, l['Close']<l['KC_lower']*1.03, l['RSI']<40, l['Close']>p['Close']]
    sell = [l['ZScore']>1.0,  l['Close']>=l['BB_upper']*0.98, l['Close']>l['KC_upper']*0.97, l['RSI']>60, l['Close']<p['Close']]
    bs,ss=sum(buy),sum(sell)
    if bs>=3: return bs,5,"BUY","Mean reversion from oversold"
    if ss>=3: return ss,5,"SELL","Mean reversion from overbought"
    return max(bs,ss),5,"NEUTRAL",""

def model_2_momentum_decay(df):
    l=df.iloc[-1]; p=df.iloc[-2]
    buy  = [l['MACD_hist']>p['MACD_hist'], l['MACD_accel']>0, l['RSI']>p['RSI'], 40<l['RSI']<65, l['Close']>p['Close']]
    sell = [l['MACD_hist']<p['MACD_hist'], l['MACD_accel']<0, l['RSI']<p['RSI'], 35<l['RSI']<60, l['Close']<p['Close']]
    bs,ss=sum(buy),sum(sell)
    if bs>=3: return bs,5,"BUY","Momentum accelerating up"
    if ss>=3: return ss,5,"SELL","Momentum decelerating down"
    return max(bs,ss),5,"NEUTRAL",""

def model_3_volatility_regime(df):
    l=df.iloc[-1]; p=df.iloc[-2]
    ap=l['ATR_percentile']
    if ap<30: regime="LOW_VOL"
    elif ap<70: regime="NORMAL_VOL"
    else: regime="HIGH_VOL"
    if regime=="LOW_VOL":
        buy  = [l['BB_width']<df['BB_width'].quantile(0.3), l['Close']>l['BB_mid'],  l['Volume']>l['Vol_SMA20']*1.1, l['MACD_hist']>0, l['Close']>l['EMA21']]
        sell = [l['BB_width']<df['BB_width'].quantile(0.3), l['Close']<l['BB_mid'],  l['Volume']>l['Vol_SMA20']*1.1, l['MACD_hist']<0, l['Close']<l['EMA21']]
    elif regime=="NORMAL_VOL":
        buy  = [l['Close']>l['EMA21'], l['EMA21']>l['EMA50'], 45<l['RSI']<75, l['MACD']>l['MACD_signal'], l['Volume']>l['Vol_SMA20']*0.8]
        sell = [l['Close']<l['EMA21'], l['EMA21']<l['EMA50'], 25<l['RSI']<55, l['MACD']<l['MACD_signal'], l['Volume']>l['Vol_SMA20']*0.8]
    else:
        buy  = [l['ZScore']<-1.0, l['RSI']<35, l['Close']<l['BB_lower']*1.02, l['Close']>p['Close'], l['ATR_pct']>1.5]
        sell = [l['ZScore']>1.0,  l['RSI']>65, l['Close']>l['BB_upper']*0.98, l['Close']<p['Close'], l['ATR_pct']>1.5]
    bs,ss=sum(buy),sum(sell)
    if bs>=3: return bs,5,"BUY",f"Regime: {regime}"
    if ss>=3: return ss,5,"SELL",f"Regime: {regime}"
    return max(bs,ss),5,"NEUTRAL",regime

def model_4_order_flow(df):
    l=df.iloc[-1]
    buy  = [l['Vol_Delta_Ratio']>1.3, l['OBV']>l['OBV_SMA'], l['Close']>l['VWAP'], l['Volume']>l['Vol_SMA20']*0.9, l['Close']>l['Open']]
    sell = [l['Vol_Delta_Ratio']<0.75,l['OBV']<l['OBV_SMA'], l['Close']<l['VWAP'], l['Volume']>l['Vol_SMA20']*0.9, l['Close']<l['Open']]
    bs,ss=sum(buy),sum(sell)
    if bs>=3: return bs,5,"BUY","Accumulation detected"
    if ss>=3: return ss,5,"SELL","Distribution detected"
    return max(bs,ss),5,"NEUTRAL",""

def model_5_multi_timeframe(df):
    l=df.iloc[-1]
    nfb=any(abs(l['Close']-l[f])/l['Close']<0.03 for f in ['Fib_382','Fib_500','Fib_618'])
    nfs=any(abs(l['Close']-l[f])/l['Close']<0.03 for f in ['Fib_236','Fib_382','Fib_500'])
    buy  = [l['EMA9']>l['EMA21'], l['Weekly_EMA9']>l['Weekly_EMA21'], l['Close']>l['Monthly_EMA'], nfb, 35<l['RSI']<70]
    sell = [l['EMA9']<l['EMA21'], l['Weekly_EMA9']<l['Weekly_EMA21'], l['Close']<l['Monthly_EMA'], nfs, 30<l['RSI']<65]
    bs,ss=sum(buy),sum(sell)
    if bs>=3: return bs,5,"BUY","Multi-timeframe bullish"
    if ss>=3: return ss,5,"SELL","Multi-timeframe bearish"
    return max(bs,ss),5,"NEUTRAL",""

# ============================================================
# RISK
# ============================================================

def calculate_risk(df, direction):
    l=df.iloc[-1]; ep=l['Close']; a=l['ATR']
    sl_pct=max(1.5,min(5.0,(a*1.5/ep)*100)); sl_dist=ep*(sl_pct/100)
    vol_pct=(a/ep)*100
    rh=df['High'].iloc[-20:].max(); rl=df['Low'].iloc[-20:].min(); sw=((rh-rl)/ep)*100
    if   vol_pct>3 and sw>15: rr=3.0
    elif vol_pct>2 and sw>10: rr=2.5
    elif vol_pct>1.5 or sw>8: rr=2.0
    else:                      rr=1.5
    td=sl_dist*rr
    if direction=="BUY":  sl=round(ep-sl_dist,2); target=max(round(ep+td,2),round(ep*1.02,2))
    else:                 sl=round(ep+sl_dist,2); target=min(round(ep-td,2),round(ep*0.98,2))
    ar=abs(ep-sl); aw=abs(target-ep)
    return {'entry':round(ep,2),'stop_loss':sl,'target':target,
            'rr_ratio':round(aw/ar,1) if ar>0 else 0,'sl_pct':round(sl_pct,2),'atr':round(a,2)}

# ============================================================
# BACKTEST
# ============================================================

def backtest_quantum(df, direction):
    wins=0; losses=0; profits=[]; loss_amounts=[]; trades=[]
    in_trade=False; trade_exit_idx=0
    if len(df)<100:
        return {'win_rate':50,'total_trades':0,'wins':0,'losses':0,
                'avg_profit_pct':0,'avg_loss_pct':0,'expectancy':0}, []
    for i in range(60, len(df)-10, 5):
        if in_trade and i<=trade_exit_idx: continue
        in_trade=False
        curr=df.iloc[i]
        if direction=="BUY"  and curr['Close']<curr['EMA21'] and curr['RSI']<35: continue
        if direction=="SELL" and curr['Close']>curr['EMA21'] and curr['RSI']>65: continue
        sub_df=df.iloc[:i+1].copy()
        if len(sub_df)<60: continue
        try:
            _,_,d1,_=model_1_mean_reversion(sub_df)
            _,_,d2,_=model_2_momentum_decay(sub_df)
            _,_,d3,_=model_3_volatility_regime(sub_df)
            _,_,d4,_=model_4_order_flow(sub_df)
            _,_,d5,_=model_5_multi_timeframe(sub_df)
        except: continue
        dirs=[d1,d2,d3,d4,d5]; signal=None
        if direction=="BUY"  and dirs.count("BUY")>=3:  signal="BUY"
        if direction=="SELL" and dirs.count("SELL")>=3: signal="SELL"
        if signal is None: continue
        ep=df.iloc[i]['Close']; a=df.iloc[i]['ATR']
        sd=ep*(max(1.5,min(5.0,(a*1.5/ep)*100))/100); td=sd*2.0
        sl =ep-sd if signal=="BUY" else ep+sd
        tgt=ep+td if signal=="BUY" else ep-td
        entry_date = fmt_date(str(df.index[i])[:10])
        for j in range(1, min(30,len(df)-i-1)+1):
            f=df.iloc[i+j]
            exit_date = fmt_date(str(df.index[i+j])[:10])
            hit=None; exit_p=0.0
            if signal=="BUY":
                if f['Low'] <=sl:    hit="LOSS"; exit_p=sl
                elif f['High']>=tgt: hit="WIN";  exit_p=tgt
            else:
                if f['High']>=sl:    hit="LOSS"; exit_p=sl
                elif f['Low'] <=tgt: hit="WIN";  exit_p=tgt
            if hit:
                pnl=round(((exit_p-ep)/ep)*100 if signal=="BUY" else ((ep-exit_p)/ep)*100,3)
                trades.append({"entry_price":round(ep,2),"stop_loss":round(sl,2),
                                "target":round(tgt,2),"exit_price":round(exit_p,2),
                                "outcome":hit,"profit_pct":pnl,"entry_bar":i,"exit_bar":i+j,
                                "bars_held":j,"entry_date":entry_date,"exit_date":exit_date})
                if hit=="WIN": wins+=1; profits.append(abs(pnl))
                else:          losses+=1; loss_amounts.append(abs(pnl))
                trade_exit_idx=i+j; in_trade=True; break
        if wins+losses>=15: break
    total=wins+losses
    if total==0:
        return {'win_rate':50,'total_trades':0,'wins':0,'losses':0,
                'avg_profit_pct':0,'avg_loss_pct':0,'expectancy':0}, []
    wr=round((wins/total)*100,1)
    ap=round(np.mean(profits),2)      if profits      else 0
    al=round(np.mean(loss_amounts),2) if loss_amounts else 0
    return {'win_rate':wr,'total_trades':total,'wins':wins,'losses':losses,
            'avg_profit_pct':ap,'avg_loss_pct':al,
            'expectancy':round((wr/100*ap)-((100-wr)/100*al),2)}, trades

# ============================================================
# MASTER SCAN
# ============================================================

def quantum_scan(symbol):
    df=fetch_stock_data(symbol)
    if df is None: return None
    try: df=calculate_all_indicators(df)
    except: return None
    if df.empty or len(df)<80: return None
    try:
        m1p,m1t,m1d,m1r=model_1_mean_reversion(df)
        m2p,m2t,m2d,m2r=model_2_momentum_decay(df)
        m3p,m3t,m3d,m3r=model_3_volatility_regime(df)
        m4p,m4t,m4d,m4r=model_4_order_flow(df)
        m5p,m5t,m5d,m5r=model_5_multi_timeframe(df)
    except: return None
    dirs=[m1d,m2d,m3d,m4d,m5d]; bc=dirs.count("BUY"); sc=dirs.count("SELL")
    if   bc>=3: direction="BUY";  agreement=bc
    elif sc>=3: direction="SELL"; agreement=sc
    else: return None
    is_fno=symbol in FNO_STOCKS
    if direction=="SELL" and not is_fno: return None
    risk=calculate_risk(df,direction)
    if direction=="BUY"  and risk['target']<=risk['entry']: return None
    if direction=="SELL" and risk['target']>=risk['entry']: return None
    if risk['rr_ratio']<1.2: return None
    cached=supa_get_cache(symbol,direction)
    if cached:
        bt={'win_rate':cached['win_rate'],'total_trades':cached['total_trades'],
            'wins':cached['wins'],'losses':cached['losses'],
            'avg_profit_pct':cached['avg_profit_pct'],'avg_loss_pct':cached['avg_loss_pct'],
            'expectancy':cached['expectancy']}
    else:
        bt,trades=backtest_quantum(df,direction)
        supa_save_cache(symbol,direction,bt)
        supa_save_trades(symbol,direction,trades)
    if bt['total_trades']>=3 and bt['win_rate']<40:  return None
    if bt['total_trades']>=3 and bt['expectancy']<0: return None
    tp=m1p+m2p+m3p+m4p+m5p; tt=m1t+m2t+m3t+m4t+m5t
    mp=(tp/tt)*100; ab=(agreement-3)*5; bb=0
    if bt['total_trades']>=3:
        if   bt['win_rate']>=65 and bt['expectancy']>1.5: bb=10
        elif bt['win_rate']>=55 and bt['expectancy']>0:   bb=6
        elif bt['win_rate']>=45:                           bb=3
    ts=round(min(mp+ab+bb,100),1)
    if   ts>=85 and bt['total_trades']>=3 and bt['win_rate']>=60 and bt['expectancy']>0: grade="A+"; conf="HIGH"
    elif ts>=80 and bt['total_trades']>=3 and bt['win_rate']>=50 and bt['expectancy']>0: grade="A+"; conf="MEDIUM"
    else: return None
    am=[n for n,d in [("MRB",m1d),("MDD",m2d),("VRC",m3d),("OFI",m4d),("MTC",m5d)] if d==direction]
    return {
        'symbol':symbol,'direction':direction,
        'entry':risk['entry'],'stop_loss':risk['stop_loss'],'target':risk['target'],
        'score':ts,'grade':grade,'rr_ratio':risk['rr_ratio'],
        'is_fno':is_fno,'confidence':conf,
        'models_agreeing':f"{agreement}/5",'agreeing_models':", ".join(am),
        'historical_win_rate':bt['win_rate'],'total_past_trades':bt['total_trades'],
        'past_wins':bt['wins'],'past_losses':bt['losses'],
        'avg_profit_pct':bt['avg_profit_pct'],'avg_loss_pct':bt['avg_loss_pct'],
        'expectancy':bt['expectancy'],
        'model_scores':{
            'Mean_Reversion':    f"{m1p}/{m1t} ({m1d})",
            'Momentum_Decay':    f"{m2p}/{m2t} ({m2d})",
            'Volatility_Regime': f"{m3p}/{m3t} ({m3d})",
            'Order_Flow':        f"{m4p}/{m4t} ({m4d})",
            'Multi_Timeframe':   f"{m5p}/{m5t} ({m5d})"},
        'backtest':{
            'Win Rate':   f"{bt['win_rate']}%",
            'Trades':     f"{bt['wins']}W / {bt['losses']}L",
            'Avg Profit': f"+{bt['avg_profit_pct']}%",
            'Avg Loss':   f"-{bt['avg_loss_pct']}%",
            'Expectancy': f"{bt['expectancy']}% per trade"}
    }

def run_quantum_scan(max_stocks=None):
    stocks=ALL_STOCKS[:max_stocks] if max_stocks else ALL_STOCKS
    buy_r=[]; sell_r=[]
    for s in stocks:
        try:
            r=quantum_scan(s)
            if r:
                if r['direction']=="BUY": buy_r.append(r)
                else:                     sell_r.append(r)
        except: pass
    buy_r.sort(key=lambda x:x['score'],reverse=True)
    sell_r.sort(key=lambda x:x['score'],reverse=True)
    return {'buy':buy_r,'sell':sell_r}

# ============================================================
# FLASK
# ============================================================

app=Flask(__name__)
CORS(app,resources={r"/*":{"origins":"*"}})
cached_results=None
scan_in_progress=False

@app.route('/')
def home():
    return jsonify({'scanner':'QUANTUM Scanner v2.0 (Fyers + Supabase)',
                    'endpoints':{'/api/scan':'Run full scan',
                                 '/api/scan?max=50':'Quick scan',
                                 '/api/results':'Cached results',
                                 '/api/health':'Health check',
                                 '/api/token':'Force token refresh',
                                 '/api/debug':'Debug connectivity',
                                 '/api/backtest/<symbol>/<direction>':'Backtest trades'}})

@app.route('/api/health')
def health():
    return jsonify({'status':'QUANTUM SCANNER v2.0 ONLINE',
                    'timestamp':datetime.now().isoformat()})

@app.route('/api/token')
def api_token():
    global _access_token, _token_date
    _access_token=None; _token_date=None
    try:
        get_token()
        return jsonify({'status':'ok','message':'Token refreshed successfully'})
    except Exception as e:
        return jsonify({'status':'error','message':str(e)}),500

@app.route('/api/debug')
def api_debug():
    results = {}

    # Test 1: Token
    try:
        t = get_token()
        results['token'] = 'OK' if t else 'EMPTY'
    except Exception as e:
        results['token'] = f'FAILED: {str(e)}'

    # Test 2: Fetch RELIANCE
    try:
        df = fetch_stock_data('RELIANCE')
        results['fetch_RELIANCE'] = f'OK - {len(df)} rows' if df is not None else 'RETURNED NONE'
    except Exception as e:
        results['fetch_RELIANCE'] = f'FAILED: {str(e)}'

    # Test 3: Fetch M&M (symbol map test)
    try:
        df2 = fetch_stock_data('M&M')
        results['fetch_MM'] = f'OK - {len(df2)} rows' if df2 is not None else 'RETURNED NONE'
    except Exception as e:
        results['fetch_MM'] = f'FAILED: {str(e)}'

    # Test 4: Supabase
    try:
        r = requests.get(f"{SUPABASE_URL}/rest/v1/fyers_tokens",
                         headers=SUPA_HEADERS,
                         params={"id":"eq.1","select":"*"},
                         timeout=10)
        results['supabase'] = f'OK - status {r.status_code}'
    except Exception as e:
        results['supabase'] = f'FAILED: {str(e)}'

    return jsonify(results)

@app.route('/api/backtest/<symbol>/<direction>')
def api_backtest_trades(symbol, direction):
    try:
        r=requests.get(f"{SUPABASE_URL}/rest/v1/backtest_trades",
                       headers=SUPA_HEADERS,
                       params={"symbol":f"eq.{symbol.upper()}",
                               "direction":f"eq.{direction.upper()}",
                               "select":"*"},
                       timeout=10)
        if r.status_code==200:
            return jsonify({'symbol':symbol,'direction':direction,'trades':r.json()})
        return jsonify({'error':'fetch failed'}),500
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/scan')
def api_scan():
    global cached_results,scan_in_progress
    if scan_in_progress:
        return jsonify({'status':'busy','message':'Scan in progress'}),429
    mx=request.args.get('max',default=None,type=int)
    scan_in_progress=True
    try:
        results=run_quantum_scan(mx)
        cached_results={'buy':results['buy'],'sell':results['sell'],
                        'last_scan':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_stocks_scanned':mx if mx else len(ALL_STOCKS)}
        return jsonify(cached_results)
    except Exception as e:
        return jsonify({'status':'error','message':str(e)}),500
    finally:
        scan_in_progress=False

@app.route('/api/results')
def api_results():
    if cached_results is None:
        return jsonify({'status':'no_data','message':'No scan run yet'}),404
    return jsonify(cached_results)

application=app

if __name__=='__main__':
    app.run(host='0.0.0.0',port=10000)
