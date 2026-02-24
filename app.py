import pandas as pd
import numpy as np
import warnings
import pyotp
import requests
import hashlib
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import time
warnings.filterwarnings('ignore')

# ============================================================
# ██████╗ FYERS CREDENTIALS — FILL IN YOUR VALUES BELOW
# ============================================================

FYERS_USER_ID   = "YOUR_FYERS_USER_ID"    # e.g. "XM12345"
FYERS_PIN       = "YOUR_4_DIGIT_PIN"      # e.g. "1234"
FYERS_TOTP_KEY  = "YOUR_TOTP_SECRET_KEY"  # Long string shown when you enabled TOTP

# These are already filled from your app creation:
FYERS_CLIENT_ID  = "VS55VDHYCW-100"
FYERS_SECRET_KEY = "724FOKKSFS"
FYERS_REDIRECT   = "https://trade.fyers.in/api-login/redirect-uri/index.html"

# ============================================================
# FYERS TOKEN MANAGER (Auto-login, no manual steps needed)
# ============================================================

_access_token = None
_token_expiry = None

def generate_fyers_token():
    """Fully automated token generation using TOTP."""
    global _access_token, _token_expiry

    session = requests.Session()

    # Step 1: Send login OTP
    totp = pyotp.TOTP(FYERS_TOTP_KEY).now()
    url1 = "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
    r1 = session.post(url1, json={"fy_id": FYERS_USER_ID, "app_id": "2"})
    if r1.status_code != 200:
        raise Exception(f"send_login_otp failed: {r1.text}")
    request_key = r1.json().get("request_key")

    # Step 2: Verify TOTP
    url2 = "https://api-t2.fyers.in/vagator/v2/verify_otp"
    r2 = session.post(url2, json={"request_key": request_key, "otp": totp})
    if r2.status_code != 200:
        raise Exception(f"verify_otp failed: {r2.text}")
    request_key2 = r2.json().get("request_key")

    # Step 3: Verify PIN
    pin_hash = hashlib.sha256(FYERS_PIN.encode()).hexdigest()
    url3 = "https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
    r3 = session.post(url3, json={"request_key": request_key2, "identity_type": "pin", "identifier": pin_hash})
    if r3.status_code != 200:
        raise Exception(f"verify_pin failed: {r3.text}")
    access_token_temp = r3.json().get("data", {}).get("access_token")

    # Step 4: Get auth code
    url4 = "https://api-t2.fyers.in/api/v3/token"
    headers = {"Authorization": f"Bearer {access_token_temp}"}
    payload = {
        "fyers_id": FYERS_USER_ID,
        "app_id": FYERS_CLIENT_ID.split("-")[0],
        "redirect_uri": FYERS_REDIRECT,
        "appType": "100",
        "code_challenge": "",
        "state": "quantum",
        "scope": "",
        "nonce": "",
        "response_type": "code",
        "create_cookie": True
    }
    r4 = session.post(url4, headers=headers, json=payload)
    if r4.status_code != 200:
        raise Exception(f"get_auth_code failed: {r4.text}")
    auth_code = r4.json().get("Url", "").split("auth_code=")[-1].split("&")[0]

    # Step 5: Exchange auth code for access token
    app_hash = hashlib.sha256(f"{FYERS_CLIENT_ID}:{FYERS_SECRET_KEY}".encode()).hexdigest()
    url5 = "https://api-t2.fyers.in/api/v3/validate-authcode"
    r5 = session.post(url5, json={
        "grant_type": "authorization_code",
        "appIdHash": app_hash,
        "code": auth_code
    })
    if r5.status_code != 200:
        raise Exception(f"validate_authcode failed: {r5.text}")

    _access_token = r5.json().get("access_token")
    _token_expiry = datetime.now() + timedelta(hours=23)
    print(f"[Fyers] Token generated successfully at {datetime.now().strftime('%H:%M:%S')}")
    return _access_token

def get_token():
    """Return valid token, refreshing if expired."""
    global _access_token, _token_expiry
    if _access_token is None or _token_expiry is None or datetime.now() >= _token_expiry:
        generate_fyers_token()
    return _access_token

def fyers_headers():
    return {
        "Authorization": f"{FYERS_CLIENT_ID}:{get_token()}",
        "Content-Type": "application/json"
    }

# ============================================================
# FYERS DATA FETCH (Replaces yfinance)
# ============================================================

RESOLUTION_MAP = {
    "18mo": ("D", 540),   # Daily, ~18 months of bars
    "1y":   ("D", 365),
    "6mo":  ("D", 180),
}

def fetch_stock_data(symbol, period="18mo"):
    """Fetch OHLCV data from Fyers API."""
    try:
        resolution, days = RESOLUTION_MAP.get(period, ("D", 540))
        fyers_symbol = f"NSE:{symbol}-EQ"

        date_to   = datetime.now()
        date_from = date_to - timedelta(days=days)

        url = "https://api-t2.fyers.in/data/v3/history"
        params = {
            "symbol":      fyers_symbol,
            "resolution":  resolution,
            "date_format": "1",
            "range_from":  date_from.strftime("%Y-%m-%d"),
            "range_to":    date_to.strftime("%Y-%m-%d"),
            "cont_flag":   "1"
        }

        r = requests.get(url, headers=fyers_headers(), params=params, timeout=15)
        if r.status_code != 200:
            return None

        data = r.json()
        if data.get("s") != "ok" or not data.get("candles"):
            return None

        candles = data["candles"]
        df = pd.DataFrame(candles, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()

        if df.empty or len(df) < 60:
            return None
        return df

    except Exception as e:
        print(f"[Fyers] fetch_stock_data error for {symbol}: {e}")
        return None

# ============================================================
# HELPER
# ============================================================

def percentileofscore(arr, score):
    arr = np.array(arr)
    return (np.sum(arr <= score) / len(arr)) * 100

# ============================================================
# FNO + ALL STOCKS LISTS (unchanged)
# ============================================================

FNO_STOCKS = {
    "AARTIIND","ABB","ABBOTINDIA","ABCAPITAL","ABFRL",
    "ACC","ADANIENT","ADANIENSOL","ADANIGREEN","ADANIPORTS",
    "ADANIPOWER","ALKEM","AMBUJACEM","ANGELONE","APLAPOLLO",
    "APOLLOHOSP","APOLLOTYRE","ASHOKLEY","ASIANPAINT",
    "ASTRAL","ATGL","ATUL","AUBANK","AUROPHARMA","AWL",
    "AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE",
    "BALKRISIND","BALRAMCHIN","BANDHANBNK","BANKBARODA",
    "BATAINDIA","BDL","BEL","BERGEPAINT","BHARATFORG",
    "BHARTIARTL","BHEL","BIOCON","BOSCHLTD","BPCL",
    "BRIGADE","BRITANNIA","BSE","BSOFT",
    "CANBK","CANFINHOME","CDSL","CHAMBLFERT","CHOLAFIN",
    "CIPLA","CLEAN","COALINDIA","COCHINSHIP","COFORGE",
    "COLPAL","CONCOR","COROMANDEL","CREDITACC","CRISIL",
    "CROMPTON","CUB","CUMMINSIND","CYIENT",
    "DABUR","DALBHARAT","DEEPAKNTR","DELHIVERY","DELTACORP",
    "DEVYANI","DIVISLAB","DIXON","DLF","DRREDDY",
    "EICHERMOT","EMCURE","ESCORTS","ETERNAL","EXIDEIND",
    "FACT","FEDERALBNK","GAIL","GLENMARK","GMRAIRPORT",
    "GNFC","GODREJCP","GODREJPROP","GRANULES","GRASIM",
    "GRSE","GSPL","GUJGASLTD",
    "HAL","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK",
    "HDFCLIFE","HEROMOTOCO","HINDALCO","HINDCOPPER",
    "HINDPETRO","HINDUNILVR","HONAUT","HUDCO",
    "ICICIBANK","ICICIGI","ICICIPRULI","IDEA","IDFCFIRSTB",
    "IEX","IGL","INDHOTEL","INDIACEM","INDIAMART",
    "INDIANB","INDIGO","INDUSINDBK","INDUSTOWER","INFY",
    "IOC","IPCALAB","IRCTC","IREDA","IRFC","ITC",
    "ITCHOTELS","JINDALSTEL","JIOFIN","JKCEMENT",
    "JSWENERGY","JSWINFRA","JSWSTEEL","JUBLFOOD",
    "KALYANKJIL","KEI","KFINTECH","KOTAKBANK","KPITTECH",
    "LALPATHLAB","LAURUSLABS","LICHSGFIN","LICI","LODHA",
    "LT","LTIM","LTTS","LUPIN",
    "M&M","M&MFIN","MANAPPURAM","MANKIND","MAPMYINDIA",
    "MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX",
    "METROPOLIS","MFSL","MGL","MOTHERSON","MPHASIS",
    "MRF","MUTHOOTFIN",
    "NATIONALUM","NAUKRI","NAVINFLUOR","NESTLEIND","NHPC",
    "NMDC","NTPC",
    "OBEROIRLTY","OFSS","ONGC",
    "PAGEIND","PATANJALI","PAYTM","PERSISTENT","PETRONET",
    "PFC","PIDILITIND","PIIND","PNB","POLICYBZR","POLYCAB",
    "POONAWALLA","POWERGRID","PRESTIGE","PVRINOX",
    "RAMCOCEM","RBLBANK","RECLTD","RELIANCE","RVNL",
    "SAIL","SBICARD","SBILIFE","SBIN","SHREECEM",
    "SHRIRAMFIN","SIEMENS","SJVN","SONACOMS","SRF",
    "SUNPHARMA","SUNTV","SUPREMEIND","SUZLON","SYNGENE","SYRMA",
    "TATACHEM","TATACOMM","TATACONSUM","TATAELXSI",
    "TATAMOTORS","TATAPOWER","TATASTEEL","TATATECH",
    "TCS","TECHM","TIINDIA","TITAN","TORNTPHARM",
    "TORNTPOWER","TRENT","TVSMOTOR",
    "UBL","ULTRACEMCO","UNIONBANK","UNITDSPR","UPL",
    "VBL","VEDL","VOLTAS",
    "WIPRO","YESBANK","ZEEL","ZOMATO","ZYDUSLIFE"
}

ALL_STOCKS = list(set([
    "360ONE","3MINDIA","5PAISA","AARTIDRUGS","AARTIIND","AAVAS",
    "ABB","ABBOTINDIA","ABCAPITAL","ABFRL","ACC",
    "ADANIENT","ADANIENSOL","ADANIGREEN","ADANIPORTS",
    "ADANIPOWER","ADVENZYMES","AFFLE","AIAENG","AJANTPHARM",
    "AKZOINDIA","ALEMBICLTD","ALKEM","ALKYLAMINE","ALLCARGO",
    "ALOKINDS","AMBER","AMBUJACEM","ANANDRATHI","ANGELONE",
    "APLAPOLLO","APOLLOHOSP","APOLLOTYRE","APTUS","ASHOKLEY",
    "ASIANPAINT","ASTERDM","ASTRAL","ATGL","ATUL","AUBANK",
    "AUROPHARMA","AVANTIFEED","AWL","AXISBANK",
    "BAJAJ-AUTO","BAJAJCON","BAJAJFINSV","BAJAJHLDNG","BAJFINANCE",
    "BALAMINES","BALKRISIND","BALMLAWRIE","BALRAMCHIN",
    "BANDHANBNK","BANKBARODA","BANKINDIA","BASF","BATAINDIA",
    "BAYERCROP","BDL","BEL","BEML","BERGEPAINT",
    "BHARATFORG","BHARTIARTL","BHEL","BIKAJI","BIOCON",
    "BIRLACORPN","BLUESTARCO","BOSCHLTD","BPCL",
    "BRIGADE","BRITANNIA","BSE","BSOFT",
    "CAMPUS","CAMS","CANBK","CANFINHOME","CAPLIPOINT",
    "CARBORUNIV","CASTROLIND","CDSL","CEATLTD","CENTRALBK",
    "CENTURYPLY","CERA","CGCL","CGPOWER","CHALET",
    "CHAMBLFERT","CHEMPLASTS","CHOLAFIN","CIPLA","CLEAN",
    "COALINDIA","COCHINSHIP","COFORGE","COLPAL","CONCOR",
    "CONCORDBIO","COROMANDEL","CRAFTSMAN","CREDITACC",
    "CRISIL","CROMPTON","CSBBANK","CUB","CUMMINSIND","CYIENT",
    "DABUR","DALBHARAT","DATAPATTNS","DCMSHRIRAM",
    "DEEPAKFERT","DEEPAKNTR","DELHIVERY","DELTACORP",
    "DEVYANI","DIVISLAB","DIXON","DLF","DMART","DRREDDY",
    "EASEMYTRIP","ECLERX","EIDPARRY","EICHERMOT",
    "ELECON","ELGIEQUIP","EMAMILTD","EMCURE","ENDURANCE",
    "ENGINERSIN","EPL","EQUITASBNK","ERIS","ESCORTS",
    "ETERNAL","EXIDEIND","FACT","FEDERALBNK","FINCABLES",
    "FINEORG","FINPIPE","FLUOROCHEM","FORTIS","FSL",
    "GAIL","GALAXYSURF","GARFIBRES","GESHIP","GICRE",
    "GILLETTE","GLAND","GLAXO","GLENMARK","GMRAIRPORT",
    "GNFC","GODREJCP","GODREJIND","GODREJPROP","GPPL",
    "GRANULES","GRAPHITE","GRASIM","GRINDWELL","GRINFRA",
    "GRSE","GSPL","GUJALKALI","GUJGASLTD",
    "HAL","HAPPSTMNDS","HATSUN","HAVELLS","HCLTECH",
    "HDFCAMC","HDFCBANK","HDFCLIFE","HEMIPROP","HEROMOTOCO",
    "HFCL","HIKAL","HINDALCO","HINDCOPPER","HINDPETRO",
    "HINDUNILVR","HINDWAREAP","HOMEFIRST","HONAUT","HSCL","HUDCO",
    "IBREALEST","ICICIBANK","ICICIGI","ICICIPRULI","IDEA",
    "IDFCFIRSTB","IEX","IGL","IIFL","INDHOTEL","INDIACEM",
    "INDIAMART","INDIANB","INDIGO","INDUSINDBK","INDUSTOWER",
    "INFY","INGERRAND","INTELLECT","IOC","IPCALAB","IRCTC",
    "IREDA","IRFC","ITC","ITCHOTELS","ITI",
    "JBCHEPHARM","JBMA","JINDALSTEL","JIOFIN","JKCEMENT",
    "JKLAKSHMI","JKPAPER","JMFINANCIL","JSL","JSWENERGY",
    "JSWINFRA","JSWSTEEL","JTEKTINDIA","JUBLFOOD","JUSTDIAL",
    "JYOTHYLAB","KAJARIACER","KALPATARU","KALYANKJIL",
    "KANSAINER","KAYNES","KEC","KEI","KFINTECH",
    "KIRLOSENG","KNRCON","KPITTECH","KSB","KOTAKBANK",
    "KRBL","KSCL","LAOPALA","LALPATHLAB","LATENTVIEW",
    "LAURUSLABS","LEMONTREE","LICHSGFIN","LICI","LINDEINDIA",
    "LLOYDSME","LODHA","LT","LTIM","LTTS","LUPIN","LUXIND","LXCHEM",
    "M&M","M&MFIN","MAHABANK","MAHLIFE","MAHLOG","MANAPPURAM",
    "MANKIND","MAPMYINDIA","MARICO","MARKSANS","MARUTI","MASTEK",
    "MAXHEALTH","MAZDOCK","MCX","METROBRAND","METROPOLIS",
    "MFSL","MGL","MIDHANI","MINDACORP","MOTHERSON","MPHASIS",
    "MRF","MSUMI","MTARTECH","MUTHOOTFIN",
    "NATCOPHARM","NATIONALUM","NAUKRI","NAVINFLUOR","NAVNETEDUL",
    "NCC","NESTLEIND","NETWORK18","NEWGEN","NHPC","NIACL",
    "NLCINDIA","NMDC","NOCIL","NTPC","NUVOCO",
    "OBEROIRLTY","OFSS","OIL","OLECTRA","ONGC","ORIENTELEC",
    "PAGEIND","PAISALO","PATANJALI","PAYTM","PCBL",
    "PERSISTENT","PETRONET","PFC","PFIZER","PGHH",
    "PIDILITIND","PIIND","PNB","PNBHOUSING","POLICYBZR",
    "POLYCAB","POONAWALLA","POWERGRID","POWERINDIA","PPLPHARMA",
    "PRESTIGE","PRINCEPIPE","PRSMJOHNSN","PVRINOX","QUESS",
    "RADICO","RAIN","RAILTEL","RAJESHEXPO","RALLIS",
    "RAMCOCEM","RATNAMANI","RAYMOND","RBA","RBLBANK","RCF",
    "RECLTD","REDINGTON","RELAXO","RELIANCE","RELIGARE",
    "RITES","RKFORGE","ROUTE","RVNL",
    "SAREGAMA","SAIL","SBICARD","SBILIFE","SBIN","SCHAEFFLER",
    "SCHNEIDER","SHILPAMED","SHREECEM","SHRIRAMFIN","SIEMENS",
    "SJVN","SKFINDIA","SOBHA","SOLARINDS","SONACOMS",
    "SONATSOFTW","SOUTHBANK","SPARC","SRF","STAR","STARHEALTH",
    "STEL","STLTECH","SUDARSCHEM","SUMICHEM","SUNDARMFIN",
    "SUNDRMFAST","SUNPHARMA","SUNTV","SUPREMEIND","SUVEN",
    "SUZLON","SWSOLAR","SYNGENE","SYRMA",
    "TANLA","TARSONS","TATACHEM","TATACOMM","TATACONSUM",
    "TATAELXSI","TATAINVEST","TATAPOWER","TATASTEEL","TATATECH",
    "TCS","TECHM","THERMAX","TIINDIA","TIMKEN","TITAN",
    "TORNTPHARM","TORNTPOWER","TRENT","TRIDENT","TRITURBINE",
    "TRIVENI","TTML","TVSMOTOR",
    "UBL","UJJIVANSFB","ULTRACEMCO","UNIONBANK","UNITDSPR",
    "UPL","UTIAMC","VAKRANGEE","VARROC","VBL","VEDL","VENKEYS",
    "VGUARD","VINATIORGA","VIPIND","VOLTAS","VSTIND",
    "WELCORP","WELSPUNLIV","WHIRLPOOL","WIPRO",
    "YESBANK","ZEEL","ZENSARTECH","ZFCVINDIA","ZYDUSLIFE","ZYDUSWELL"
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
    gain = delta.where(delta > 0, 0.0).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def atr(high, low, close, length=14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()

def bbands(series, length=20, std=2):
    mid = sma(series, length)
    rolling_std = series.rolling(window=length).std()
    upper = mid + std * rolling_std
    lower = mid - std * rolling_std
    return upper, mid, lower

def obv(close, volume):
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    return (volume * direction).cumsum()

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

    df['ZScore'] = (df['Close'] - df['SMA20']) / df['Close'].rolling(20).std()
    df['RSI']    = rsi(df['Close'], 14)

    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df['Close'], 12, 26, 9)
    df['RSI_ROC']    = df['RSI'].diff(3)
    df['MACD_accel'] = df['MACD_hist'].diff(1)

    df['ATR']        = atr(df['High'], df['Low'], df['Close'], 14)
    df['ATR_pct']    = (df['ATR'] / df['Close']) * 100
    df['ATR_percentile'] = df['ATR'].rolling(100).apply(
        lambda x: percentileofscore(x, x.iloc[-1]), raw=False)

    df['Returns'] = df['Close'].pct_change()
    df['HV20']    = df['Returns'].rolling(20).std() * np.sqrt(252) * 100

    df['Vol_SMA20']      = sma(df['Volume'], 20)
    df['VWAP']           = vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Up_Vol']         = np.where(df['Close'] > df['Open'], df['Volume'], 0)
    df['Down_Vol']       = np.where(df['Close'] < df['Open'], df['Volume'], 0)
    df['Vol_Delta']      = pd.Series(df['Up_Vol']).rolling(5).sum() - pd.Series(df['Down_Vol']).rolling(5).sum()
    df['Vol_Delta_Ratio']= pd.Series(df['Up_Vol']).rolling(5).sum() / (pd.Series(df['Down_Vol']).rolling(5).sum() + 1)
    df['OBV']            = pd.Series(obv(df['Close'], df['Volume']), index=df.index)
    df['OBV_SMA']        = sma(df['OBV'], 20)

    df['Weekly_EMA9']  = ema(df['Close'], 45)
    df['Weekly_EMA21'] = ema(df['Close'], 105)
    df['Monthly_EMA']  = ema(df['Close'], 200)

    df['High_52w'] = df['High'].rolling(window=min(252, len(df)), min_periods=50).max()
    df['Low_52w']  = df['Low'].rolling(window=min(252, len(df)), min_periods=50).min()

    recent_high = df['High'].iloc[-60:].max()
    recent_low  = df['Low'].iloc[-60:].min()
    fib_range   = recent_high - recent_low
    df['Fib_236'] = recent_high - fib_range * 0.236
    df['Fib_382'] = recent_high - fib_range * 0.382
    df['Fib_500'] = recent_high - fib_range * 0.500
    df['Fib_618'] = recent_high - fib_range * 0.618

    return df.dropna()

# ============================================================
# 5 MODELS
# ============================================================

def model_1_mean_reversion(df):
    latest = df.iloc[-1]; prev = df.iloc[-2]
    buy  = [latest['ZScore']<-1.0, latest['Close']<=latest['BB_lower']*1.02,
            latest['Close']<latest['KC_lower']*1.03, latest['RSI']<40, latest['Close']>prev['Close']]
    sell = [latest['ZScore']>1.0,  latest['Close']>=latest['BB_upper']*0.98,
            latest['Close']>latest['KC_upper']*0.97, latest['RSI']>60, latest['Close']<prev['Close']]
    bs, ss = sum(buy), sum(sell)
    if bs >= 3: return bs, 5, "BUY",  "Mean reversion from oversold"
    if ss >= 3: return ss, 5, "SELL", "Mean reversion from overbought"
    return max(bs, ss), 5, "NEUTRAL", ""

def model_2_momentum_decay(df):
    latest = df.iloc[-1]; prev = df.iloc[-2]
    buy  = [latest['MACD_hist']>prev['MACD_hist'], latest['MACD_accel']>0,
            latest['RSI']>prev['RSI'], 40<latest['RSI']<65, latest['Close']>prev['Close']]
    sell = [latest['MACD_hist']<prev['MACD_hist'], latest['MACD_accel']<0,
            latest['RSI']<prev['RSI'], 35<latest['RSI']<60, latest['Close']<prev['Close']]
    bs, ss = sum(buy), sum(sell)
    if bs >= 3: return bs, 5, "BUY",  "Momentum accelerating up"
    if ss >= 3: return ss, 5, "SELL", "Momentum decelerating down"
    return max(bs, ss), 5, "NEUTRAL", ""

def model_3_volatility_regime(df):
    latest = df.iloc[-1]; prev = df.iloc[-2]
    ap = latest['ATR_percentile']
    if ap < 30:   regime = "LOW_VOL"
    elif ap < 70: regime = "NORMAL_VOL"
    else:         regime = "HIGH_VOL"
    if regime == "LOW_VOL":
        buy  = [latest['BB_width']<df['BB_width'].quantile(0.3), latest['Close']>latest['BB_mid'],
                latest['Volume']>latest['Vol_SMA20']*1.1, latest['MACD_hist']>0, latest['Close']>latest['EMA21']]
        sell = [latest['BB_width']<df['BB_width'].quantile(0.3), latest['Close']<latest['BB_mid'],
                latest['Volume']>latest['Vol_SMA20']*1.1, latest['MACD_hist']<0, latest['Close']<latest['EMA21']]
    elif regime == "NORMAL_VOL":
        buy  = [latest['Close']>latest['EMA21'], latest['EMA21']>latest['EMA50'],
                45<latest['RSI']<75, latest['MACD']>latest['MACD_signal'], latest['Volume']>latest['Vol_SMA20']*0.8]
        sell = [latest['Close']<latest['EMA21'], latest['EMA21']<latest['EMA50'],
                25<latest['RSI']<55, latest['MACD']<latest['MACD_signal'], latest['Volume']>latest['Vol_SMA20']*0.8]
    else:
        buy  = [latest['ZScore']<-1.0, latest['RSI']<35, latest['Close']<latest['BB_lower']*1.02,
                latest['Close']>prev['Close'], latest['ATR_pct']>1.5]
        sell = [latest['ZScore']>1.0,  latest['RSI']>65, latest['Close']>latest['BB_upper']*0.98,
                latest['Close']<prev['Close'], latest['ATR_pct']>1.5]
    bs, ss = sum(buy), sum(sell)
    if bs >= 3: return bs, 5, "BUY",  f"Regime: {regime}"
    if ss >= 3: return ss, 5, "SELL", f"Regime: {regime}"
    return max(bs, ss), 5, "NEUTRAL", regime

def model_4_order_flow(df):
    latest = df.iloc[-1]
    buy  = [latest['Vol_Delta_Ratio']>1.3, latest['OBV']>latest['OBV_SMA'],
            latest['Close']>latest['VWAP'], latest['Volume']>latest['Vol_SMA20']*0.9, latest['Close']>latest['Open']]
    sell = [latest['Vol_Delta_Ratio']<0.75, latest['OBV']<latest['OBV_SMA'],
            latest['Close']<latest['VWAP'], latest['Volume']>latest['Vol_SMA20']*0.9, latest['Close']<latest['Open']]
    bs, ss = sum(buy), sum(sell)
    if bs >= 3: return bs, 5, "BUY",  "Accumulation detected"
    if ss >= 3: return ss, 5, "SELL", "Distribution detected"
    return max(bs, ss), 5, "NEUTRAL", ""

def model_5_multi_timeframe(df):
    latest = df.iloc[-1]
    nfb = any([abs(latest['Close']-latest[f])/latest['Close']<0.03 for f in ['Fib_382','Fib_500','Fib_618']])
    nfs = any([abs(latest['Close']-latest[f])/latest['Close']<0.03 for f in ['Fib_236','Fib_382','Fib_500']])
    buy  = [latest['EMA9']>latest['EMA21'], latest['Weekly_EMA9']>latest['Weekly_EMA21'],
            latest['Close']>latest['Monthly_EMA'], nfb, 35<latest['RSI']<70]
    sell = [latest['EMA9']<latest['EMA21'], latest['Weekly_EMA9']<latest['Weekly_EMA21'],
            latest['Close']<latest['Monthly_EMA'], nfs, 30<latest['RSI']<65]
    bs, ss = sum(buy), sum(sell)
    if bs >= 3: return bs, 5, "BUY",  "Multi-timeframe bullish"
    if ss >= 3: return ss, 5, "SELL", "Multi-timeframe bearish"
    return max(bs, ss), 5, "NEUTRAL", ""

# ============================================================
# RISK
# ============================================================

def calculate_risk(df, direction):
    latest = df.iloc[-1]; entry_price = latest['Close']; a = latest['ATR']
    sl_pct = max(1.5, min(5.0, (a * 1.5 / entry_price) * 100))
    sl_distance = entry_price * (sl_pct / 100)
    vol_pct = (a / entry_price) * 100
    rh = df['High'].iloc[-20:].max(); rl = df['Low'].iloc[-20:].min()
    sw = ((rh - rl) / entry_price) * 100
    if   vol_pct > 3 and sw > 15: rr = 3.0
    elif vol_pct > 2 and sw > 10: rr = 2.5
    elif vol_pct > 1.5 or sw > 8: rr = 2.0
    else:                          rr = 1.5
    td = sl_distance * rr
    if direction == "BUY":
        sl = round(entry_price - sl_distance, 2)
        target = max(round(entry_price + td, 2), round(entry_price * 1.02, 2))
    else:
        sl = round(entry_price + sl_distance, 2)
        target = min(round(entry_price - td, 2), round(entry_price * 0.98, 2))
    ar = abs(entry_price - sl); aw = abs(target - entry_price)
    return {'entry': round(entry_price, 2), 'stop_loss': sl, 'target': target,
            'rr_ratio': round(aw / ar, 1) if ar > 0 else 0,
            'sl_pct': round(sl_pct, 2), 'atr': round(a, 2)}

# ============================================================
# BACKTEST
# ============================================================

def backtest_quantum(df, direction):
    wins = 0; losses = 0; profits = []; loss_amounts = []
    in_trade = False; trade_exit_idx = 0
    if len(df) < 100:
        return {'win_rate':50,'total_trades':0,'wins':0,'losses':0,
                'avg_profit_pct':0,'avg_loss_pct':0,'expectancy':0}
    for i in range(60, len(df) - 10, 5):
        if in_trade and i <= trade_exit_idx: continue
        in_trade = False
        curr = df.iloc[i]
        if direction == "BUY"  and curr['Close'] < curr['EMA21'] and curr['RSI'] < 35: continue
        if direction == "SELL" and curr['Close'] > curr['EMA21'] and curr['RSI'] > 65: continue
        sub_df = df.iloc[:i+1].copy()
        if len(sub_df) < 60: continue
        try:
            _,_,d1,_ = model_1_mean_reversion(sub_df)
            _,_,d2,_ = model_2_momentum_decay(sub_df)
            _,_,d3,_ = model_3_volatility_regime(sub_df)
            _,_,d4,_ = model_4_order_flow(sub_df)
            _,_,d5,_ = model_5_multi_timeframe(sub_df)
        except: continue
        dirs = [d1, d2, d3, d4, d5]; signal = None
        if direction == "BUY"  and dirs.count("BUY") >= 3:  signal = "BUY"
        if direction == "SELL" and dirs.count("SELL") >= 3: signal = "SELL"
        if signal is None: continue
        ep = df.iloc[i]['Close']; a = df.iloc[i]['ATR']
        sd = ep * (max(1.5, min(5.0, (a * 1.5 / ep) * 100)) / 100)
        td = sd * 2.0
        sl  = ep - sd if signal == "BUY" else ep + sd
        tgt = ep + td if signal == "BUY" else ep - td
        for j in range(1, min(30, len(df) - i - 1) + 1):
            f = df.iloc[i + j]
            if signal == "BUY":
                if f['Low']  <= sl:  losses += 1; loss_amounts.append((sd/ep)*100); trade_exit_idx=i+j; in_trade=True; break
                if f['High'] >= tgt: wins   += 1; profits.append((td/ep)*100);      trade_exit_idx=i+j; in_trade=True; break
            else:
                if f['High'] >= sl:  losses += 1; loss_amounts.append((sd/ep)*100); trade_exit_idx=i+j; in_trade=True; break
                if f['Low']  <= tgt: wins   += 1; profits.append((td/ep)*100);      trade_exit_idx=i+j; in_trade=True; break
        if wins + losses >= 15: break
    total = wins + losses
    if total == 0:
        return {'win_rate':50,'total_trades':0,'wins':0,'losses':0,
                'avg_profit_pct':0,'avg_loss_pct':0,'expectancy':0}
    wr = round((wins / total) * 100, 1)
    ap = round(np.mean(profits), 2)      if profits      else 0
    al = round(np.mean(loss_amounts), 2) if loss_amounts else 0
    return {'win_rate':wr,'total_trades':total,'wins':wins,'losses':losses,
            'avg_profit_pct':ap,'avg_loss_pct':al,
            'expectancy':round((wr/100*ap) - ((100-wr)/100*al), 2)}

# ============================================================
# MASTER SCAN
# ============================================================

def quantum_scan(symbol):
    df = fetch_stock_data(symbol)
    if df is None: return None
    try: df = calculate_all_indicators(df)
    except: return None
    if df.empty or len(df) < 80: return None
    try:
        m1p,m1t,m1d,m1r = model_1_mean_reversion(df)
        m2p,m2t,m2d,m2r = model_2_momentum_decay(df)
        m3p,m3t,m3d,m3r = model_3_volatility_regime(df)
        m4p,m4t,m4d,m4r = model_4_order_flow(df)
        m5p,m5t,m5d,m5r = model_5_multi_timeframe(df)
    except: return None

    dirs = [m1d,m2d,m3d,m4d,m5d]
    bc = dirs.count("BUY"); sc = dirs.count("SELL")
    if   bc >= 3: direction = "BUY";  agreement = bc
    elif sc >= 3: direction = "SELL"; agreement = sc
    else: return None

    is_fno = symbol in FNO_STOCKS
    if direction == "SELL" and not is_fno: return None

    risk = calculate_risk(df, direction)
    if direction == "BUY"  and risk['target'] <= risk['entry']: return None
    if direction == "SELL" and risk['target'] >= risk['entry']: return None
    if risk['rr_ratio'] < 1.2: return None

    bt = backtest_quantum(df, direction)
    if bt['total_trades'] >= 3 and bt['win_rate']   < 40: return None
    if bt['total_trades'] >= 3 and bt['expectancy'] < 0:  return None

    tp = m1p+m2p+m3p+m4p+m5p; tt = m1t+m2t+m3t+m4t+m5t
    mp = (tp/tt)*100; ab = (agreement-3)*5; bb = 0
    if bt['total_trades'] >= 3:
        if   bt['win_rate'] >= 65 and bt['expectancy'] > 1.5: bb = 10
        elif bt['win_rate'] >= 55 and bt['expectancy'] > 0:   bb = 6
        elif bt['win_rate'] >= 45:                             bb = 3
    ts = round(min(mp+ab+bb, 100), 1)

    if ts >= 85 and bt['total_trades'] >= 3 and bt['win_rate'] >= 60 and bt['expectancy'] > 0:
        grade = "A+"; conf = "HIGH"
    elif ts >= 80 and bt['total_trades'] >= 3 and bt['win_rate'] >= 50 and bt['expectancy'] > 0:
        grade = "A+"; conf = "MEDIUM"
    else: return None

    am = [n for n,d in [("MRB",m1d),("MDD",m2d),("VRC",m3d),("OFI",m4d),("MTC",m5d)] if d == direction]

    return {
        'symbol': symbol, 'direction': direction,
        'entry': risk['entry'], 'stop_loss': risk['stop_loss'], 'target': risk['target'],
        'score': ts, 'grade': grade, 'rr_ratio': risk['rr_ratio'],
        'is_fno': is_fno, 'confidence': conf,
        'models_agreeing': f"{agreement}/5", 'agreeing_models': ", ".join(am),
        'historical_win_rate': bt['win_rate'], 'total_past_trades': bt['total_trades'],
        'past_wins': bt['wins'], 'past_losses': bt['losses'],
        'avg_profit_pct': bt['avg_profit_pct'], 'avg_loss_pct': bt['avg_loss_pct'],
        'expectancy': bt['expectancy'],
        'model_scores': {
            'Mean_Reversion':    f"{m1p}/{m1t} ({m1d})",
            'Momentum_Decay':    f"{m2p}/{m2t} ({m2d})",
            'Volatility_Regime': f"{m3p}/{m3t} ({m3d})",
            'Order_Flow':        f"{m4p}/{m4t} ({m4d})",
            'Multi_Timeframe':   f"{m5p}/{m5t} ({m5d})"},
        'backtest': {
            'Win Rate':   f"{bt['win_rate']}%",
            'Trades':     f"{bt['wins']}W / {bt['losses']}L",
            'Avg Profit': f"+{bt['avg_profit_pct']}%",
            'Avg Loss':   f"-{bt['avg_loss_pct']}%",
            'Expectancy': f"{bt['expectancy']}% per trade"}
    }

def run_quantum_scan(max_stocks=None):
    stocks = ALL_STOCKS[:max_stocks] if max_stocks else ALL_STOCKS
    buy_r = []; sell_r = []
    for s in stocks:
        try:
            r = quantum_scan(s)
            if r:
                if r['direction'] == "BUY": buy_r.append(r)
                else:                       sell_r.append(r)
        except: pass
    buy_r.sort(key=lambda x: x['score'], reverse=True)
    sell_r.sort(key=lambda x: x['score'], reverse=True)
    return {'buy': buy_r, 'sell': sell_r}

# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
cached_results  = None
scan_in_progress = False

@app.route('/')
def home():
    return jsonify({'scanner': 'QUANTUM Scanner v2.0 (Fyers)',
                    'endpoints': {'/api/scan': 'Run scan', '/api/scan?max=50': 'Quick scan',
                                  '/api/results': 'Cached', '/api/health': 'Health',
                                  '/api/token': 'Force token refresh'}})

@app.route('/api/health')
def health():
    return jsonify({'status': 'QUANTUM SCANNER v2.0 ONLINE (Fyers)',
                    'timestamp': datetime.now().isoformat()})

@app.route('/api/token')
def api_token():
    """Force a fresh Fyers token — call this if scans start failing."""
    try:
        generate_fyers_token()
        return jsonify({'status': 'ok', 'message': 'Token refreshed successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/scan')
def api_scan():
    global cached_results, scan_in_progress
    if scan_in_progress:
        return jsonify({'status': 'busy', 'message': 'Scan in progress'}), 429
    mx = request.args.get('max', default=None, type=int)
    scan_in_progress = True
    try:
        results = run_quantum_scan(mx)
        cached_results = {
            'buy': results['buy'], 'sell': results['sell'],
            'last_scan': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_stocks_scanned': mx if mx else len(ALL_STOCKS)
        }
        return jsonify(cached_results)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        scan_in_progress = False

@app.route('/api/results')
def api_results():
    if cached_results is None:
        return jsonify({'status': 'no_data', 'message': 'No scan run yet'}), 404
    return jsonify(cached_results)

application = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
