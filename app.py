import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import warnings
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import time
warnings.filterwarnings('ignore')

def percentileofscore(arr, score):
    arr = np.array(arr)
    count = np.sum(arr <= score)
    return (count / len(arr)) * 100

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

ALL_STOCKS = [
    "360ONE","3MINDIA","5PAISA","AARTIDRUGS","AARTIIND","AAVAS",
    "ABB","ABBOTINDIA","ABCAPITAL","ABFRL","ACC",
    "ADANIENT","ADANIENSOL","ADANIGREEN","ADANIPORTS",
    "ADANIPOWER","ADVENZYMES","AFFLE",
    "AIAENG","AJANTPHARM","AKZOINDIA",
    "ALEMBICLTD","ALKEM","ALKYLAMINE","ALLCARGO","ALOKINDS",
    "AMBER","AMBUJACEM","ANANDRATHI",
    "ANGELONE","APLAPOLLO","APOLLOHOSP","APOLLOTYRE",
    "APTUS","ARE&M","ASAHIINDIA","ASHOKLEY","ASIANPAINT",
    "ASTERDM","ASTRAL","ATGL","ATUL","AUBANK",
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
    "CHAMBLFERT","CHEMPLASTS","CHOLAFIN",
    "CIPLA","CLEAN","COALINDIA",
    "COCHINSHIP","COFORGE","COLPAL","CONCOR",
    "CONCORDBIO","COROMANDEL","CRAFTSMAN","CREDITACC",
    "CRISIL","CROMPTON","CSBBANK","CUB","CUMMINSIND","CYIENT",
    "DABUR","DALBHARAT","DATAPATTNS","DCMSHRIRAM",
    "DEEPAKFERT","DEEPAKNTR","DELHIVERY","DELTACORP",
    "DEVYANI","DIVISLAB","DIXON","DLF","DMART",
    "DRREDDY",
    "EASEMYTRIP","ECLERX","EIDPARRY","EICHERMOT",
    "ELECON","ELGIEQUIP","EMAMILTD","EMCURE","ENDURANCE",
    "ENGINERSIN","EPL","EQUITASBNK","ERIS","ESCORTS",
    "ETERNAL","EXIDEIND",
    "FACT","FEDERALBNK","FINCABLES","FINEORG",
    "FINPIPE","FLUOROCHEM","FORTIS","FSL",
    "GAIL","GALAXYSURF","GARFIBRES","GESHIP",
    "GICRE","GILLETTE","GLAND","GLAXO","GLENMARK",
    "GMRAIRPORT","GNFC","GODREJCP",
    "GODREJIND","GODREJPROP","GPPL",
    "GRANULES","GRAPHITE","GRASIM","GRINDWELL",
    "GRINFRA","GRSE","GSPL","GUJALKALI","GUJGASLTD",
    "HAL","HAPPSTMNDS","HATSUN",
    "HAVELLS","HCLTECH","HDFCAMC","HDFCBANK",
    "HDFCLIFE","HEMIPROP","HEROMOTOCO",
    "HFCL","HIKAL","HINDALCO","HINDCOPPER",
    "HINDPETRO","HINDUNILVR","HINDWAREAP",
    "HOMEFIRST","HONAUT","HSCL","HUDCO",
    "IBREALEST","ICICIBANK","ICICIGI","ICICIPRULI",
    "IDEA","IDFCFIRSTB","IEX","IGL",
    "IIFL","INDHOTEL","INDIACEM","INDIAMART",
    "INDIANB","INDIGO","INDUSINDBK",
    "INDUSTOWER","INFY","INGERRAND",
    "INTELLECT","IOC","IPCALAB","IRCTC",
    "IREDA","IRFC","ITC","ITCHOTELS","ITI",
    "JBCHEPHARM","JBMA","JINDALSTEL","JIOFIN","JKCEMENT",
    "JKLAKSHMI","JKPAPER","JMFINANCIL","JSL","JSWENERGY",
    "JSWINFRA","JSWSTEEL","JTEKTINDIA","JUBLFOOD","JUSTDIAL",
    "JYOTHYLAB","KAJARIACER","KALPATARU","KALYANKJIL",
    "KANSAINER","KAYNES","KEC","KEI","KFINTECH",
    "KIRLOSENG","KNRCON","KPITTECH","KSB",
    "KOTAKBANK","KRBL","KSCL","LAOPALA","LALPATHLAB",
    "LATENTVIEW","LAURUSLABS","LEMONTREE",
    "LICHSGFIN","LICI","LINDEINDIA","LLOYDSME","LODHA",
    "LT","LTIM","LTTS","LUPIN","LUXIND","LXCHEM",
    "M&M","M&MFIN","MAHABANK","MAHLIFE",
    "MAHLOG","MANAPPURAM","MANKIND",
    "MAPMYINDIA","MARICO","MARKSANS",
    "MARUTI","MASTEK","MAXHEALTH","MAZDOCK","MCX",
    "METROBRAND","METROPOLIS","MFSL","MGL",
    "MIDHANI","MINDACORP","MOTHERSON",
    "MPHASIS","MRF","MSUMI","MTARTECH","MUTHOOTFIN",
    "NATCOPHARM","NATIONALUM","NAUKRI","NAVINFLUOR","NAVNETEDUL",
    "NCC","NESTLEIND","NETWORK18","NEWGEN","NHPC",
    "NIACL","NLCINDIA","NMDC","NOCIL","NTPC","NUVOCO",
    "OBEROIRLTY","OFSS","OIL","OLECTRA","ONGC","ORIENTELEC",
    "PAGEIND","PAISALO","PATANJALI","PAYTM","PCBL",
    "PERSISTENT","PETRONET","PFC","PFIZER","PGHH",
    "PIDILITIND","PIIND","PNB","PNBHOUSING",
    "POLICYBZR","POLYCAB","POONAWALLA","POWERGRID",
    "POWERINDIA","PPLPHARMA","PRESTIGE","PRINCEPIPE",
    "PRSMJOHNSN","PVRINOX","QUESS",
    "RADICO","RAIN","RAILTEL","RAJESHEXPO","RALLIS",
    "RAMCOCEM","RATNAMANI","RAYMOND","RBA","RBLBANK",
    "RCF","RECLTD","REDINGTON","RELAXO","RELIANCE",
    "RELIGARE","RITES","RKFORGE",
    "ROUTE","RVNL",
    "SAREGAMA","SAIL","SBICARD","SBILIFE",
    "SBIN","SCHAEFFLER","SCHNEIDER","SHILPAMED",
    "SHREECEM","SHRIRAMFIN","SIEMENS",
    "SJVN","SKFINDIA","SOBHA","SOLARINDS","SONACOMS",
    "SONATSOFTW","SOUTHBANK","SPARC","SRF","STAR",
    "STARHEALTH","STEL","STLTECH","SUDARSCHEM","SUMICHEM",
    "SUNDARMFIN","SUNDRMFAST","SUNPHARMA","SUNTV",
    "SUPREMEIND","SUVEN","SUZLON","SWSOLAR",
    "SYNGENE","SYRMA",
    "TANLA","TARSONS","TATACHEM","TATACOMM","TATACONSUM",
    "TATAELXSI","TATAINVEST","TATAPOWER",
    "TATASTEEL","TATATECH","TCS","TECHM",
    "THERMAX","TIINDIA","TIMKEN","TITAN",
    "TORNTPHARM","TORNTPOWER","TRENT","TRIDENT","TRITURBINE",
    "TRIVENI","TTML","TVSMOTOR",
    "UBL","UJJIVANSFB","ULTRACEMCO","UNIONBANK",
    "UNITDSPR","UPL","UTIAMC",
    "VAKRANGEE","VARROC","VBL","VEDL","VENKEYS","VGUARD",
    "VINATIORGA","VIPIND","VOLTAS","VSTIND",
    "WELCORP","WELSPUNLIV","WHIRLPOOL","WIPRO",
    "YESBANK","ZEEL","ZENSARTECH","ZFCVINDIA",
    "ZYDUSLIFE","ZYDUSWELL"
]

ALL_STOCKS = list(set(ALL_STOCKS))

def fetch_stock_data(symbol, period="18mo"):
    try:
        yahoo_sym = symbol.replace("&", "%26")
        ticker = yf.Ticker(f"{yahoo_sym}.NS")
        df = ticker.history(period=period)
        if df.empty or len(df) < 60: return None
        return df
    except: return None

def calculate_all_indicators(df):
    df['EMA9'] = ta.ema(df['Close'], length=9)
    df['EMA21'] = ta.ema(df['Close'], length=21)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['SMA20'] = ta.sma(df['Close'], length=20)
    df['SMA50'] = ta.sma(df['Close'], length=50)
    df['SMA200'] = ta.sma(df['Close'], length=200)
    bb = ta.bbands(df['Close'], length=20)
    if bb is not None:
        bb_cols = bb.columns.tolist()
        df['BB_upper'] = bb[[c for c in bb_cols if 'BBU' in c][0]]
        df['BB_lower'] = bb[[c for c in bb_cols if 'BBL' in c][0]]
        df['BB_mid'] = bb[[c for c in bb_cols if 'BBM' in c][0]]
    else:
        df['BB_upper'] = df['SMA20'] + 2*df['Close'].rolling(20).std()
        df['BB_lower'] = df['SMA20'] - 2*df['Close'].rolling(20).std()
        df['BB_mid'] = df['SMA20']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
    kc = ta.kc(df['High'], df['Low'], df['Close'], length=20)
    if kc is not None:
        kc_cols = kc.columns.tolist()
        df['KC_upper'] = kc[[c for c in kc_cols if 'KCU' in c][0]]
        df['KC_lower'] = kc[[c for c in kc_cols if 'KCL' in c][0]]
    else:
        atr20 = ta.atr(df['High'], df['Low'], df['Close'], length=20)
        df['KC_upper'] = df['EMA21'] + 2*atr20
        df['KC_lower'] = df['EMA21'] - 2*atr20
    df['ZScore'] = (df['Close'] - df['SMA20']) / df['Close'].rolling(20).std()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    df['RSI_ROC'] = df['RSI'].diff(3)
    df['MACD_accel'] = df['MACD_hist'].diff(1)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_pct'] = (df['ATR'] / df['Close']) * 100
    df['ATR_percentile'] = df['ATR'].rolling(100).apply(
        lambda x: percentileofscore(x, x.iloc[-1]), raw=False)
    df['Returns'] = df['Close'].pct_change()
    df['HV20'] = df['Returns'].rolling(20).std() * np.sqrt(252) * 100
    df['Vol_SMA20'] = ta.sma(df['Volume'], length=20)
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Up_Vol'] = np.where(df['Close'] > df['Open'], df['Volume'], 0)
    df['Down_Vol'] = np.where(df['Close'] < df['Open'], df['Volume'], 0)
    df['Vol_Delta'] = df['Up_Vol'].rolling(5).sum() - df['Down_Vol'].rolling(5).sum()
    df['Vol_Delta_Ratio'] = df['Up_Vol'].rolling(5).sum() / (df['Down_Vol'].rolling(5).sum() + 1)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBV_SMA'] = ta.sma(df['OBV'], length=20)
    df['Weekly_EMA9'] = ta.ema(df['Close'], length=45)
    df['Weekly_EMA21'] = ta.ema(df['Close'], length=105)
    df['Monthly_EMA'] = ta.ema(df['Close'], length=200)
    df['High_52w'] = df['High'].rolling(window=min(252, len(df)), min_periods=50).max()
    df['Low_52w'] = df['Low'].rolling(window=min(252, len(df)), min_periods=50).min()
    recent_high = df['High'].iloc[-60:].max()
    recent_low = df['Low'].iloc[-60:].min()
    fib_range = recent_high - recent_low
    df['Fib_236'] = recent_high - fib_range * 0.236
    df['Fib_382'] = recent_high - fib_range * 0.382
    df['Fib_500'] = recent_high - fib_range * 0.500
    df['Fib_618'] = recent_high - fib_range * 0.618
    return df.dropna()

def model_1_mean_reversion(df):
    latest = df.iloc[-1]; prev = df.iloc[-2]
    buy = [latest['ZScore']<-1.0, latest['Close']<=latest['BB_lower']*1.02,
           latest['Close']<latest['KC_lower']*1.03, latest['RSI']<40, latest['Close']>prev['Close']]
    sell = [latest['ZScore']>1.0, latest['Close']>=latest['BB_upper']*0.98,
            latest['Close']>latest['KC_upper']*0.97, latest['RSI']>60, latest['Close']<prev['Close']]
    bs, ss = sum(buy), sum(sell)
    if bs >= 3: return bs, 5, "BUY", "Mean reversion from oversold"
    if ss >= 3: return ss, 5, "SELL", "Mean reversion from overbought"
    return max(bs,ss), 5, "NEUTRAL", ""

def model_2_momentum_decay(df):
    latest = df.iloc[-1]; prev = df.iloc[-2]
    buy = [latest['MACD_hist']>prev['MACD_hist'], latest['MACD_accel']>0,
           latest['RSI']>prev['RSI'], latest['RSI']>40 and latest['RSI']<65, latest['Close']>prev['Close']]
    sell = [latest['MACD_hist']<prev['MACD_hist'], latest['MACD_accel']<0,
            latest['RSI']<prev['RSI'], latest['RSI']>35 and latest['RSI']<60, latest['Close']<prev['Close']]
    bs, ss = sum(buy), sum(sell)
    if bs >= 3: return bs, 5, "BUY", "Momentum accelerating up"
    if ss >= 3: return ss, 5, "SELL", "Momentum decelerating down"
    return max(bs,ss), 5, "NEUTRAL", ""

def model_3_volatility_regime(df):
    latest = df.iloc[-1]; prev = df.iloc[-2]
    atr_pctl = latest['ATR_percentile']
    if atr_pctl < 30: regime = "LOW_VOL"
    elif atr_pctl < 70: regime = "NORMAL_VOL"
    else: regime = "HIGH_VOL"
    if regime == "LOW_VOL":
        buy = [latest['BB_width']<df['BB_width'].quantile(0.3), latest['Close']>latest['BB_mid'],
               latest['Volume']>latest['Vol_SMA20']*1.1, latest['MACD_hist']>0, latest['Close']>latest['EMA21']]
        sell = [latest['BB_width']<df['BB_width'].quantile(0.3), latest['Close']<latest['BB_mid'],
                latest['Volume']>latest['Vol_SMA20']*1.1, latest['MACD_hist']<0, latest['Close']<latest['EMA21']]
    elif regime == "NORMAL_VOL":
        buy = [latest['Close']>latest['EMA21'], latest['EMA21']>latest['EMA50'],
               latest['RSI']>45 and latest['RSI']<75, latest['MACD']>latest['MACD_signal'], latest['Volume']>latest['Vol_SMA20']*0.8]
        sell = [latest['Close']<latest['EMA21'], latest['EMA21']<latest['EMA50'],
                latest['RSI']<55 and latest['RSI']>25, latest['MACD']<latest['MACD_signal'], latest['Volume']>latest['Vol_SMA20']*0.8]
    else:
        buy = [latest['ZScore']<-1.0, latest['RSI']<35, latest['Close']<latest['BB_lower']*1.02,
               latest['Close']>prev['Close'], latest['ATR_pct']>1.5]
        sell = [latest['ZScore']>1.0, latest['RSI']>65, latest['Close']>latest['BB_upper']*0.98,
                latest['Close']<prev['Close'], latest['ATR_pct']>1.5]
    bs, ss = sum(buy), sum(sell)
    if bs >= 3: return bs, 5, "BUY", f"Regime: {regime}"
    if ss >= 3: return ss, 5, "SELL", f"Regime: {regime}"
    return max(bs,ss), 5, "NEUTRAL", regime

def model_4_order_flow(df):
    latest = df.iloc[-1]
    buy = [latest['Vol_Delta_Ratio']>1.3, latest['OBV']>latest['OBV_SMA'],
           latest['Close']>latest['VWAP'], latest['Volume']>latest['Vol_SMA20']*0.9, latest['Close']>latest['Open']]
    sell = [latest['Vol_Delta_Ratio']<0.75, latest['OBV']<latest['OBV_SMA'],
            latest['Close']<latest['VWAP'], latest['Volume']>latest['Vol_SMA20']*0.9, latest['Close']<latest['Open']]
    bs, ss = sum(buy), sum(sell)
    if bs >= 3: return bs, 5, "BUY", "Accumulation detected"
    if ss >= 3: return ss, 5, "SELL", "Distribution detected"
    return max(bs,ss), 5, "NEUTRAL", ""

def model_5_multi_timeframe(df):
    latest = df.iloc[-1]
    near_fib_buy = (abs(latest['Close']-latest['Fib_382'])/latest['Close']<0.03) or \
                   (abs(latest['Close']-latest['Fib_500'])/latest['Close']<0.03) or \
                   (abs(latest['Close']-latest['Fib_618'])/latest['Close']<0.03)
    near_fib_sell = (abs(latest['Close']-latest['Fib_382'])/latest['Close']<0.03) or \
                    (abs(latest['Close']-latest['Fib_500'])/latest['Close']<0.03) or \
                    (abs(latest['Close']-latest['Fib_236'])/latest['Close']<0.03)
    buy = [latest['EMA9']>latest['EMA21'], latest['Weekly_EMA9']>latest['Weekly_EMA21'],
           latest['Close']>latest['Monthly_EMA'], near_fib_buy, latest['RSI']>35 and latest['RSI']<70]
    sell = [latest['EMA9']<latest['EMA21'], latest['Weekly_EMA9']<latest['Weekly_EMA21'],
            latest['Close']<latest['Monthly_EMA'], near_fib_sell, latest['RSI']>30 and latest['RSI']<65]
    bs, ss = sum(buy), sum(sell)
    if bs >= 3: return bs, 5, "BUY", "Multi-timeframe bullish"
    if ss >= 3: return ss, 5, "SELL", "Multi-timeframe bearish"
    return max(bs,ss), 5, "NEUTRAL", ""

def calculate_risk(df, direction):
    latest = df.iloc[-1]
    entry_price = latest['Close']
    atr = latest['ATR']
    sl_distance = atr * 1.5
    sl_pct = (sl_distance / entry_price) * 100
    sl_pct = max(1.5, min(5.0, sl_pct))
    sl_distance = entry_price * (sl_pct / 100)
    volatility_pct = (atr / entry_price) * 100
    recent_high = df['High'].iloc[-20:].max()
    recent_low = df['Low'].iloc[-20:].min()
    swing_pct = ((recent_high - recent_low) / entry_price) * 100
    if volatility_pct > 3 and swing_pct > 15: rr = 3.0
    elif volatility_pct > 2 and swing_pct > 10: rr = 2.5
    elif volatility_pct > 1.5 or swing_pct > 8: rr = 2.0
    else: rr = 1.5
    target_distance = sl_distance * rr
    if direction == "BUY":
        sl = round(entry_price - sl_distance, 2)
        target = round(entry_price + target_distance, 2)
        target = max(target, round(entry_price * 1.02, 2))
    else:
        sl = round(entry_price + sl_distance, 2)
        target = round(entry_price - target_distance, 2)
        target = min(target, round(entry_price * 0.98, 2))
    actual_risk = abs(entry_price - sl)
    actual_reward = abs(target - entry_price)
    actual_rr = round(actual_reward / actual_risk, 1) if actual_risk > 0 else 0
    return {'entry':round(entry_price,2),'stop_loss':sl,'target':target,'rr_ratio':actual_rr,'sl_pct':round(sl_pct,2),'atr':round(atr,2)}

def backtest_quantum(df, direction):
    wins=0;losses=0;profits=[];loss_amounts=[];in_trade=False;trade_exit_idx=0
    if len(df)<100:
        return {'win_rate':50,'total_trades':0,'wins':0,'losses':0,'avg_profit_pct':0,'avg_loss_pct':0,'expectancy':0}
    check_points = range(60, len(df)-10, 5)
    for i in check_points:
        if in_trade and i<=trade_exit_idx: continue
        in_trade=False
        curr=df.iloc[i]
        if direction=="BUY" and curr['Close']<curr['EMA21'] and curr['RSI']<35: continue
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
        dirs=[d1,d2,d3,d4,d5]
        signal=None
        if direction=="BUY" and dirs.count("BUY")>=3: signal="BUY"
        elif direction=="SELL" and dirs.count("SELL")>=3: signal="SELL"
        if signal is None: continue
        entry_price=df.iloc[i]['Close']
        atr=df.iloc[i]['ATR']
        sl_dist=entry_price*(max(1.5,min(5.0,(atr*1.5/entry_price)*100))/100)
        target_dist=sl_dist*2.0
        if signal=="BUY": sl=entry_price-sl_dist;target=entry_price+target_dist
        else: sl=entry_price+sl_dist;target=entry_price-target_dist
        max_days=min(30,len(df)-i-1)
        for j in range(1,max_days+1):
            future=df.iloc[i+j]
            if signal=="BUY":
                if future['Low']<=sl: losses+=1;loss_amounts.append((sl_dist/entry_price)*100);trade_exit_idx=i+j;in_trade=True;break
                if future['High']>=target: wins+=1;profits.append((target_dist/entry_price)*100);trade_exit_idx=i+j;in_trade=True;break
            else:
                if future['High']>=sl: losses+=1;loss_amounts.append((sl_dist/entry_price)*100);trade_exit_idx=i+j;in_trade=True;break
                if future['Low']<=target: wins+=1;profits.append((target_dist/entry_price)*100);trade_exit_idx=i+j;in_trade=True;break
        if wins+losses>=15: break
    total=wins+losses
    if total==0:
        return {'win_rate':50,'total_trades':0,'wins':0,'losses':0,'avg_profit_pct':0,'avg_loss_pct':0,'expectancy':0}
    wr=round((wins/total)*100,1)
    ap=round(np.mean(profits),2) if profits else 0
    al=round(np.mean(loss_amounts),2) if loss_amounts else 0
    exp=round((wr/100*ap)-((100-wr)/100*al),2)
    return {'win_rate':wr,'total_trades':total,'wins':wins,'losses':losses,'avg_profit_pct':ap,'avg_loss_pct':al,'expectancy':exp}

def quantum_scan(symbol):
    df = fetch_stock_data(symbol)
    if df is None: return None
    try: df = calculate_all_indicators(df)
    except: return None
    if df.empty or len(df)<80: return None
    try:
        m1_passed,m1_total,m1_dir,m1_reason = model_1_mean_reversion(df)
        m2_passed,m2_total,m2_dir,m2_reason = model_2_momentum_decay(df)
        m3_passed,m3_total,m3_dir,m3_reason = model_3_volatility_regime(df)
        m4_passed,m4_total,m4_dir,m4_reason = model_4_order_flow(df)
        m5_passed,m5_total,m5_dir,m5_reason = model_5_multi_timeframe(df)
    except: return None
    directions = [m1_dir,m2_dir,m3_dir,m4_dir,m5_dir]
    buy_count = directions.count("BUY")
    sell_count = directions.count("SELL")
    if buy_count >= 3: direction="BUY";agreement=buy_count
    elif sell_count >= 3: direction="SELL";agreement=sell_count
    else: return None
    is_fno = symbol in FNO_STOCKS
    if direction=="SELL" and not is_fno: return None
    risk = calculate_risk(df, direction)
    if direction=="BUY" and risk['target']<=risk['entry']: return None
    if direction=="SELL" and risk['target']>=risk['entry']: return None
    if risk['rr_ratio']<1.2: return None
    bt = backtest_quantum(df, direction)
    if bt['total_trades']>=3 and bt['win_rate']<40: return None
    if bt['total_trades']>=3 and bt['expectancy']<0: return None
    total_passed = m1_passed+m2_passed+m3_passed+m4_passed+m5_passed
    total_possible = m1_total+m2_total+m3_total+m4_total+m5_total
    model_pct = (total_passed/total_possible)*100
    agreement_bonus = (agreement-3)*5
    bt_bonus = 0
    if bt['total_trades']>=3:
        if bt['win_rate']>=65 and bt['expectancy']>1.5: bt_bonus=10
        elif bt['win_rate']>=55 and bt['expectancy']>0: bt_bonus=6
        elif bt['win_rate']>=45: bt_bonus=3
    total_score = round(min(model_pct+agreement_bonus+bt_bonus, 100), 1)
    if total_score >= 85 and bt['total_trades'] >= 3 and bt['win_rate'] >= 60 and bt['expectancy'] > 0:
        grade="A+";confidence="HIGH"
    elif total_score >= 80 and bt['total_trades'] >= 3 and bt['win_rate'] >= 50 and bt['expectancy'] > 0:
        grade="A+";confidence="MEDIUM"
    else: return None
    agreeing_models = []
    for name,d in [("MRB",m1_dir),("MDD",m2_dir),("VRC",m3_dir),("OFI",m4_dir),("MTC",m5_dir)]:
        if d==direction: agreeing_models.append(name)
    return {
        'symbol':symbol,'direction':direction,'entry':risk['entry'],
        'stop_loss':risk['stop_loss'],'target':risk['target'],
        'score':total_score,'grade':grade,'rr_ratio':risk['rr_ratio'],
        'is_fno':is_fno,'confidence':confidence,
        'models_agreeing':f"{agreement}/5",'agreeing_models':", ".join(agreeing_models),
        'historical_win_rate':bt['win_rate'],'total_past_trades':bt['total_trades'],
        'past_wins':bt['wins'],'past_losses':bt['losses'],
        'avg_profit_pct':bt['avg_profit_pct'],'avg_loss_pct':bt['avg_loss_pct'],
        'expectancy':bt['expectancy'],
        'model_scores':{
            'Mean_Reversion':f"{m1_passed}/{m1_total} ({m1_dir})",
            'Momentum_Decay':f"{m2_passed}/{m2_total} ({m2_dir})",
            'Volatility_Regime':f"{m3_passed}/{m3_total} ({m3_dir})",
            'Order_Flow':f"{m4_passed}/{m4_total} ({m4_dir})",
            'Multi_Timeframe':f"{m5_passed}/{m5_total} ({m5_dir})",
        },
        'backtest':{
            'Win Rate':f"{bt['win_rate']}%",
            'Trades':f"{bt['wins']}W / {bt['losses']}L",
            'Avg Profit':f"+{bt['avg_profit_pct']}%",
            'Avg Loss':f"-{bt['avg_loss_pct']}%",
            'Expectancy':f"{bt['expectancy']}% per trade"
        }
    }

def run_quantum_scan(max_stocks=None):
    stocks = ALL_STOCKS[:max_stocks] if max_stocks else ALL_STOCKS
    buy_results = []
    sell_results = []
    for symbol in stocks:
        try:
            result = quantum_scan(symbol)
            if result:
                if result['direction'] == "BUY": buy_results.append(result)
                else: sell_results.append(result)
        except: pass
    buy_results.sort(key=lambda x: x['score'], reverse=True)
    sell_results.sort(key=lambda x: x['score'], reverse=True)
    return {'buy': buy_results, 'sell': sell_results}

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

cached_results = None
scan_in_progress = False

@app.route('/')
def home():
    return jsonify({'scanner':'QUANTUM Scanner v2.0','endpoints':{'/api/scan':'Run scan','/api/scan?max=50':'Quick scan','/api/results':'Cached results','/api/health':'Health check'}})

@app.route('/api/health')
def health():
    return jsonify({'status':'QUANTUM SCANNER v2.0 ONLINE','timestamp':datetime.now().isoformat()})

@app.route('/api/scan')
def api_scan():
    global cached_results, scan_in_progress
    if scan_in_progress:
        return jsonify({'status':'busy','message':'Scan in progress'}), 429
    mx = request.args.get('max', default=None, type=int)
    scan_in_progress = True
    try:
        results = run_quantum_scan(mx)
        cached_results = {
            'buy': results['buy'],
            'sell': results['sell'],
            'last_scan': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_stocks_scanned': mx if mx else len(ALL_STOCKS)
        }
        return jsonify(cached_results)
    except Exception as e:
        return jsonify({'status':'error','message':str(e)}), 500
    finally:
        scan_in_progress = False

@app.route('/api/results')
def api_results():
    if cached_results is None:
        return jsonify({'status':'no_data','message':'No scan run yet'}), 404
    return jsonify(cached_results)

application = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
