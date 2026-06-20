import os, json, re, asyncio, threading, math, time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "Asia/Dubai")
CAPITAL_BASE_URL = os.getenv("CAPITAL_BASE_URL", "https://demo-api-capital.backend-capital.com/api/v1")
CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY", "")
CAPITAL_IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER", "")
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD", "")
CAPITAL_SESSION = {"cst": "", "security": "", "ts": 0}
CAPITAL_LOCK = threading.Lock()

app = FastAPI(title="AI MARKETLENS - Capital.com - By Syed Abbas")
executor = ThreadPoolExecutor(max_workers=int(os.getenv("APP_WORKERS", "3")))

HIT_COUNTER_FILE = Path(os.getenv("HIT_COUNTER_FILE", "/home/site/hit_counter.json"))
SIGNAL_TRACKER_FILE = Path(os.getenv("SIGNAL_TRACKER_FILE", "/home/site/signal_tracker.json"))
if not HIT_COUNTER_FILE.parent.exists():
    HIT_COUNTER_FILE = Path("hit_counter.json")
if not SIGNAL_TRACKER_FILE.parent.exists():
    SIGNAL_TRACKER_FILE = Path("signal_tracker.json")

HIT_LOCK = threading.Lock()
TRACKER_LOCK = threading.Lock()
CACHE_LOCK = threading.RLock()
CANDLE_CACHE = {}
NEWS_CACHE = {}
AI_CACHE = {}
RESPONSE_CACHE = {}
CHART_CACHE = {}

CANDLE_TTL_SECONDS = int(os.getenv("CANDLE_TTL_SECONDS", "180"))
NEWS_TTL_SECONDS = int(os.getenv("NEWS_TTL_SECONDS", "300"))
AI_TTL_SECONDS = int(os.getenv("AI_TTL_SECONDS", "600"))
RESPONSE_TTL_SECONDS = int(os.getenv("RESPONSE_TTL_SECONDS", "25"))
CHART_TTL_SECONDS = int(os.getenv("CHART_TTL_SECONDS", "120"))

# ============================================================
# AUTO ORDER / CAPITAL.COM EXECUTION CONFIG
# ============================================================
AUTO_ORDER_RUNTIME = {"enabled": os.getenv("AUTO_ORDER", "false").lower() == "true"}
AUTO_ORDER_LOCK = threading.Lock()
ORDER_STATE_FILE = Path(os.getenv("ORDER_STATE_FILE", "/home/site/order_state.json"))
if not ORDER_STATE_FILE.parent.exists():
    ORDER_STATE_FILE = Path("order_state.json")

AUTO_ORDER_MIN_CONFIDENCE = float(os.getenv("AUTO_ORDER_MIN_CONFIDENCE", "80"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "10"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "2"))
ONE_TRADE_PER_ASSET = os.getenv("ONE_TRADE_PER_ASSET", "true").lower() == "true"
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "1.00"))
ORDER_RISK_PER_TRADE_PCT = float(os.getenv("ORDER_RISK_PER_TRADE_PCT", "0.003"))
ORDER_FALLBACK_SIZE = float(os.getenv("ORDER_FALLBACK_SIZE", "1"))
ORDER_CURRENCY_CODE = os.getenv("ORDER_CURRENCY_CODE", "USD")

# Regression health gate used by both the Regression Pack popup and Auto Order module.
# 30M is intentionally fixed because the production signal engine uses 30M for trade signals.
REGRESSION_PACK_HOURS = int(os.getenv("REGRESSION_PACK_HOURS", "168"))  # 7 days
REGRESSION_MIN_CONFIDENCE = float(os.getenv("REGRESSION_MIN_CONFIDENCE", "80"))
REGRESSION_MIN_SIGNALS = int(os.getenv("REGRESSION_MIN_SIGNALS", "5"))
REGRESSION_MIN_WINS = int(os.getenv("REGRESSION_MIN_WINS", "3"))
# Strict tradable gate: win rate >= 80%, positive Total R, and enough sample size.
# Sample is accepted when either total signals >= 5 OR wins >= 3.
REGRESSION_GREEN_WIN_RATE = float(os.getenv("REGRESSION_GREEN_WIN_RATE", "80"))
# Extra production gate: do not approve weak positive-R assets or low-confidence backtests.
REGRESSION_MIN_TOTAL_R = float(os.getenv("REGRESSION_MIN_TOTAL_R", "1.0"))
REGRESSION_MIN_AVG_CONFIDENCE = float(os.getenv("REGRESSION_MIN_AVG_CONFIDENCE", "85"))
REGRESSION_HEALTH_TTL_SECONDS = int(os.getenv("REGRESSION_HEALTH_TTL_SECONDS", "900"))
REGRESSION_HEALTH_CACHE = {"ts": 0.0, "data": None}


def _safe_copy(value):
    try:
        if isinstance(value, pd.DataFrame):
            return value.copy(deep=False)
        if isinstance(value, (dict, list)):
            return json.loads(json.dumps(value))
        return value.copy() if hasattr(value, "copy") else value
    except Exception:
        return value


def _cache_get(cache, key, ttl):
    now = time.time()
    with CACHE_LOCK:
        item = cache.get(key)
        if not item:
            return None
        ts, value = item
        if now - ts <= ttl:
            return _safe_copy(value)
        cache.pop(key, None)
    return None


def _cache_set(cache, key, value):
    with CACHE_LOCK:
        cache[key] = (time.time(), _safe_copy(value))
        if len(cache) > 200:
            for k, _ in sorted(cache.items(), key=lambda kv: kv[1][0])[:50]:
                cache.pop(k, None)


def _news_fingerprint(news):
    try:
        return "|".join([(n.get("headline", "")[:80] + str(n.get("source", ""))) for n in (news or [])[:8]])
    except Exception:
        return ""


def read_hit_count() -> int:
    try:
        if HIT_COUNTER_FILE.exists():
            return int(json.loads(HIT_COUNTER_FILE.read_text()).get("hits", 0))
    except Exception:
        pass
    return 0


def write_hit_count(count: int) -> None:
    try:
        HIT_COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
        HIT_COUNTER_FILE.write_text(json.dumps({"hits": int(count)}))
    except Exception:
        pass


ASSETS = {
    "GOLD": {"name": "Gold Spot", "icon": "ðŸŸ¡", "epic": "GOLD", "keywords": ["gold", "xau", "bullion", "safe haven", "inflation", "fed", "rates", "dollar"]},
    "SILVER": {"name": "Silver", "icon": "âšª", "epic": "SILVER", "keywords": ["silver", "xag", "precious metal", "industrial metal", "solar", "dollar", "rates"]},
    "WTI": {"name": "Crude Oil WTI", "icon": "ðŸ›¢ï¸", "epic": "OIL_CRUDE", "keywords": ["wti", "crude", "oil", "opec", "iran", "gulf", "hormuz", "sanctions", "us strikes", "middle east"]},
    "BRENT": {"name": "Brent Crude", "icon": "ðŸ›¢ï¸", "epic": "OIL_BRENT", "keywords": ["brent", "crude", "oil", "opec", "iran", "gulf", "hormuz", "sanctions", "shipping"]},
    "BTC": {"name": "Bitcoin", "icon": "â‚¿", "epic": "BTCUSD", "keywords": ["bitcoin", "btc", "crypto", "etf", "risk assets", "liquidity", "fed", "rates"]},
    "USTEC100": {"name": "US Tech 100", "icon": "US100", "epic": "US100", "keywords": ["nasdaq", "nasdaq 100", "tech stocks", "ai stocks", "fed", "rates", "yields"]},
}

INTERVALS = {
    "1M": {"cap": "MINUTE", "max": 420},
    "15M": {"cap": "MINUTE_15", "max": 420},
    "30M": {"cap": "MINUTE_30", "max": 420},
    "1H": {"cap": "HOUR", "max": 420},
    "1D": {"cap": "DAY", "max": 220},
}


def clamp(x, lo=-100, hi=100):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def clean(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()


def _mid_price(price_obj):
    if not isinstance(price_obj, dict):
        return np.nan
    bid = safe_float(price_obj.get("bid"), np.nan)
    ask = safe_float(price_obj.get("ask"), np.nan)
    if not pd.isna(bid) and not pd.isna(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2
    return safe_float(price_obj.get("lastTraded"), np.nan)


def capital_login():
    with CAPITAL_LOCK:
        if CAPITAL_SESSION["cst"] and CAPITAL_SESSION["security"] and time.time() - CAPITAL_SESSION["ts"] < 540:
            return CAPITAL_SESSION["cst"], CAPITAL_SESSION["security"]
        if not CAPITAL_API_KEY or not CAPITAL_IDENTIFIER or not CAPITAL_PASSWORD:
            raise RuntimeError("Capital.com credentials missing. Set CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_PASSWORD.")
        headers = {"X-CAP-API-KEY": CAPITAL_API_KEY, "Content-Type": "application/json"}
        payload = {"identifier": CAPITAL_IDENTIFIER, "password": CAPITAL_PASSWORD, "encryptedPassword": False}
        with httpx.Client(timeout=15) as client:
            r = client.post(f"{CAPITAL_BASE_URL.rstrip('/')}/session", headers=headers, json=payload)
        r.raise_for_status()
        cst = r.headers.get("CST")
        security = r.headers.get("X-SECURITY-TOKEN")
        if not cst or not security:
            raise RuntimeError("Capital login succeeded but tokens were not returned.")
        CAPITAL_SESSION.update({"cst": cst, "security": security, "ts": time.time()})
        return cst, security


def get_capital_market_snapshot(epic):
    """Return live price and daily/24h change directly from Capital.com market snapshot.

    Important: the dashboard must not calculate 24H change from intraday candles,
    because CFD daily change on Capital.com is based on the platform snapshot fields.
    This keeps Crude Oil WTI aligned with Capital.com, e.g. ~1% instead of 2–4%.
    """
    cst, security = capital_login()
    headers = {"X-CAP-API-KEY": CAPITAL_API_KEY, "CST": cst, "X-SECURITY-TOKEN": security}
    url = f"{CAPITAL_BASE_URL.rstrip('/')}/markets/{epic}"
    with httpx.Client(timeout=10) as client:
        r = client.get(url, headers=headers)
        if r.status_code in (401, 403):
            CAPITAL_SESSION.update({"cst": "", "security": "", "ts": 0})
            cst, security = capital_login()
            headers.update({"CST": cst, "X-SECURITY-TOKEN": security})
            r = client.get(url, headers=headers)
        r.raise_for_status()
        payload = r.json()

    snapshot = payload.get("snapshot", {}) or {}
    bid = safe_float(snapshot.get("bid"), 0)
    offer = safe_float(snapshot.get("offer"), 0)
    price = round((bid + offer) / 2, 3) if bid > 0 and offer > 0 else round(offer or bid or 0, 3)

    # Capital.com may return the percentage field with different names depending on account/API version.
    pct_keys = [
        "percentageChange", "percentChange", "percentage_change", "changePercentage",
        "changePct", "changePercent", "dailyChangePercentage", "dailyChangePct",
    ]
    val_keys = ["netChange", "change", "dailyChange", "priceChange", "changeValue"]

    pct = None
    for k in pct_keys:
        if k in snapshot and snapshot.get(k) is not None:
            pct = safe_float(snapshot.get(k), None)
            break

    value = None
    for k in val_keys:
        if k in snapshot and snapshot.get(k) is not None:
            value = safe_float(snapshot.get(k), None)
            break

    # If API gives only netChange, derive percent from previous price.
    # Formula: pct = netChange / (current - netChange) * 100
    if pct is None and value is not None and price and (price - value) > 0:
        pct = value / (price - value) * 100

    # If API gives percent but no value, derive approximate point change.
    if value is None and pct is not None and price:
        value = price - (price / (1 + pct / 100))

    return {
        "price": round(price, 3),
        "change": {
            "value": round(safe_float(value, 0), 3),
            "percent": round(safe_float(pct, 0), 3),
        },
        "snapshot": snapshot,
    }


def get_capital_live_price(epic):
    return get_capital_market_snapshot(epic).get("price", 0)


def clean_market_candles(df, tf="30M"):
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["time", "open", "high", "low", "close"])
    d = d[(d["open"] > 0) & (d["high"] > 0) & (d["low"] > 0) & (d["close"] > 0)]
    d = d.sort_values("time").drop_duplicates("time")
    if str(tf).upper() in {"1M", "15M", "30M", "1H"}:
        d = d[d["time"].dt.weekday < 5]
    if d.empty:
        return pd.DataFrame()
    d["high"] = d[["open", "high", "low", "close"]].max(axis=1)
    d["low"] = d[["open", "high", "low", "close"]].min(axis=1)
    if len(d) >= 40:
        rng = (d["high"] - d["low"]).abs()
        body = (d["close"] - d["open"]).abs()
        ref = rng.rolling(30, min_periods=10).median().replace(0, np.nan)
        bad = ((rng > ref * 8) & (rng > d["close"] * 0.006)) | ((body > ref * 10) & (body > d["close"] * 0.008))
        if bad.fillna(False).any() and bad.mean() < 0.08:
            d = d.loc[~bad.fillna(False)].copy()
    return d.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def compute_daily_change(df, market_snapshot=None):
    """Return dashboard change. Prefer Capital.com snapshot change over candles."""
    try:
        if isinstance(market_snapshot, dict):
            ch = market_snapshot.get("change") or {}
            if ch and (ch.get("percent") is not None):
                return {"value": round(safe_float(ch.get("value"), 0), 3), "percent": round(safe_float(ch.get("percent"), 0), 3)}

        # Fallback only if Capital.com snapshot change is unavailable.
        d = df.copy().dropna(subset=["time", "close"])
        if len(d) < 2:
            return {"value": 0.0, "percent": 0.0}
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
        d = d.sort_values("time")
        last = safe_float(d["close"].iloc[-1])
        prior = safe_float(d["close"].iloc[-2])
        value = last - prior if prior > 0 else 0
        pct = value / prior * 100 if prior > 0 else 0
        return {"value": round(value, 3), "percent": round(pct, 3)}
    except Exception:
        return {"value": 0.0, "percent": 0.0}


def _load_candles_uncached(asset_key, tf):
    epic = ASSETS[asset_key]["epic"]
    cfg = INTERVALS[tf]
    cst, security = capital_login()
    headers = {"X-CAP-API-KEY": CAPITAL_API_KEY, "CST": cst, "X-SECURITY-TOKEN": security}
    params = {"resolution": cfg["cap"], "max": cfg["max"]}
    try:
        with httpx.Client(timeout=12) as client:
            r = client.get(f"{CAPITAL_BASE_URL.rstrip('/')}/prices/{epic}", headers=headers, params=params)
            if r.status_code in (401, 403):
                CAPITAL_SESSION.update({"cst": "", "security": "", "ts": 0})
                cst, security = capital_login()
                headers.update({"CST": cst, "X-SECURITY-TOKEN": security})
                r = client.get(f"{CAPITAL_BASE_URL.rstrip('/')}/prices/{epic}", headers=headers, params=params)
            r.raise_for_status()
            prices = r.json().get("prices", [])
    except Exception as e:
        raise RuntimeError(f"Capital.com candle fetch failed for {epic}: {str(e)[:160]}")
    rows = []
    for p in prices:
        rows.append({
            "time": pd.to_datetime(p.get("snapshotTimeUTC") or p.get("snapshotTime"), errors="coerce", utc=True),
            "open": _mid_price(p.get("openPrice")),
            "high": _mid_price(p.get("highPrice")),
            "low": _mid_price(p.get("lowPrice")),
            "close": _mid_price(p.get("closePrice")),
            "volume": safe_float(p.get("lastTradedVolume"), 0),
        })
    df = clean_market_candles(pd.DataFrame(rows), tf)
    if len(df) < 30 or df["close"].nunique() <= 2:
        return pd.DataFrame()
    df = df.tail(420).copy()
    live = get_capital_live_price(epic)
    if live > 0:
        df.loc[df.index[-1], "close"] = live
        df.loc[df.index[-1], "high"] = max(df.loc[df.index[-1], "high"], live)
        df.loc[df.index[-1], "low"] = min(df.loc[df.index[-1], "low"], live)
    return df


def load_candles(asset_key, tf):
    key = (asset_key.upper(), tf.upper())
    cached = _cache_get(CANDLE_CACHE, key, CANDLE_TTL_SECONDS)
    if cached is not None:
        return cached
    df = _load_candles_uncached(asset_key.upper(), tf.upper())
    if df is not None and not df.empty:
        _cache_set(CANDLE_CACHE, key, df)
    return df


def add_indicators(df):
    d = df.copy().sort_values("time")
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()
    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1 / 14, adjust=False).mean() / loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    d["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - d["close"].shift()).abs()
    tr3 = (d["low"] - d["close"].shift()).abs()
    d["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr"] = d["tr"].ewm(alpha=1 / 14, adjust=False).mean().fillna(d["tr"].mean())
    d["atr_pct"] = (d["atr"] / d["close"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    up_move = d["high"].diff()
    down_move = -d["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = d["atr"].replace(0, np.nan)
    d["plus_di"] = (100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr).fillna(0)
    d["minus_di"] = (100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr).fillna(0)
    dx = ((d["plus_di"] - d["minus_di"]).abs() / (d["plus_di"] + d["minus_di"]).replace(0, np.nan)) * 100
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(15)
    d["vol_ma"] = d["volume"].rolling(20).mean().fillna(d["volume"].mean())
    d["ret"] = d["close"].pct_change().fillna(0)
    return d


def support_resistance(df, lookback=80):
    d = df.tail(lookback).copy()
    price = safe_float(d["close"].iloc[-1])
    supports, resistances = [], []
    for i in range(2, len(d) - 2):
        lo = d["low"].iloc[i]
        hi = d["high"].iloc[i]
        if lo <= d["low"].iloc[i - 2:i + 3].min():
            supports.append(float(lo))
        if hi >= d["high"].iloc[i - 2:i + 3].max():
            resistances.append(float(hi))
    sup = max([x for x in supports if x < price], default=float(d["low"].min()))
    res = min([x for x in resistances if x > price], default=float(d["high"].max()))
    return {"support": round(sup, 3), "resistance": round(res, 3)}


def volume_profile(df, bins=24):
    d = df.tail(120).copy()
    typical = (d["high"] + d["low"] + d["close"]) / 3
    vol = d["volume"].replace(0, np.nan)
    if vol.isna().all() or vol.sum(skipna=True) <= 0:
        vol = pd.Series(np.ones(len(d)), index=d.index)
    try:
        grouped = vol.groupby(pd.cut(typical, bins=bins, duplicates="drop"), observed=False).sum()
        if grouped.empty:
            return {"poc": safe_float(d["close"].iloc[-1]), "bias": 0}
        interval = grouped.idxmax()
        poc = float((interval.left + interval.right) / 2)
        price = safe_float(d["close"].iloc[-1])
        atr = safe_float((d["high"] - d["low"]).tail(14).mean(), price * 0.005)
        bias = clamp((price - poc) / max(atr, price * 0.001) * 10, -15, 15)
        return {"poc": round(poc, 3), "bias": round(bias, 3)}
    except Exception:
        return {"poc": safe_float(d["close"].iloc[-1]), "bias": 0}


def market_regime(df):
    d = add_indicators(df)
    last = d.iloc[-1]
    adx = safe_float(last.adx, 15)
    atr_pct = safe_float(last.atr_pct, 0)
    ema_spread = abs(safe_float(last.ema21 - last.ema50)) / max(safe_float(last.close), 1) * 100
    if adx >= 25 and ema_spread > atr_pct * 0.20:
        return {"regime": "TRENDING", "adx": round(adx, 2), "atr_pct": round(atr_pct, 3), "multiplier": 1.12}
    if atr_pct > d["atr_pct"].tail(120).quantile(0.75):
        return {"regime": "HIGH VOLATILITY", "adx": round(adx, 2), "atr_pct": round(atr_pct, 3), "multiplier": 0.88}
    if adx < 17:
        return {"regime": "RANGING", "adx": round(adx, 2), "atr_pct": round(atr_pct, 3), "multiplier": 0.78}
    return {"regime": "NORMAL", "adx": round(adx, 2), "atr_pct": round(atr_pct, 3), "multiplier": 1.0}


def raw_technical_model(df):
    d = add_indicators(df)
    if len(d) < 60:
        return 0.0
    last = d.iloc[-1]
    prev = d.iloc[-4] if len(d) > 4 else d.iloc[-2]
    score = 0.0
    score += 18 if last.ema9 > last.ema21 else -18
    score += 16 if last.ema21 > last.ema50 else -16
    score += 16 if last.close > last.ema200 else -16
    score += 10 if last.close > last.ema21 else -10
    score += 14 if last.macd > last.macd_signal else -14
    score += clamp(last.macd_hist / max(last.atr, last.close * 0.001) * 25, -12, 12)
    mom10 = (last.close - d["close"].iloc[-10]) / max(d["close"].iloc[-10], 1) * 100
    mom20 = (last.close - d["close"].iloc[-20]) / max(d["close"].iloc[-20], 1) * 100
    score += clamp(mom10 * 18, -12, 12) + clamp(mom20 * 10, -10, 10)
    rsi = safe_float(last.rsi, 50)
    if 52 <= rsi <= 68:
        score += 10
    elif 32 <= rsi <= 48:
        score -= 10
    elif rsi > 75:
        score -= 8
    elif rsi < 25:
        score += 8
    if last.adx >= 20:
        score += 8 if last.plus_di > last.minus_di else -8
    recent_high = d["high"].iloc[-31:-1].max()
    recent_low = d["low"].iloc[-31:-1].min()
    if last.close > recent_high:
        score += 10
    elif last.close < recent_low:
        score -= 10
    if score > 0 and last.close < prev.close and last.macd_hist < prev.macd_hist:
        score -= 8
    if score < 0 and last.close > prev.close and last.macd_hist > prev.macd_hist:
        score += 8
    return clamp(score)


def backtest_technical(df, horizon=3):
    d = add_indicators(df).tail(240).reset_index(drop=True)
    if len(d) < 90:
        return {"win_rate": 0.50, "trades": 0, "expectancy": 0.0, "score_adj": 0}
    trend = np.where(d["ema9"] > d["ema21"], 1, -1) + np.where(d["ema21"] > d["ema50"], 1, -1) + np.where(d["close"] > d["ema200"], 1, -1)
    momentum = np.where(d["macd"] > d["macd_signal"], 1, -1)
    rsi_ok = np.where(d["rsi"] >= 52, 1, np.where(d["rsi"] <= 48, -1, 0))
    proxy = trend * 14 + momentum * 14 + rsi_ok * 8
    entries = np.where(np.abs(proxy) >= 24)[0]
    entries = entries[(entries >= 70) & (entries < len(d) - horizon)]
    if len(entries) == 0:
        return {"win_rate": 0.50, "trades": 0, "expectancy": 0.0, "score_adj": 0}
    close = d["close"].to_numpy(float)
    atr = d["atr"].replace(0, np.nan).fillna(d["close"] * 0.005).to_numpy(float)
    direction = np.where(proxy[entries] > 0, 1, -1)
    r_mult = ((close[entries + horizon] - close[entries]) * direction) / np.maximum(atr[entries], close[entries] * 0.0005)
    win_rate = float((r_mult > 0).mean())
    expectancy = float(np.mean(r_mult))
    score_adj = clamp((win_rate - 0.50) * 50 + expectancy * 8, -12, 12)
    return {"win_rate": round(win_rate, 3), "trades": int(len(r_mult)), "expectancy": round(expectancy, 3), "score_adj": round(score_adj, 2)}


def multi_timeframe_confirmation(dfs):
    scores = {tf: raw_technical_model(df) for tf, df in dfs.items() if df is not None and not df.empty}
    base = scores.get("30M", 0)
    if not scores:
        return {"score": 0, "alignment": 0, "scores": {}}
    base_sign = 1 if base > 0 else -1 if base < 0 else 0
    signs = []
    for tf in ["15M", "30M", "1H", "1D"]:
        s = scores.get(tf)
        if s is not None:
            signs.append(1 if s > 12 else -1 if s < -12 else 0)
    aligned = sum(1 for s in signs if s == base_sign and s != 0)
    opposed = sum(1 for s in signs if s == -base_sign and s != 0)
    alignment = aligned - opposed
    return {"score": round(clamp(alignment * 7, -18, 18), 2), "alignment": alignment, "scores": {k: round(v, 2) for k, v in scores.items()}}


def economic_event_filter(news, asset_key):
    text = " ".join([(n.get("headline", "") + " " + n.get("summary", "")) for n in news]).lower()
    words = ["fed", "fomc", "powell", "cpi", "inflation", "jobs report", "nonfarm", "nfp", "pce", "rate decision", "ecb", "opec", "eia", "inventory", "war", "strike", "hormuz", "sanction"]
    hits = [w for w in words if w in text]
    penalty = min(12, len(hits) * 2)
    if asset_key in ["WTI", "BRENT"] and any(w in text for w in ["opec", "eia", "inventory", "hormuz", "iran"]):
        penalty = max(penalty, 6)
    if asset_key in ["GOLD", "SILVER", "USTEC100", "BTC"] and any(w in text for w in ["fed", "cpi", "inflation", "powell", "rate"]):
        penalty = max(penalty, 6)
    return {"hits": hits[:6], "risk_penalty": penalty}


def optimized_levels(df, direction):
    d = add_indicators(df)
    last = d.iloc[-1]
    price = safe_float(last.close)
    atr = max(safe_float(last.atr, price * 0.006), price * 0.001)
    sr = support_resistance(d)
    vp = volume_profile(d)
    if direction >= 0:
        raw_stop = price - atr * 1.35
        stop = min(raw_stop, sr["support"] - atr * 0.15) if sr["support"] < price else raw_stop
        target = price + max(atr * 2.15, (price - stop) * 1.65)
    else:
        raw_stop = price + atr * 1.35
        stop = max(raw_stop, sr["resistance"] + atr * 0.15) if sr["resistance"] > price else raw_stop
        target = price - max(atr * 2.15, (stop - price) * 1.65)
    return {"entry": round(price, 3), "target": round(target, 3), "stop": round(stop, 3), "support": sr["support"], "resistance": sr["resistance"], "poc": vp["poc"]}


def technical_score(df, mtf=None):
    d = add_indicators(df)
    last = d.iloc[-1]
    base = raw_technical_model(d)
    regime = market_regime(d)
    sr = support_resistance(d)
    vp = volume_profile(d)
    bt = backtest_technical(d)
    mtf_score = safe_float((mtf or {}).get("score", 0))
    price = safe_float(last.close)
    sr_bias = 8 if price > sr["resistance"] else -8 if price < sr["support"] else 0
    final_score = clamp((base + mtf_score + sr_bias + safe_float(vp.get("bias", 0)) + bt["score_adj"]) * regime["multiplier"])

    ema9 = safe_float(last.ema9)
    ema21 = safe_float(last.ema21)
    ema50 = safe_float(last.ema50)
    adx = safe_float(last.adx, 0)
    rsi = safe_float(last.rsi, 50)
    atr_now = safe_float(last.atr, price * 0.01)
    atr_avg = safe_float(d["atr"].tail(50).mean(), atr_now)
    atr_spike = bool(atr_avg > 0 and atr_now > atr_avg * 2.0)
    trend_direction = "BULLISH" if ema9 > ema21 > ema50 else "BEARISH" if ema9 < ema21 < ema50 else "MIXED"

    levels = optimized_levels(d, 1 if final_score >= 0 else -1)
    return {
        "score": round(final_score, 2), "price": round(price, 3), "rsi": round(rsi, 2),
        "macd": round(safe_float(last.macd), 3), "atr": round(atr_now, 3),
        "atr_avg": round(atr_avg, 3), "atr_spike": atr_spike,
        "ema9": round(ema9, 3), "ema21": round(ema21, 3), "ema50": round(ema50, 3),
        "adx": round(adx, 2), "trend_direction": trend_direction,
        "entry": levels["entry"], "target": levels["target"], "stop": levels["stop"],
        "support": levels["support"], "resistance": levels["resistance"],
        "entry_zone": f"{min(levels['support'], levels['resistance']):.3f} - {max(levels['support'], levels['resistance']):.3f}",
        "poc": levels["poc"], "regime": regime, "backtest": bt, "raw_score": round(base, 2),
    }


async def fetch_finnhub_news(asset_key):
    cached = _cache_get(NEWS_CACHE, asset_key.upper(), NEWS_TTL_SECONDS)
    if cached is not None:
        return cached
    if not FINNHUB_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=6) as client:
            r = await client.get("https://finnhub.io/api/v1/news", params={"category": "general", "token": FINNHUB_API_KEY})
            r.raise_for_status()
            raw = r.json()
    except Exception:
        return []
    keywords = [k.lower() for k in ASSETS[asset_key]["keywords"]]
    geopolitics = ["iran", "hormuz", "gulf", "sanction", "strike", "middle east", "shipping", "war", "ceasefire", "deal"]
    out = []
    for n in raw:
        headline = clean(n.get("headline"))
        summary = clean(n.get("summary"))
        source = clean(n.get("source"))
        text = f"{headline} {summary}".lower()
        relevance = sum(1 for k in keywords if k in text) + sum(1 for k in geopolitics if k in text) * 1.3
        if relevance > 0:
            out.append({"headline": headline, "summary": summary[:180], "source": source, "url": n.get("url", ""), "relevance": relevance})
    final = sorted(out, key=lambda x: x["relevance"], reverse=True)[:10]
    _cache_set(NEWS_CACHE, asset_key.upper(), final)
    return final


def news_lexicon_score(news, asset_key):
    if not news:
        return 0
    text = " ".join([n["headline"] + " " + n["summary"] for n in news]).lower()
    bullish = ["rally", "surge", "gain", "rise", "rebound", "strong demand", "supply cut", "disruption", "shortage", "drawdown", "inventory draw", "sanctions", "strike", "hormuz", "shipping risk", "safe haven", "rate cut", "weak dollar", "breakout", "record high"]
    bearish = ["drop", "fall", "slump", "selloff", "weak demand", "inventory build", "surplus", "oversupply", "recession", "strong dollar", "rate hike", "peace deal", "ceasefire", "supply restored", "risk off"]
    score = sum(8 for w in bullish if w in text) - sum(8 for w in bearish if w in text)
    if asset_key in ["WTI", "BRENT"]:
        if any(x in text for x in ["iran", "hormuz", "gulf", "sanction", "strike", "shipping risk"]):
            score += 20
        if any(x in text for x in ["inventory build", "oversupply", "weak demand"]):
            score -= 20
    return clamp(score)


async def openai_sentiment(asset_key, news, tech, institutional_context=None):
    key = (asset_key.upper(), round(safe_float(tech.get("score", 0)), 1), round(safe_float(tech.get("price", 0)), 1), round(safe_float(tech.get("rsi", 50)), 1), _news_fingerprint(news))
    cached = _cache_get(AI_CACHE, key, AI_TTL_SECONDS)
    if cached is not None:
        return cached
    if not OPENAI_API_KEY:
        return {"score": 0, "bias": "NEUTRAL", "summary": "OpenAI key missing. Using technical and news fallback only.", "risk": "AI sentiment unavailable."}
    headlines = [f"{n['source']}: {n['headline']} - {n['summary']}" for n in news[:8]]
    ctx = institutional_context or {}
    prompt = f"""
Asset: {ASSETS[asset_key]['name']}
30M Technical Score={tech['score']}
Price={tech['price']} RSI={tech['rsi']} MACD={tech['macd']} ATR={tech['atr']}
Regime={tech.get('regime', {}).get('regime')} Backtest={tech.get('backtest')}
Support={tech.get('support')} Resistance={tech.get('resistance')} VolumePOC={tech.get('poc')}
MTF={ctx.get('mtf')} EventRisk={ctx.get('event_risk')}
News:
{chr(10).join(headlines)}
Return only JSON:
{{"score": number between -100 and 100,"bias": "BULLISH" or "BEARISH" or "NEUTRAL","summary": "short reason","risk": "short risk"}}
"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Return valid JSON only. Be conservative and do not overrule strong 30M technical evidence without clear news catalyst."}, {"role": "user", "content": prompt}],
            temperature=0,
        )
        text = re.sub(r"^```json|```$", "", response.choices[0].message.content.strip(), flags=re.I).strip()
        data = json.loads(text)
        result = {"score": clamp(data.get("score", 0)), "bias": data.get("bias", "NEUTRAL"), "summary": data.get("summary", ""), "risk": data.get("risk", "")}
        _cache_set(AI_CACHE, key, result)
        return result
    except Exception as e:
        return {"score": 0, "bias": "NEUTRAL", "summary": "OpenAI sentiment failed. Fallback active.", "risk": str(e)[:100]}


def read_tracker():
    try:
        if SIGNAL_TRACKER_FILE.exists():
            return json.loads(SIGNAL_TRACKER_FILE.read_text())
    except Exception:
        pass
    return {"signals": []}


def write_tracker(data):
    try:
        SIGNAL_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
        SIGNAL_TRACKER_FILE.write_text(json.dumps(data)[-250000:])
    except Exception:
        pass


def update_signal_tracker(asset_key, signal, price, confidence):
    if signal not in ["BUY SIGNAL", "SELL SIGNAL"]:
        return {"paper_trades": 0, "recent_accuracy": 0.5}
    with TRACKER_LOCK:
        data = read_tracker()
        rows = data.get("signals", [])[-299:]
        direction = 1 if signal == "BUY SIGNAL" else -1
        rows.append({"asset": asset_key, "time": datetime.now(timezone.utc).isoformat(), "signal": signal, "direction": direction, "price": price, "confidence": confidence})
        same = [r for r in rows if r.get("asset") == asset_key]
        closed = []
        if len(same) >= 2:
            for r in same[:-1][-80:]:
                move = (price - safe_float(r.get("price"))) * int(r.get("direction", 1))
                closed.append(1 if move > 0 else 0)
        acc = float(np.mean(closed)) if closed else 0.5
        data["signals"] = rows
        write_tracker(data)
        return {"paper_trades": len(closed), "recent_accuracy": round(acc, 3)}


def probability_from_inputs(fusion, tech, backtest, event_risk):
    bt_wr = safe_float(backtest.get("win_rate", 0.5), 0.5)
    bt_trades = int(backtest.get("trades", 0) or 0)
    bt_weight = min(1.0, bt_trades / 40.0)
    base = 1 / (1 + math.exp(-abs(fusion) / 24.0))
    prob = (base * 0.72) + ((bt_wr * bt_weight + 0.5 * (1 - bt_weight)) * 0.28)
    prob -= safe_float(event_risk.get("risk_penalty", 0)) / 250.0
    return round(max(0.45, min(0.86, prob)), 3)


def signal_quality_filter(side, tech):
    """Final WTI-quality guard used by live signals and regression.

    It removes the main weakness found in regression: early BUY entries against a
    bearish trend, low-ADX chop, RSI/trend conflict, and ATR spike conditions.
    """
    side = str(side or "").upper()
    reasons = []
    ema9 = safe_float(tech.get("ema9"), 0)
    ema21 = safe_float(tech.get("ema21"), 0)
    ema50 = safe_float(tech.get("ema50"), 0)
    rsi = safe_float(tech.get("rsi"), 50)
    adx = safe_float(tech.get("adx"), safe_float((tech.get("regime") or {}).get("adx"), 0))
    atr_spike = bool(tech.get("atr_spike", False))

    if adx < 20:
        reasons.append("ADX below 20 - low trend strength")
    if atr_spike:
        reasons.append("ATR spike above 2x average - volatility risk")

    if side == "BUY":
        if not (ema9 > ema21 > ema50):
            reasons.append("BUY rejected because EMA9 > EMA21 > EMA50 is not aligned")
        if rsi < 50:
            reasons.append("BUY rejected because RSI is below 50")
    elif side == "SELL":
        if not (ema9 < ema21 < ema50):
            reasons.append("SELL rejected because EMA9 < EMA21 < EMA50 is not aligned")
        if rsi > 50:
            reasons.append("SELL rejected because RSI is above 50")

    return {"allowed": len(reasons) == 0, "reasons": reasons}


def fusion_signal(tech, news_score, ai, mtf=None, event_risk=None, tracker=None):
    tech_s, news_s, ai_s = clamp(tech["score"]), clamp(news_score), clamp(ai.get("score", 0))
    mtf_s = clamp((mtf or {}).get("score", 0))
    event_penalty = safe_float((event_risk or {}).get("risk_penalty", 0))
    active_news = abs(news_s) >= 8
    active_ai = abs(ai_s) >= 8
    if active_news and active_ai:
        fusion = tech_s * 0.50 + news_s * 0.18 + ai_s * 0.22 + mtf_s * 0.10
    elif active_ai:
        fusion = tech_s * 0.62 + ai_s * 0.25 + mtf_s * 0.13
    elif active_news:
        fusion = tech_s * 0.64 + news_s * 0.22 + mtf_s * 0.14
    else:
        fusion = tech_s * 0.78 + mtf_s * 0.22
    fusion = fusion - event_penalty * 0.45 if fusion > 0 else fusion + event_penalty * 0.45 if fusion < 0 else fusion
    fusion = clamp(fusion)
    regime_name = tech.get("regime", {}).get("regime", "NORMAL")
    threshold = 18 if regime_name == "TRENDING" else 28 if regime_name == "RANGING" else 30 if regime_name == "HIGH VOLATILITY" else 22
    prob = probability_from_inputs(fusion, tech, tech.get("backtest", {}), event_risk or {})
    if fusion >= threshold and prob >= 0.54:
        signal, label = "BUY SIGNAL", "BULLISH"
    elif fusion <= -threshold and prob >= 0.54:
        signal, label = "SELL SIGNAL", "BEARISH"
    else:
        signal, label = "HOLD / WAIT", "NEUTRAL"

    filter_check = {"allowed": True, "reasons": []}
    if signal == "BUY SIGNAL":
        filter_check = signal_quality_filter("BUY", tech)
    elif signal == "SELL SIGNAL":
        filter_check = signal_quality_filter("SELL", tech)

    if not filter_check.get("allowed", True):
        signal, label = "HOLD / WAIT", "NEUTRAL"
        prob = min(prob, 0.53)

    tracker_acc = safe_float((tracker or {}).get("recent_accuracy", 0.5), 0.5)
    tracker_boost = (tracker_acc - 0.5) * 10 if (tracker or {}).get("paper_trades", 0) >= 5 else 0
    confidence = min(100, max(25, abs(fusion) * 1.28 + prob * 25 + tracker_boost))
    return {"fusion": round(fusion, 2), "confidence": round(confidence, 1), "probability": prob, "signal": signal, "label": label, "tech_percent": round((tech_s + 100) / 2, 1), "news_percent": round((news_s + 100) / 2, 1), "ai_percent": round((ai_s + 100) / 2, 1), "threshold": threshold, "filter": filter_check}


def _to_local_chart_times(series):
    try:
        local_tz = ZoneInfo(LOCAL_TIMEZONE)
    except Exception:
        local_tz = ZoneInfo("Asia/Dubai")
    try:
        times = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(local_tz)
        return times.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    except Exception:
        return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").tolist()


def _chart_tick_arrays(times, max_ticks=4):
    vals = [str(t) for t in (times or []) if str(t) and str(t).lower() != "nat"]
    if not vals:
        return [], []
    step = max(1, int(math.ceil(len(vals) / max_ticks)))
    tickvals = vals[::step]
    if vals[-1] not in tickvals:
        tickvals.append(vals[-1])
    ticktext = []
    for v in tickvals:
        dt = pd.to_datetime(v, errors="coerce")
        ticktext.append(v if pd.isna(dt) else dt.strftime("%d %b %Y<br>%H:%M"))
    return tickvals, ticktext


def build_chart(df, asset_name):
    try:
        chart_key = (asset_name, len(df), str(pd.to_datetime(df["time"].iloc[-1])), round(safe_float(df["close"].iloc[-1]), 3))
        cached = _cache_get(CHART_CACHE, chart_key, CHART_TTL_SECONDS)
        if cached is not None:
            return cached
    except Exception:
        chart_key = None
    d = add_indicators(df).tail(220)
    times = _to_local_chart_times(d["time"])
    tickvals, ticktext = _chart_tick_arrays(times)
    lows, highs = pd.to_numeric(d["low"], errors="coerce").dropna(), pd.to_numeric(d["high"], errors="coerce").dropna()
    y_min, y_max = (0, 1) if lows.empty or highs.empty else (float(lows.tail(200).min()), float(highs.tail(200).max()))
    pad = max((y_max - y_min) * 0.14, abs(y_max) * 0.003, 0.01)
    data = [
        {"type": "candlestick", "x": times, "open": d["open"].round(3).tolist(), "high": d["high"].round(3).tolist(), "low": d["low"].round(3).tolist(), "close": d["close"].round(3).tolist(), "name": asset_name, "increasing": {"line": {"width": 1}}, "decreasing": {"line": {"width": 1}}},
        {"type": "scatter", "mode": "lines", "x": times, "y": d["ema9"].round(3).tolist(), "name": "EMA9", "line": {"width": 1}},
        {"type": "scatter", "mode": "lines", "x": times, "y": d["ema21"].round(3).tolist(), "name": "EMA21", "line": {"width": 1}},
        {"type": "scatter", "mode": "lines", "x": times, "y": d["ema50"].round(3).tolist(), "name": "EMA50", "line": {"width": 1}},
    ]
    layout = {"template": "plotly_dark", "height": 330, "margin": {"l": 58, "r": 34, "t": 22, "b": 118}, "paper_bgcolor": "#111a26", "plot_bgcolor": "#111a26", "font": {"color": "#dce7f3", "size": 10}, "xaxis": {"type": "date", "rangeslider": {"visible": False}, "showgrid": True, "automargin": True, "tickmode": "array", "tickvals": tickvals, "ticktext": ticktext, "tickangle": 0, "tickfont": {"size": 9}, "ticklabelstandoff": 10, "fixedrange": True}, "yaxis": {"range": [y_min - pad, y_max + pad], "fixedrange": True, "automargin": True, "zeroline": False}, "legend": {"orientation": "h", "y": 1.02, "x": 0, "font": {"size": 9}}, "uirevision": "keep"}
    result = {"data": data, "layout": layout}
    if chart_key is not None:
        _cache_set(CHART_CACHE, chart_key, result)
    return result


async def process(asset_key, tf):
    response_key = (asset_key.upper(), tf.upper())
    cached = _cache_get(RESPONSE_CACHE, response_key, RESPONSE_TTL_SECONDS)
    if cached is not None:
        cached["updated"] = datetime.now().strftime("%H:%M:%S")
        return cached
    loop = asyncio.get_running_loop()
    if tf == "30M":
        signal_df = await loop.run_in_executor(executor, load_candles, asset_key, "30M")
        chart_df = signal_df.copy()
        h1_df, d1_df = await asyncio.gather(loop.run_in_executor(executor, load_candles, asset_key, "1H"), loop.run_in_executor(executor, load_candles, asset_key, "1D"))
    else:
        chart_df, signal_df, h1_df, d1_df = await asyncio.gather(loop.run_in_executor(executor, load_candles, asset_key, tf), loop.run_in_executor(executor, load_candles, asset_key, "30M"), loop.run_in_executor(executor, load_candles, asset_key, "1H"), loop.run_in_executor(executor, load_candles, asset_key, "1D"))
    if chart_df.empty and signal_df.empty:
        raise RuntimeError("No candle data returned.")
    if chart_df.empty:
        chart_df = signal_df.copy()
    if signal_df.empty:
        signal_df = chart_df.copy()
    mtf = multi_timeframe_confirmation({"30M": signal_df, "1H": h1_df, "1D": d1_df})
    tech = technical_score(signal_df, mtf=mtf)
    news = await fetch_finnhub_news(asset_key)
    news_score = news_lexicon_score(news, asset_key)
    event_risk = economic_event_filter(news, asset_key)
    ai = await openai_sentiment(asset_key, news, tech, {"mtf": mtf, "event_risk": event_risk})
    fusion_pre = fusion_signal(tech, news_score, ai, mtf=mtf, event_risk=event_risk)
    tracker = update_signal_tracker(asset_key, fusion_pre["signal"], tech["price"], fusion_pre["confidence"])
    fusion = fusion_signal(tech, news_score, ai, mtf=mtf, event_risk=event_risk, tracker=tracker)
    try:
        market_snapshot = await loop.run_in_executor(executor, get_capital_market_snapshot, ASSETS[asset_key]["epic"])
        if safe_float(market_snapshot.get("price"), 0) > 0:
            tech["price"] = market_snapshot["price"]
    except Exception:
        market_snapshot = None
    result = {"asset_key": asset_key, "asset": ASSETS[asset_key], "data_source": "Capital.com", "tf": tf, "signal_tf": "30M", "tech": tech, "news": news, "news_score": round(news_score, 2), "ai": ai, "fusion": fusion, "chart": build_chart(chart_df, ASSETS[asset_key]["name"]), "change": compute_daily_change(signal_df, market_snapshot), "institutional": {"mtf": mtf, "event_risk": event_risk, "tracker": tracker}, "updated": datetime.now().strftime("%H:%M:%S")}
    _cache_set(RESPONSE_CACHE, response_key, result)
    return result



# ============================================================
# CAPITAL.COM ORDER EXECUTION HELPERS
# ============================================================
def _asset_epic(asset_key):
    return ASSETS[str(asset_key).upper()]["epic"]


def _order_state_today():
    return datetime.now().strftime("%Y-%m-%d")


def read_order_state():
    try:
        if ORDER_STATE_FILE.exists():
            data = json.loads(ORDER_STATE_FILE.read_text())
        else:
            data = {}
    except Exception:
        data = {}
    if data.get("date") != _order_state_today():
        data = {"date": _order_state_today(), "orders": []}
    data.setdefault("orders", [])
    return data


def write_order_state(data):
    try:
        ORDER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ORDER_STATE_FILE.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass


def today_order_count():
    state = read_order_state()
    write_order_state(state)
    return len(state.get("orders", []))


def record_order(row):
    with AUTO_ORDER_LOCK:
        state = read_order_state()
        state.setdefault("orders", []).append(row)
        write_order_state(state)


def capital_auth_headers():
    cst, security = capital_login()
    return {
        "X-CAP-API-KEY": CAPITAL_API_KEY,
        "CST": cst,
        "X-SECURITY-TOKEN": security,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def capital_get(path):
    try:
        headers = await asyncio.to_thread(capital_auth_headers)
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{CAPITAL_BASE_URL.rstrip('/')}{path}", headers=headers)
        if r.status_code in (401, 403):
            CAPITAL_SESSION.update({"cst": "", "security": "", "ts": 0})
            headers = await asyncio.to_thread(capital_auth_headers)
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(f"{CAPITAL_BASE_URL.rstrip('/')}{path}", headers=headers)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        return 500, {"error": str(e)}


async def capital_post(path, payload):
    try:
        headers = await asyncio.to_thread(capital_auth_headers)
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(f"{CAPITAL_BASE_URL.rstrip('/')}{path}", headers=headers, json=payload)
        if r.status_code in (401, 403):
            CAPITAL_SESSION.update({"cst": "", "security": "", "ts": 0})
            headers = await asyncio.to_thread(capital_auth_headers)
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(f"{CAPITAL_BASE_URL.rstrip('/')}{path}", headers=headers, json=payload)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        return 500, {"error": str(e)}


async def account_equity():
    code, data = await capital_get("/accounts")
    if code != 200:
        return 0.0
    accounts = data.get("accounts", []) if isinstance(data, dict) else []
    if not accounts:
        return 0.0
    bal = accounts[0].get("balance", {}) or {}
    return safe_float(bal.get("available"), 0) or safe_float(bal.get("balance"), 0) or safe_float(bal.get("deposit"), 0)


async def open_positions():
    code, data = await capital_get("/positions")
    if code != 200:
        return [], code, data
    return (data.get("positions", []) if isinstance(data, dict) else []), code, data


def _position_epic(pos):
    market = pos.get("market", {}) if isinstance(pos, dict) else {}
    position = pos.get("position", {}) if isinstance(pos, dict) else {}
    return market.get("epic") or position.get("epic") or pos.get("epic") or ""


async def count_positions(asset_key):
    positions, code, data = await open_positions()
    if code != 200:
        return 0, 0, f"Unable to read open positions: {str(data)[:120]}"
    epic = _asset_epic(asset_key)
    return len(positions), sum(1 for p in positions if _position_epic(p) == epic), "OK"


async def capital_quote(asset_key):
    epic = _asset_epic(asset_key)
    code, data = await capital_get(f"/markets/{epic}")
    if code != 200:
        return {"price": 0, "bid": 0, "ask": 0, "spread_pct": 999, "status_code": code, "raw": data}
    snap = (data.get("snapshot") or {}) if isinstance(data, dict) else {}
    bid = safe_float(snap.get("bid") or snap.get("bidPrice") or snap.get("sell"), 0)
    ask = safe_float(snap.get("offer") or snap.get("ask") or snap.get("offerPrice") or snap.get("buy"), 0)
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
        return {"price": round(mid, 4), "bid": bid, "ask": ask, "spread_pct": round((ask - bid) / mid * 100, 4), "epic": epic}
    price = safe_float(snap.get("lastTraded"), 0)
    return {"price": price, "bid": bid, "ask": ask, "spread_pct": 999, "epic": epic, "raw": snap}


def _regression_cache_valid():
    return (time.time() - safe_float(REGRESSION_HEALTH_CACHE.get("ts"), 0)) <= REGRESSION_HEALTH_TTL_SECONDS and REGRESSION_HEALTH_CACHE.get("data") is not None


def get_regression_health_pack(force=False):
    """Cached 7-day asset health used as a live Auto Order safety gate."""
    if not force and _regression_cache_valid():
        return _safe_copy(REGRESSION_HEALTH_CACHE["data"])
    data = run_quick_regression_pack("30M", REGRESSION_PACK_HOURS, 3, REGRESSION_MIN_CONFIDENCE)
    REGRESSION_HEALTH_CACHE["ts"] = time.time()
    REGRESSION_HEALTH_CACHE["data"] = _safe_copy(data)
    return data


def get_asset_regression_health(asset_key):
    pack = get_regression_health_pack(force=False)
    key = str(asset_key or "").upper()
    for row in pack.get("assets", []):
        if str(row.get("asset_key", "")).upper() == key:
            return row
    return {"asset_key": key, "health": "RED", "tradable": False, "auto_order_allowed": False, "status": "No regression health found"}


async def order_guard(asset_key, signal_data, execute=True):
    if not execute:
        return False, "Execution disabled"
    if not AUTO_ORDER_RUNTIME["enabled"]:
        return False, "Auto order is OFF"
    if not CAPITAL_API_KEY or not CAPITAL_IDENTIFIER or not CAPITAL_PASSWORD:
        return False, "Capital.com credentials missing"
    if today_order_count() >= MAX_TRADES_PER_DAY:
        return False, f"Daily trade limit reached: {MAX_TRADES_PER_DAY}"
    total, same_asset, pos_msg = await count_positions(asset_key)
    if pos_msg != "OK":
        return False, pos_msg
    if total >= MAX_OPEN_POSITIONS:
        return False, f"Max running positions reached: {total}/{MAX_OPEN_POSITIONS}. New trade blocked."
    if ONE_TRADE_PER_ASSET and same_asset > 0:
        return False, f"Existing open position found for {asset_key}; duplicate trade blocked."
    quote = await capital_quote(asset_key)
    if quote.get("spread_pct", 999) > MAX_SPREAD_PCT:
        return False, f"Spread too wide: {quote.get('spread_pct')}% > {MAX_SPREAD_PCT}%"
    fusion = signal_data.get("fusion", {}) or {}
    if fusion.get("signal") not in ("BUY SIGNAL", "SELL SIGNAL"):
        return False, "No BUY/SELL signal from Signal API"
    if safe_float(fusion.get("confidence"), 0) < AUTO_ORDER_MIN_CONFIDENCE:
        return False, f"Confidence below minimum: {fusion.get('confidence')} < {AUTO_ORDER_MIN_CONFIDENCE}"
    filt = fusion.get("filter", {}) or {}
    if filt and not filt.get("allowed", True):
        return False, "Signal quality filter rejected trade: " + "; ".join(filt.get("reasons", []))

    # Final live safety gate: asset must be GREEN and tradable in the latest cached 7-day regression pack.
    # GREEN means >=80% win rate, Total R >= configured minimum, average confidence >= configured minimum, and sample accepted: signals >= 5 OR wins >= 3.
    health = get_asset_regression_health(asset_key)
    if not (health.get("health") == "GREEN" and health.get("tradable") is True):
        return False, (
            f"Regression health gate blocked {asset_key}: "
            f"health={health.get('health')}, tradable={health.get('tradable')}, "
            f"signals={health.get('signals')}, win_rate={health.get('win_rate_percent')}%, "
            f"total_r={health.get('total_r')}R, status={health.get('status')}"
        )

    return True, f"OK - order allowed. Regression health gate GREEN with win rate >= {REGRESSION_GREEN_WIN_RATE:.0f}%, Total R >= {REGRESSION_MIN_TOTAL_R}, avg confidence >= {REGRESSION_MIN_AVG_CONFIDENCE}, and sample accepted."


async def calculate_order_size(asset_key, entry, stop):
    equity = await account_equity()
    risk_per_unit = abs(safe_float(entry) - safe_float(stop))
    if equity <= 0 or risk_per_unit <= 0:
        return ORDER_FALLBACK_SIZE, 0
    risk_capital = equity * ORDER_RISK_PER_TRADE_PCT
    qty = max(ORDER_FALLBACK_SIZE, risk_capital / risk_per_unit)
    if asset_key in ("BTC",):
        qty = round(qty, 4)
    else:
        qty = round(qty, 2)
    return qty, round(risk_capital, 2)


async def place_order_from_signal(asset_key, signal_data):
    asset_key = str(asset_key).upper()
    fusion = signal_data.get("fusion", {}) or {}
    tech = signal_data.get("tech", {}) or {}
    direction = "BUY" if fusion.get("signal") == "BUY SIGNAL" else "SELL" if fusion.get("signal") == "SELL SIGNAL" else ""
    if direction not in ("BUY", "SELL"):
        return {"ok": False, "error": "No executable signal"}

    quote = await capital_quote(asset_key)
    entry = safe_float(quote.get("price"), 0) or safe_float(tech.get("entry"), 0) or safe_float(tech.get("price"), 0)
    stop = safe_float(tech.get("stop"), 0)
    target = safe_float(tech.get("target"), 0)

    # Ensure SL/TP are on the correct side of the live entry.
    atr = max(safe_float(tech.get("atr"), entry * 0.006), entry * 0.004)
    if direction == "BUY":
        if stop <= 0 or stop >= entry:
            stop = entry - atr * 2.0
        if target <= entry:
            target = entry + abs(entry - stop) * 1.65
    else:
        if stop <= entry:
            stop = entry + atr * 2.0
        if target <= 0 or target >= entry:
            target = entry - abs(stop - entry) * 1.65

    qty, risk_capital = await calculate_order_size(asset_key, entry, stop)
    payload = {
        "epic": _asset_epic(asset_key),
        "direction": direction,
        "size": qty,
        "orderType": "MARKET",
        "currencyCode": ORDER_CURRENCY_CODE,
        "forceOpen": True,
        "guaranteedStop": False,
        "stopLevel": round(stop, 4),
        "profitLevel": round(target, 4),
    }
    code, response = await capital_post("/positions", payload)
    result = {
        "ok": 200 <= int(code) < 300,
        "status_code": code,
        "asset": asset_key,
        "direction": direction,
        "entry_price_used": round(entry, 4),
        "stop_loss": round(stop, 4),
        "take_profit": round(target, 4),
        "qty": qty,
        "risk_capital": risk_capital,
        "spread_pct": quote.get("spread_pct"),
        "payload": payload,
        "response": response,
        "time": datetime.now(timezone.utc).isoformat(),
    }
    record_order(result)
    return result


async def run_auto_order(asset_key, tf="30M", execute=True):
    asset_key = str(asset_key or "WTI").upper()
    # Production order execution is locked to 30M signal timeframe.
    # Chart timeframe can change in UI, but orders must use the validated signal timeframe.
    tf = "30M"
    if asset_key not in ASSETS:
        return {"ok": False, "error": "Invalid asset"}
    signal_data = await process(asset_key, tf)
    allowed, reason = await order_guard(asset_key, signal_data, execute=execute)
    order = None
    if allowed:
        order = await place_order_from_signal(asset_key, signal_data)
    return {
        "auto_order_enabled": AUTO_ORDER_RUNTIME["enabled"],
        "execute": bool(execute),
        "allowed": bool(allowed),
        "guard": reason,
        "signal": {
            "asset": asset_key,
            "tf": tf,
            "signal": signal_data.get("fusion", {}).get("signal"),
            "confidence": signal_data.get("fusion", {}).get("confidence"),
            "entry": signal_data.get("tech", {}).get("entry"),
            "target": signal_data.get("tech", {}).get("target"),
            "stop": signal_data.get("tech", {}).get("stop"),
        },
        "order": order,
    }


# ============================================================
# WTI SIGNAL REGRESSION / BACKTEST
# ============================================================
def _regression_signal_from_score(score, regime_name="NORMAL", tech=None):
    threshold = 18 if regime_name == "TRENDING" else 28 if regime_name == "RANGING" else 30 if regime_name == "HIGH VOLATILITY" else 22
    side = "HOLD"
    if score >= threshold:
        side = "BUY"
    elif score <= -threshold:
        side = "SELL"
    if side in ("BUY", "SELL") and tech is not None:
        q = signal_quality_filter(side, tech)
        if not q.get("allowed", True):
            return "HOLD"
    return side


def _simulate_trade_path(future_df, direction, entry, target, stop):
    """Return WIN / STOP / FORCED_EXIT using conservative intrabar handling.

    If target and stop are both touched in the same candle, STOP is assumed first.
    If neither target nor stop is hit, the trade is force-closed at the final
    lookahead candle so regression P/L is fully counted.
    """
    entry = safe_float(entry)
    target = safe_float(target)
    stop = safe_float(stop)
    for offset, (_, row) in enumerate(future_df.iterrows(), start=1):
        high = safe_float(row.get("high"))
        low = safe_float(row.get("low"))
        t = str(row.get("time"))
        if direction == "BUY":
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            stop_hit = high >= stop
            target_hit = low <= target
        if stop_hit and target_hit:
            return {"result": "STOP", "exit_price": stop, "exit_time": t, "bars_held": offset, "reason": "Both target and stop touched; conservative stop-first rule"}
        if stop_hit:
            return {"result": "STOP", "exit_price": stop, "exit_time": t, "bars_held": offset, "reason": "Stop loss hit"}
        if target_hit:
            return {"result": "WIN", "exit_price": target, "exit_time": t, "bars_held": offset, "reason": "Target achieved"}
    last = future_df.iloc[-1] if len(future_df) else None
    exit_price = safe_float(last.get("close"), entry) if last is not None else entry
    exit_time = str(last.get("time")) if last is not None else ""
    bars_held = len(future_df) if len(future_df) else 0
    return {"result": "FORCED_EXIT", "exit_price": exit_price, "exit_time": exit_time, "bars_held": bars_held, "reason": "Force closed at lookahead window; target/stop not hit"}


def run_wti_signal_regression(tf="30M", lookahead=24, max_bars=360, min_gap=3):
    """Backtest WTI BUY/SELL signals using only historical candles.

    The live dashboard signal also uses current news/AI. Historical news/AI is not replayed here,
    so this regression tests the repeatable technical signal engine and its generated entry,
    target and stop loss levels.
    """
    tf = str(tf or "30M").upper()
    if tf not in INTERVALS:
        raise ValueError("Invalid timeframe")
    lookahead = max(3, min(int(lookahead or 24), 120))
    max_bars = max(120, min(int(max_bars or 360), 420))
    min_gap = max(1, min(int(min_gap or 3), 20))

    df = load_candles("WTI", tf)
    if df is None or df.empty or len(df) < 90:
        return {"error": "Not enough WTI candle history returned from Capital.com for regression."}

    d = clean_market_candles(df, tf).tail(max_bars).reset_index(drop=True)
    rows = []
    active_until = []
    last_trade_i = -999
    start_i = max(80, min(220, len(d) // 2))
    end_i = len(d) - lookahead - 1

    for i in range(start_i, end_i):
        active_until = [x for x in active_until if x > i]
        if len(active_until) >= 2:
            continue
        if i - last_trade_i < min_gap:
            continue
        hist = d.iloc[: i + 1].copy()
        try:
            tech = technical_score(hist, mtf={"score": 0})
            score = safe_float(tech.get("score"), 0)
            regime_name = (tech.get("regime") or {}).get("regime", "NORMAL")
            side = _regression_signal_from_score(score, regime_name, tech)
            if side == "HOLD":
                continue
            entry = safe_float(tech.get("entry"), safe_float(hist["close"].iloc[-1]))
            target = safe_float(tech.get("target"))
            stop = safe_float(tech.get("stop"))
            if entry <= 0 or target <= 0 or stop <= 0:
                continue
            future = d.iloc[i + 1 : i + 1 + lookahead].copy()
            sim = _simulate_trade_path(future, side, entry, target, stop)
            if side == "BUY":
                pnl_points = safe_float(sim["exit_price"]) - entry
                risk_points = entry - stop
                reward_points = target - entry
            else:
                pnl_points = entry - safe_float(sim["exit_price"])
                risk_points = stop - entry
                reward_points = entry - target
            r_multiple = pnl_points / risk_points if risk_points > 0 else 0
            result = sim["result"]
            if result == "FORCED_EXIT":
                result = "FORCED_WIN" if r_multiple > 0 else "FORCED_LOSS" if r_multiple < 0 else "FORCED_FLAT"
            rows.append({
                "sno": len(rows) + 1,
                "signal_time": str(hist["time"].iloc[-1]),
                "side": side,
                "score": round(score, 2),
                "confidence_proxy": round(min(100, max(25, abs(score) * 1.28 + 12.5)), 1),
                "regime": regime_name,
                "entry": round(entry, 3),
                "target": round(target, 3),
                "stop": round(stop, 3),
                "exit_time": sim["exit_time"],
                "exit_price": round(safe_float(sim["exit_price"]), 3),
                "result": result,
                "pnl_points": round(pnl_points, 3),
                "r_multiple": round(r_multiple, 3),
                "risk_points": round(abs(risk_points), 3),
                "reward_points": round(abs(reward_points), 3),
                "reason": sim["reason"],
            })
            active_until.append(i + int(sim.get("bars_held", lookahead)))
            last_trade_i = i
        except Exception:
            continue

    closed = rows
    wins = sum(1 for r in closed if r["result"] in ("WIN", "FORCED_WIN"))
    losses = sum(1 for r in closed if r["result"] in ("STOP", "FORCED_LOSS"))
    stops = sum(1 for r in closed if r["result"] == "STOP")
    forced_exits = sum(1 for r in rows if str(r["result"]).startswith("FORCED"))
    open_trades = 0
    avg_r = float(np.mean([r["r_multiple"] for r in closed])) if closed else 0.0
    total_r = float(np.sum([r["r_multiple"] for r in closed])) if closed else 0.0
    win_rate = wins / len(closed) * 100 if closed else 0.0
    return {
        "asset": "WTI",
        "epic": ASSETS["WTI"]["epic"],
        "timeframe": tf,
        "lookahead_bars": lookahead,
        "tested_candles": int(len(d)),
        "note": "Technical-only regression with improved quality filters: ADX >= 20, EMA trend alignment, RSI confirmation, ATR spike guard, max two simultaneous trades, and forced close at lookahead.",
        "summary": {
            "total_signals": len(rows),
            "closed_trades": len(closed),
            "wins": wins,
            "losses": losses,
            "stops": stops,
            "forced_exits": forced_exits,
            "open": open_trades,
            "max_simultaneous_trades": 2,
            "win_rate_percent": round(win_rate, 2),
            "avg_r_multiple": round(avg_r, 3),
            "total_r_multiple": round(total_r, 3),
        },
        "trades": rows[-100:],
    }



@app.get("/api/login")
async def api_login():
    try:
        await asyncio.to_thread(capital_login)
        return {"ok": True, "message": "Capital.com login OK", "session_time": CAPITAL_SESSION.get("ts")}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)


@app.get("/api/account")
async def api_account():
    code, data = await capital_get("/accounts")
    return {"status_code": code, "data": data}


@app.get("/api/positions")
async def api_positions():
    code, data = await capital_get("/positions")
    return {"status_code": code, "data": data}


@app.get("/api/orders")
async def api_orders():
    code, data = await capital_get("/workingorders")
    return {"status_code": code, "data": data, "local_order_state": read_order_state()}


@app.get("/api/auto-order/status")
async def api_auto_order_status():
    return {
        "enabled": AUTO_ORDER_RUNTIME["enabled"],
        "max_open_positions": MAX_OPEN_POSITIONS,
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "today_order_count": today_order_count(),
        "min_confidence": AUTO_ORDER_MIN_CONFIDENCE,
    }


@app.post("/api/auto-order/toggle")
async def api_auto_order_toggle():
    AUTO_ORDER_RUNTIME["enabled"] = not AUTO_ORDER_RUNTIME["enabled"]
    return await api_auto_order_status()


@app.get("/api/place-order")
async def api_place_order(asset: str = Query("WTI"), tf: str = Query("30M"), execute: bool = Query(True)):
    try:
        return await run_auto_order(asset, tf, execute)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)


@app.get("/api/hit")
async def api_hit():
    with HIT_LOCK:
        hits = read_hit_count() + 1
        write_hit_count(hits)
    return {"hits": hits}


@app.get("/api/hits")
async def api_hits():
    return {"hits": read_hit_count()}


@app.get("/api/signal")
async def api_signal(asset: str = Query("GOLD"), tf: str = Query("1H")):
    asset = asset.upper()
    tf = tf.upper()
    if asset not in ASSETS:
        return JSONResponse({"error": "Invalid asset"}, status_code=400)
    if tf not in INTERVALS:
        return JSONResponse({"error": "Invalid timeframe"}, status_code=400)
    try:
        return await process(asset, tf)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=200)


@app.get("/api/regression/wti")
async def api_regression_wti(tf: str = Query("30M"), lookahead: int = Query(24), max_bars: int = Query(360)):
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, run_wti_signal_regression, tf, lookahead, max_bars)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=200)


# ============================================================
# QUICK REGRESSION PACK - ALL COMMODITIES / ASSET HEALTH
# ============================================================
def run_quick_regression_pack(tf="30M", hours=REGRESSION_PACK_HOURS, min_gap=3, min_confidence=REGRESSION_MIN_CONFIDENCE):
    """Quick technical regression across all dashboard commodities.

    Uses the same signal quality filter as the live Signal API, then checks
    whether generated BUY/SELL setups hit target or stop inside the remaining
    recent-data window. Forced exits are counted by actual P/L at the last
    available candle in the lookahead window.
    """
    # Always regression-test the validated 30M signal timeframe.
    # This keeps regression aligned with the production order module.
    tf = "30M"
    hours = max(24, min(int(hours or REGRESSION_PACK_HOURS), 168))
    min_gap = max(1, min(int(min_gap or 3), 20))
    min_confidence = max(0, min(float(min_confidence or 80), 100))
    now_utc = pd.Timestamp.now(tz="UTC")
    start_cutoff = now_utc - pd.Timedelta(hours=hours)
    rows = []

    for asset_key in ASSETS.keys():
        asset_rows = []
        try:
            df = load_candles(asset_key, tf)
            if df is None or df.empty or len(df) < 90:
                rows.append({
                    "asset_key": asset_key, "name": ASSETS[asset_key]["name"],
                    "signals": 0, "wins": 0, "losses": 0, "stops": 0, "forced": 0,
                    "win_rate_percent": 0.0, "total_r": 0.0, "avg_r": 0.0, "avg_confidence": 0.0,
                    "rank_score": -9999, "health": "INSUFFICIENT", "tradable": False,
                    "status": "Not enough candle history", "trades": []
                })
                continue

            d = clean_market_candles(df, tf).reset_index(drop=True)
            d["time"] = pd.to_datetime(d["time"], errors="coerce", utc=True)
            if d.empty:
                raise RuntimeError("No clean candles")

            start_i = max(80, int(d.index[d["time"] >= start_cutoff].min()) if (d["time"] >= start_cutoff).any() else len(d) - 48)
            start_i = max(80, min(start_i, len(d) - 2))
            active_until = []
            last_trade_i = -999

            for i in range(start_i, len(d) - 1):
                active_until = [x for x in active_until if x > i]
                if len(active_until) >= 2:
                    continue
                if i - last_trade_i < min_gap:
                    continue

                hist = d.iloc[: i + 1].copy()
                tech = technical_score(hist, mtf={"score": 0})
                score = safe_float(tech.get("score"), 0)
                regime_name = (tech.get("regime") or {}).get("regime", "NORMAL")
                side = _regression_signal_from_score(score, regime_name, tech)
                if side == "HOLD":
                    continue

                confidence_proxy = round(min(100, max(25, abs(score) * 1.28 + 12.5)), 1)
                if confidence_proxy < min_confidence:
                    continue

                entry = safe_float(tech.get("entry"), safe_float(hist["close"].iloc[-1]))
                target = safe_float(tech.get("target"), 0)
                stop = safe_float(tech.get("stop"), 0)
                if entry <= 0 or target <= 0 or stop <= 0:
                    continue

                # Only evaluate against candles inside the last-hours test window.
                future = d.iloc[i + 1 :].copy()
                future = future[future["time"] <= d["time"].iloc[-1]]
                if future.empty:
                    continue
                sim = _simulate_trade_path(future, side, entry, target, stop)

                if side == "BUY":
                    pnl_points = safe_float(sim["exit_price"]) - entry
                    risk_points = entry - stop
                    reward_points = target - entry
                else:
                    pnl_points = entry - safe_float(sim["exit_price"])
                    risk_points = stop - entry
                    reward_points = entry - target
                r_multiple = pnl_points / risk_points if risk_points > 0 else 0
                result = sim["result"]
                if result == "FORCED_EXIT":
                    result = "FORCED_WIN" if r_multiple > 0 else "FORCED_LOSS" if r_multiple < 0 else "FORCED_FLAT"

                asset_rows.append({
                    "time": str(hist["time"].iloc[-1]),
                    "side": side,
                    "score": round(score, 2),
                    "confidence_proxy": confidence_proxy,
                    "regime": regime_name,
                    "entry": round(entry, 3),
                    "target": round(target, 3),
                    "stop": round(stop, 3),
                    "exit_time": sim.get("exit_time", ""),
                    "exit_price": round(safe_float(sim.get("exit_price")), 3),
                    "result": result,
                    "pnl_points": round(pnl_points, 3),
                    "r_multiple": round(r_multiple, 3),
                    "risk_points": round(abs(risk_points), 3),
                    "reward_points": round(abs(reward_points), 3),
                })
                active_until.append(i + int(sim.get("bars_held", 1)))
                last_trade_i = i

            wins = sum(1 for r in asset_rows if r["result"] in ("WIN", "FORCED_WIN"))
            losses = sum(1 for r in asset_rows if r["result"] in ("STOP", "FORCED_LOSS"))
            stops = sum(1 for r in asset_rows if r["result"] == "STOP")
            forced = sum(1 for r in asset_rows if str(r["result"]).startswith("FORCED"))
            total = len(asset_rows)
            total_r = float(np.sum([r["r_multiple"] for r in asset_rows])) if asset_rows else 0.0
            avg_r = float(np.mean([r["r_multiple"] for r in asset_rows])) if asset_rows else 0.0
            win_rate = round((wins / total * 100) if total else 0, 2)
            avg_confidence = round(float(np.mean([r.get("confidence_proxy", 0) for r in asset_rows])) if asset_rows else 0.0, 2)

            sample_ok = (total >= REGRESSION_MIN_SIGNALS) or (wins >= REGRESSION_MIN_WINS)
            rate_ok = win_rate >= REGRESSION_GREEN_WIN_RATE
            r_ok = total_r >= REGRESSION_MIN_TOTAL_R
            confidence_ok = avg_confidence >= REGRESSION_MIN_AVG_CONFIDENCE

            if rate_ok and r_ok and sample_ok and confidence_ok:
                health = "GREEN"
                tradable = True
                status = f"OK - auto-trade approved. Win rate >= {REGRESSION_GREEN_WIN_RATE:.0f}%, Total R >= {REGRESSION_MIN_TOTAL_R}, avg confidence >= {REGRESSION_MIN_AVG_CONFIDENCE}, and sample accepted ({total} signals / {wins} wins)."
            elif rate_ok and total_r > 0 and not sample_ok:
                health = "PASS RATE LOW SAMPLE"
                tradable = False
                status = f"PASS RATE BUT LOW SAMPLE: {total}/{REGRESSION_MIN_SIGNALS} signals and {wins}/{REGRESSION_MIN_WINS} wins. Need more data before auto order."
            elif total_r > 0:
                health = "AMBER"
                tradable = False
                status = f"WATCH ONLY - positive R but gate not passed. Requires win rate >= {REGRESSION_GREEN_WIN_RATE:.0f}%, Total R >= {REGRESSION_MIN_TOTAL_R}, avg confidence >= {REGRESSION_MIN_AVG_CONFIDENCE}, and sample accepted."
            else:
                health = "RED"
                tradable = False
                status = f"BLOCKED - Total R is not positive or win rate is weak."

            # Weighted ranking: profitability matters most, then win rate, then average confidence.
            # Low-sample pass-rate assets are ranked after sample-approved assets but still visible.
            rank_score_raw = (total_r * 70.0) + ((win_rate / 100.0) * 30.0) + ((avg_confidence / 100.0) * 10.0)
            rank_score = round(rank_score_raw, 3) if sample_ok else round(-5000 + rank_score_raw, 3)

            rows.append({
                "asset_key": asset_key,
                "name": ASSETS[asset_key]["name"],
                "signals": total,
                "wins": wins,
                "losses": losses,
                "stops": stops,
                "forced": forced,
                "win_rate_percent": win_rate,
                "total_r": round(total_r, 3),
                "avg_r": round(avg_r, 3),
                "avg_confidence": avg_confidence,
                "rank_score": rank_score,
                "health": health,
                "tradable": bool(tradable),
                "auto_order_allowed": bool(health == "GREEN" and tradable),
                "status": status,
                "trades": asset_rows[-20:],
            })
        except Exception as e:
            rows.append({
                "asset_key": asset_key, "name": ASSETS.get(asset_key, {}).get("name", asset_key),
                "signals": 0, "wins": 0, "losses": 0, "stops": 0, "forced": 0,
                "win_rate_percent": 0.0, "total_r": 0.0, "avg_r": 0.0, "avg_confidence": 0.0, "rank_score": -9999,
                "health": "RED", "tradable": False, "auto_order_allowed": False,
                "status": "ERROR: " + str(e)[:160], "trades": []
            })

    ranked_rows = sorted(rows, key=lambda r: (safe_float(r.get("rank_score"), -9999), safe_float(r.get("total_r"), 0), safe_float(r.get("win_rate_percent"), 0)), reverse=True)
    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx
    rank_lookup = {r["asset_key"]: r.get("rank") for r in ranked_rows}
    for row in rows:
        row["rank"] = rank_lookup.get(row.get("asset_key"), 0)

    total_signals = sum(r["signals"] for r in rows)
    total_wins = sum(r["wins"] for r in rows)
    total_losses = sum(r["losses"] for r in rows)
    total_r = float(np.sum([r["total_r"] for r in rows])) if rows else 0.0
    return {
        "title": f"Regression Pack - Last {hours} Hours",
        "timeframe": tf,
        "hours": hours,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": f"Quick technical-only regression using 30M Signal API logic with global quality filter, confidence >= {min_confidence}. Auto-trade approval requires win rate >= {REGRESSION_GREEN_WIN_RATE:.0f}%, Total R >= {REGRESSION_MIN_TOTAL_R}, avg confidence >= {REGRESSION_MIN_AVG_CONFIDENCE}, and sample accepted: {REGRESSION_MIN_SIGNALS}+ signals OR {REGRESSION_MIN_WINS}+ wins. News/AI is not historically replayed.",
        "summary": {
            "assets_tested": len(rows),
            "total_signals": total_signals,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "overall_win_rate_percent": round((total_wins / total_signals * 100) if total_signals else 0, 2),
            "total_r_multiple": round(total_r, 3),
            "min_confidence": min_confidence,
            "min_total_r": REGRESSION_MIN_TOTAL_R,
            "min_avg_confidence": REGRESSION_MIN_AVG_CONFIDENCE,
            "approved_assets": [r["asset_key"] for r in ranked_rows if r.get("auto_order_allowed")],
            "watchlist_assets": [r["asset_key"] for r in ranked_rows if r.get("health") == "AMBER"],
            "disabled_assets": [r["asset_key"] for r in ranked_rows if r.get("health") in ("RED", "INSUFFICIENT")],
        },
        "ranking": ranked_rows,
        "assets": rows,
    }


@app.get("/api/regression/pack")
async def api_regression_pack(tf: str = Query("30M"), hours: int = Query(REGRESSION_PACK_HOURS), min_confidence: int = Query(int(REGRESSION_MIN_CONFIDENCE))):
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, run_quick_regression_pack, "30M", hours, 3, min_confidence)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=200)


@app.get("/api/market-strip")
async def api_market_strip():
    """Live top market cards. No dummy values: every card is sourced from Capital.com candles/cache."""
    loop = asyncio.get_running_loop()

    async def one(asset_key):
        try:
            df = await loop.run_in_executor(executor, load_candles, asset_key, "30M")
            if df is None or df.empty:
                return {"asset_key": asset_key, "error": "No data"}
            d = add_indicators(df)
            last = d.iloc[-1]
            try:
                market_snapshot = await loop.run_in_executor(executor, get_capital_market_snapshot, ASSETS[asset_key]["epic"])
            except Exception:
                market_snapshot = None
            ch = compute_daily_change(d, market_snapshot)
            live_price = safe_float((market_snapshot or {}).get("price"), safe_float(last.close))
            return {
                "asset_key": asset_key,
                "name": ASSETS[asset_key]["name"],
                "icon": ASSETS[asset_key]["icon"],
                "epic": ASSETS[asset_key]["epic"],
                "price": round(live_price, 3),
                "change": ch,
                "rsi": round(safe_float(last.rsi), 2),
                "trend": "BULLISH" if last.ema9 > last.ema21 else "BEARISH" if last.ema9 < last.ema21 else "NEUTRAL",
            }
        except Exception as e:
            return {"asset_key": asset_key, "error": str(e)[:120]}

    rows = await asyncio.gather(*[one(k) for k in ASSETS.keys()])
    return {"items": rows, "updated": datetime.now().strftime("%H:%M:%S")}


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI MARKETLENS - By Syed Abbas</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root{
  --bg:#050d18;--panel:#0a1728;--panel2:#0e1d31;--line:#20344e;--text:#eef6ff;--muted:#8ea2bb;
  --green:#20d66b;--red:#ff405d;--amber:#f5b02e;--blue:#2d8cff;--cyan:#21d4ff;--violet:#a855f7;
}
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;font-family:Inter,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text)}
.app{height:100vh;padding:6px;display:grid;grid-template-rows:5px 54px 42px minmax(0,1fr) 28px;gap:5px;background:radial-gradient(circle at 8% 0%,rgba(45,140,255,.18),transparent 28%),linear-gradient(180deg,#06101d,#081320)}
.topbar{display:grid;grid-template-columns:340px 1fr 285px;gap:10px;align-items:center;border:1px solid var(--line);border-radius:8px;background:rgba(5,13,24,.88);padding:6px 10px}.brand{display:flex;gap:10px;align-items:center;min-width:0}.logo{width:46px;height:46px;border-radius:12px;display:grid;place-items:center;background:linear-gradient(135deg,#4259ff,#00d4ff);font-weight:1000;font-size:30px;letter-spacing:.02em}.brand h1{font-size:18px;margin:0;letter-spacing:.02em}.brand p{margin:3px 0 0;color:#74a7ff;font-size:18px;font-weight:1000;line-height:1}.marketStrip{height:42px;display:grid;grid-template-columns:repeat(6,minmax(110px,1fr));border:1px solid var(--line);border-radius:7px;overflow:hidden;background:#081523}.tickerCard{padding:6px 10px;border-right:1px solid var(--line);min-width:0}.tickerCard:last-child{border-right:0}.tickerCard b{display:block;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.tickerCard strong{display:inline-block;margin-top:4px;font-size:14px}.tickerCard span{float:right;margin-top:5px;font-size:11px;font-weight:900}.topRight{display:flex;align-items:center;justify-content:flex-end;gap:10px}.iconBtn{width:34px;height:34px;border-radius:10px;border:1px solid var(--line);display:grid;place-items:center;background:#091827;color:#dbeafe;font-weight:900}.hitBox{border-left:1px solid var(--line);padding-left:12px;font-size:18px;line-height:1.05;color:#d8e6ff;font-weight:1000}.hitBox b{color:#24d6ff;font-size:32px;line-height:1.05}.live{border-left:1px solid var(--line);padding-left:10px;font-size:12px;color:#d8e6ff}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);margin-left:5px}.refreshBar{height:5px;border-radius:99px;background:#102237;overflow:hidden;border:1px solid rgba(45,140,255,.25)}.refreshFill{height:100%;width:100%;border-radius:99px;background:linear-gradient(90deg,#2d8cff,#20d66b,#f5b02e);transform-origin:left center;animation:refreshRun 30s linear infinite}@keyframes refreshRun{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.toolbar{display:grid;grid-template-columns:48px 292px 1fr;gap:8px;align-items:center}.menu,.selectBox,.tfTabs{height:38px;border:1px solid var(--line);background:#091827;border-radius:8px}.menu{display:grid;place-items:center;font-size:18px;color:#b7c7dc}.selectBox{display:flex;align-items:center;gap:10px;padding:0 12px}.selectBox label{font-size:12px;color:var(--muted)}select{background:transparent;border:0;outline:0;color:var(--text);font-weight:900;font-size:14px;flex:1}option{background:#0d1828;color:#fff}.tfTabs{display:flex;align-items:center;gap:5px;padding:5px;width:max-content;min-width:360px}.tfTabs button{height:28px;min-width:48px;border:0;border-radius:7px;background:transparent;color:#cbd5e1;font-weight:800}.tfTabs button.active{background:#0d4a87;color:#fff;box-shadow:inset 0 0 0 1px #2d8cff}.error{position:absolute;top:112px;left:370px;z-index:5;color:#ff8093;font-weight:900;font-size:12px}
.grid{min-height:0;height:100%;display:grid;grid-template-columns:340px minmax(720px,1fr);gap:7px;overflow:hidden}.left{min-height:0;height:100%;display:grid;grid-template-rows:124px 146px 126px minmax(116px,1fr);gap:5px;overflow:hidden}.panel{min-height:0;background:linear-gradient(180deg,rgba(15,32,52,.98),rgba(8,20,34,.98));border:1px solid var(--line);border-radius:8px;overflow:hidden;box-shadow:0 16px 45px rgba(0,0,0,.30)}.panelHead{height:30px;display:flex;align-items:center;justify-content:space-between;padding:0 13px;border-bottom:1px solid rgba(255,255,255,.07);font-weight:900;font-size:13px}.info{color:#65768d}.aiSignal{display:block;padding:10px 16px;overflow:hidden}.signalLine{display:flex;align-items:flex-end;gap:12px;min-width:0;margin:2px 0 8px}.assetBadge{font-size:14px;font-weight:1000;line-height:1.05;letter-spacing:.03em;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-bottom:4px}.bullIcon{display:none}.signalText{font-size:27px;line-height:1;font-weight:1000;margin:0;color:var(--green);white-space:nowrap}.confidenceLabel{font-size:15px;line-height:1.1}.confidence{font-size:22px;font-weight:1000;color:#00ff93;line-height:1.05}.bars{display:flex;gap:3px;align-items:flex-end;height:14px;margin-top:5px}.bars i{width:4px;border-radius:8px;background:var(--green)}.tradeRows{padding:5px 14px}.tradeRow{height:27px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.07);font-weight:900}.tradeRow span{font-size:13px}.tradeRow b{font-size:16px;white-space:nowrap}.gaugeWrap{height:91px;display:flex;align-items:center;justify-content:center}.gaugeSvg{width:190px;height:96px}.gaugeTrack{fill:none;stroke:#173150;stroke-width:18;stroke-linecap:round}.gaugeGreen{fill:none;stroke:var(--green);stroke-width:18;stroke-linecap:round}.gaugeYellow{fill:none;stroke:#e3df2f;stroke-width:18;stroke-linecap:round}.gaugeOrange{fill:none;stroke:var(--amber);stroke-width:18;stroke-linecap:round}.gaugeRed{fill:none;stroke:var(--red);stroke-width:18;stroke-linecap:round}.needle{stroke:#fff;stroke-width:4;stroke-linecap:round;filter:drop-shadow(0 0 5px rgba(255,255,255,.7));transform-origin:115px 110px}.gaugeNum{font-size:31px;font-weight:1000;fill:#fff;text-anchor:middle}.gaugeLabelText{font-size:12px;font-weight:1000;text-anchor:middle}.donutBox{height:calc(100% - 30px);display:grid;grid-template-columns:92px 1fr;gap:7px;align-items:center;padding:6px 10px;overflow:hidden}.donut{width:86px;height:86px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--green) 0deg 190deg,var(--violet) 190deg 285deg,var(--cyan) 285deg 360deg);position:relative}.donut:after{content:"";position:absolute;inset:13px;background:#091827;border-radius:50%}.donut b{z-index:1;font-size:24px}.legend div{display:flex;justify-content:space-between;margin:4px 0;color:#dbeafe;font-size:11px}.legend h3{margin:4px 0 0;font-size:15px}.pos{color:var(--green)!important}.neg{color:var(--red)!important}.neu{color:var(--amber)!important}
.rightMain{min-height:0;height:100%;display:grid;grid-template-rows:52px minmax(250px,1fr) 216px;gap:6px;overflow:hidden}.kpiStrip{display:grid;grid-template-columns:repeat(7,1fr);border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#0e1d30}.kpi{padding:7px 10px;border-right:1px solid rgba(255,255,255,.08)}.kpi:last-child{border-right:0}.kpi small{display:block;color:#c8d6e8;font-size:11px;font-weight:900}.kpi b{display:block;margin-top:5px;font-size:16px}.chartPanel{position:relative}.chartTop{position:absolute;top:8px;left:14px;right:14px;z-index:2;display:flex;justify-content:space-between;font-size:12px;color:#dbeafe;pointer-events:none}.tvBadge{height:28px;border:1px solid var(--line);border-radius:7px;padding:6px 10px;background:#091827;font-weight:900}.chartWrap{height:100%;padding-top:30px}.chart{width:100%;height:100%}.bottomCards{min-height:0;display:grid;grid-template-columns:1fr 1.05fr 1.1fr 1.05fr;gap:7px}.body{padding:8px 12px;overflow:hidden;height:calc(100% - 30px)}.line{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.06);font-size:12px}.aiText{font-size:12px;line-height:1.35;color:#e2edf9}.recommend{margin-top:8px;color:#52a8ff;font-size:11px;font-weight:900}.newsSent{display:grid;grid-template-columns:1fr 96px;gap:8px;align-items:center;border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:8px;margin-bottom:6px}.newsScore{font-size:22px;font-weight:1000}.newsGaugeSvg{width:96px;height:58px}.newsNeedle{stroke:#fff;stroke-width:3;stroke-linecap:round;filter:drop-shadow(0 0 4px rgba(255,255,255,.7));transform-origin:48px 50px}.newsGaugeNum{font-size:12px;font-weight:1000;fill:#fff;text-anchor:middle}.newsItem{display:grid;grid-template-columns:34px 1fr 10px;gap:8px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.06);font-size:11px}.newsItem a{color:#e2edf9;text-decoration:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.driver{display:grid;grid-template-columns:28px 1fr 62px;gap:8px;align-items:center;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.06);font-size:12px}.driverIcon{width:25px;height:25px;border-radius:50%;display:grid;place-items:center;background:rgba(245,158,11,.13);color:#facc15;font-size:11px;font-weight:900}.foot{display:flex;align-items:center;justify-content:space-between;color:#8ea2bb;font-size:11px;border:1px solid var(--line);border-radius:8px;background:#07111f;padding:0 14px}

/* ===== THEME TOGGLE: Default is Dark. Compact button to avoid overlapping header content ===== */
.themeSwitch{
  width:32px;height:32px;min-width:32px;border:1px solid var(--line);border-radius:8px;
  background:#091827;color:#dbeafe;font-weight:900;font-size:14px;line-height:1;
  display:flex;align-items:center;justify-content:center;padding:0;margin:0;
  cursor:pointer;user-select:none;transition:.2s ease;flex:0 0 32px;
}
.themeSwitch:hover{transform:scale(1.05)}
.themeSwitch .modeText{display:block;min-width:0;text-align:center;line-height:1}
body.light-theme{
  --bg:#edf3fb;--panel:#ffffff;--panel2:#f5f8fc;--line:#c9d6e8;--text:#102033;--muted:#5c6d82;
}
body.light-theme{
  background:#edf3fb;color:var(--text);
}
body.light-theme .app{
  background:radial-gradient(circle at 8% 0%,rgba(45,140,255,.12),transparent 28%),linear-gradient(180deg,#f8fbff,#eaf1fb);
}
body.light-theme .topbar,
body.light-theme .menu,
body.light-theme .selectBox,
body.light-theme .tfTabs,
body.light-theme .tvBadge,
body.light-theme .foot,
body.light-theme .marketStrip{
  background:#ffffff;color:#102033;border-color:var(--line);
}
body.light-theme .panel{
  background:linear-gradient(180deg,#ffffff,#f4f8fd);
  color:#102033;border-color:var(--line);box-shadow:0 14px 32px rgba(30,64,120,.12);
}
body.light-theme .panelHead,
body.light-theme .tradeRow,
body.light-theme .line,
body.light-theme .driver,
body.light-theme .newsItem{
  border-color:rgba(15,35,65,.10);
}
body.light-theme .tickerCard{border-color:var(--line)}
body.light-theme .kpiStrip{background:#ffffff;border-color:var(--line)}
body.light-theme .kpi{border-color:rgba(15,35,65,.10)}
body.light-theme .chartPanel{background:#ffffff}
body.light-theme .chartWrap{background:#ffffff}
body.light-theme .chartTop,
body.light-theme .aiText,
body.light-theme .legend div,
body.light-theme .driver span,
body.light-theme .newsItem a,
body.light-theme .kpi small,
body.light-theme .live,
body.light-theme .hitBox{
  color:#102033;
}
body.light-theme select{color:#102033}
body.light-theme option{background:#ffffff;color:#102033}
body.light-theme .tfTabs button{color:#334155}
body.light-theme .tfTabs button.active{background:#dcecff;color:#0b3b72;box-shadow:inset 0 0 0 1px #2d8cff}
body.light-theme .themeSwitch{background:#eaf2ff;color:#0b3b72}
body.light-theme .gaugeTrack{stroke:#d5e1f1}
body.light-theme .donut:after{background:#ffffff}


/* ===== AUTO ORDER BUTTONS / POPUPS ===== */
.orderBtn{height:28px;min-width:58px;border:1px solid var(--line);border-radius:8px;background:#091827;color:#dbeafe;font-size:10px;font-weight:1000;cursor:pointer;padding:0 8px}
.orderBtn:hover{transform:scale(1.03)}
.autoOrderBtn{height:32px;min-width:70px;border:1px solid var(--line);border-radius:8px;font-size:10px;font-weight:1000;cursor:pointer;padding:0 8px}
.autoOrderBtn.on{background:#09351f;color:#20d66b;border-color:#20d66b}
.autoOrderBtn.off{background:#35111a;color:#ff405d;border-color:#ff405d}
.bottomTradeControls{display:flex;align-items:center;justify-content:center;gap:6px;flex-wrap:wrap;min-width:390px}
.foot{gap:10px}
@media(max-width:900px){
  .foot{height:auto;min-height:56px;flex-wrap:wrap;justify-content:center;padding:6px 10px;line-height:1.2}
  .bottomTradeControls{order:2;width:100%;min-width:0;gap:6px}
  .foot span:first-child{order:1}
  .foot span:last-child{order:3;width:100%;text-align:center}
  .bottomTradeControls .orderBtn{height:26px;min-width:54px;font-size:9px;padding:0 6px}
  .bottomTradeControls .autoOrderBtn{height:28px;min-width:68px;font-size:9px;padding:0 6px}
}


.regPackBtn{min-width:96px}
.regSummary{display:grid;grid-template-columns:repeat(5,minmax(90px,1fr));gap:8px;margin-bottom:10px}
.regCard{border:1px solid var(--line);border-radius:10px;padding:8px;background:rgba(45,140,255,.08);text-align:center}
.regCard small{display:block;color:var(--muted);font-weight:900;font-size:10px}
.regCard b{font-size:18px}
.regTable{width:100%;border-collapse:collapse;font-size:11px}
.regTable th,.regTable td{border-bottom:1px solid rgba(255,255,255,.08);padding:6px;text-align:left}
.regTable th{color:#8ea2bb;font-size:10px;text-transform:uppercase}
.regGood{color:#20d66b!important;font-weight:1000}.regBad{color:#ff405d!important;font-weight:1000}.regNeu{color:#f5b02e!important;font-weight:1000}
body.light-theme .regCard{background:#f4f8ff}
body.light-theme .regTable th,body.light-theme .regTable td{border-color:rgba(15,35,65,.10)}
@media(max-width:760px){.regSummary{grid-template-columns:repeat(2,minmax(0,1fr))}.regTable{font-size:10px}.regPackBtn{min-width:86px}}

.modalOverlay{position:fixed;inset:0;background:rgba(0,0,0,.62);z-index:99999;display:none;align-items:center;justify-content:center;padding:18px}
.modalBox{width:min(760px,94vw);max-height:82vh;overflow:auto;background:#091827;border:1px solid var(--line);border-radius:14px;box-shadow:0 20px 70px rgba(0,0,0,.45);color:var(--text)}
.modalHead{height:42px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid var(--line);font-weight:1000}
.modalClose{border:0;background:transparent;color:var(--text);font-size:22px;cursor:pointer}
.modalBody{padding:12px;font-size:12px;white-space:pre-wrap;word-break:break-word}
.modalBody pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:11px;line-height:1.35}
body.light-theme .orderBtn,body.light-theme .modalBox{background:#ffffff;color:#102033;border-color:var(--line)}
body.light-theme .autoOrderBtn.off{background:#fff0f2}
body.light-theme .autoOrderBtn.on{background:#eafff2}
@media(max-width:760px){.topRight{flex-wrap:wrap;justify-content:flex-start}.orderBtn{height:26px;min-width:52px;font-size:9px}.autoOrderBtn{height:28px;min-width:62px;font-size:9px}}

@media(max-width:1300px){
  html,body{height:auto;min-height:100%;overflow-x:hidden;overflow-y:auto}
  .app{height:auto;min-height:100vh;overflow:visible;display:flex;flex-direction:column;padding:6px;gap:6px}
  .topbar{grid-template-columns:1fr;align-items:stretch;height:auto;gap:8px;padding:8px}
  .brand{justify-content:flex-start}.marketStrip{height:auto;grid-template-columns:repeat(3,minmax(0,1fr))}.tickerCard{min-height:42px;border-bottom:1px solid var(--line)}
  .topRight{justify-content:space-between}.hitBox{font-size:14px}.hitBox b{font-size:24px}.live{font-size:12px}
  .toolbar{grid-template-columns:48px minmax(210px,292px) minmax(0,1fr);width:100%}.tfTabs{width:100%;min-width:0;overflow-x:auto;overflow-y:hidden;white-space:nowrap;-webkit-overflow-scrolling:touch}.tfTabs button{flex:0 0 auto}
  .grid{height:auto;min-height:0;grid-template-columns:1fr;overflow:visible}.left{height:auto;grid-template-rows:none;grid-template-columns:repeat(2,minmax(0,1fr));overflow:visible}.panel{min-height:126px;overflow:hidden}.rightMain{height:auto;grid-template-rows:auto 520px auto;overflow:visible}.kpiStrip{grid-template-columns:repeat(4,minmax(0,1fr))}.bottomCards{grid-template-columns:repeat(2,minmax(0,1fr))}.body{height:auto;min-height:150px;max-height:none;overflow:visible}.chartWrap{min-height:480px}.chart{min-height:480px}.error{position:static}.foot{height:auto;min-height:34px;gap:10px;flex-wrap:wrap}
}
@media(max-width:760px){
  .themeSwitch{width:28px;height:28px;min-width:28px;flex-basis:28px;font-size:12px;border-radius:7px}

  .app{padding:5px;gap:6px}.topbar{padding:7px}.brand{gap:8px}.logo{width:40px;height:40px;font-size:26px;border-radius:10px}.brand h1{font-size:15px}.brand p{font-size:16px}.marketStrip{grid-template-columns:repeat(2,minmax(0,1fr))}.tickerCard{padding:6px 8px}.tickerCard strong{font-size:13px}.tickerCard span{float:none;display:inline-block;margin-left:8px}.topRight{gap:8px}.hitBox{border-left:0;padding-left:0}.live{border-left:1px solid var(--line);padding-left:8px}
  .toolbar{grid-template-columns:1fr}.menu{display:none}.selectBox{width:100%;height:40px}.tfTabs{height:42px;padding:6px}.tfTabs button{min-width:54px;height:30px}.grid{gap:6px}.left{grid-template-columns:1fr}.panel{border-radius:9px}.aiSignal{padding:10px 14px}.signalLine{align-items:flex-start;flex-direction:column;gap:4px}.signalText{font-size:25px;white-space:normal}.assetBadge{font-size:13px;max-width:100%;white-space:normal}.tradeRow b{font-size:14px;text-align:right;white-space:normal}.donutBox{grid-template-columns:88px 1fr}.rightMain{grid-template-rows:auto 430px auto}.kpiStrip{grid-template-columns:repeat(2,minmax(0,1fr));height:auto}.kpi{min-height:52px}.chartTop{left:10px;right:10px;font-size:11px}.tvBadge{display:none}.chartWrap{min-height:400px;padding-top:32px}.chart{min-height:400px}.bottomCards{grid-template-columns:1fr}.newsSent{grid-template-columns:1fr 88px}.foot{align-items:flex-start;justify-content:flex-start;font-size:10px;padding:8px 10px}
}
@media(max-width:420px){
  .marketStrip{grid-template-columns:1fr}.topRight{align-items:flex-start}.hitBox b{font-size:22px}.rightMain{grid-template-rows:auto 390px auto}.chartWrap{min-height:360px}.chart{min-height:360px}.kpiStrip{grid-template-columns:1fr}.newsSent{grid-template-columns:1fr}.newsGaugeSvg{display:none}.gaugeSvg{width:176px}.donutBox{grid-template-columns:78px 1fr}.donut{width:76px;height:76px}.donut:after{inset:12px}.donut b{font-size:21px}
}

.rankBox{margin:10px 0 12px;padding:10px;border:1px solid var(--line);border-radius:10px;background:rgba(45,140,255,.08)}
.rankBox h3{margin:0 0 8px;font-size:14px;color:var(--text)}
.rankItem{display:grid;grid-template-columns:1fr 70px 80px 70px;gap:10px;align-items:center;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.06);font-size:12px}
.rankItem:last-child{border-bottom:0}
.rankItem small{font-weight:900;text-align:right}

</style>
</head>
<body>
<div class="app">
  <div class="refreshBar"><div class="refreshFill" id="refreshFill"></div></div>
  <header class="topbar">
    <div class="brand"><div class="logo">AI</div><div><h1>AI MARKETLENS</h1><p>By Syed Abbas</p></div></div>
    <div class="marketStrip" id="marketStrip"></div>
    <div class="topRight"><button class="themeSwitch" id="themeToggle" type="button" title="Switch theme"><span class="modeText" id="themeText">☀</span></button><div class="hitBox">Page Hits<br><b id="hitCount">--</b></div><div class="live">Live <span class="dot"></span><br><b id="clock">--</b></div></div>
  </header>
  <section class="toolbar">
    <div class="menu">MENU</div>
    <div class="selectBox"><label>Commodity</label><select id="assetSelect"></select></div>
    <div class="tfTabs" id="tfTabs"></div>
  </section>
  <div class="error" id="error"></div>
  <main class="grid">
    <aside class="left">
      <section class="panel"><div class="panelHead">AI SIGNAL <span class="info">i</span></div><div class="aiSignal"><div class="signalLine"><div class="signalText" id="signalText">LOADING</div><div class="assetBadge" id="assetBadge">GOLD</div></div><div class="confidenceLabel">Confidence</div><div class="confidence"><span id="confidenceValue">--</span>%</div><div class="bars" id="bars"></div></div></section>
      <section class="panel"><div class="panelHead">TRADE PLAN <span class="info">i</span></div><div class="tradeRows"><div class="tradeRow"><span class="pos">ENTRY / RANGE</span><b id="entryVal">--</b></div><div class="tradeRow"><span class="pos">TARGET</span><b id="targetVal">--</b></div><div class="tradeRow"><span class="neg">STOP LOSS</span><b id="stopVal">--</b></div><div class="tradeRow"><span style="color:#52a8ff">RISK REWARD</span><b id="rrVal">--</b></div></div></section>
      <section class="panel"><div class="panelHead">EXTREME ZONE <span class="info">i</span></div><div class="gaugeWrap"><svg class="gaugeSvg" viewBox="0 0 230 132"><path class="gaugeTrack" d="M25 110 A90 90 0 0 1 205 110"/><path class="gaugeGreen" d="M25 110 A90 90 0 0 1 72 31"/><path class="gaugeYellow" d="M72 31 A90 90 0 0 1 115 20"/><path class="gaugeOrange" d="M115 20 A90 90 0 0 1 158 31"/><path class="gaugeRed" d="M158 31 A90 90 0 0 1 205 110"/><line id="rsiNeedle" class="needle" x1="115" y1="110" x2="115" y2="42"/><circle cx="115" cy="110" r="5" fill="#fff"/><text id="rsiNum" class="gaugeNum" x="115" y="92">--</text><text id="rsiLabel" class="gaugeLabelText" x="115" y="126" fill="#f5b02e">--</text></svg></div></section>
      <section class="panel"><div class="panelHead">FUSION SCORE <span class="info">i</span></div><div class="donutBox"><div class="donut" id="fusionDonut"><b id="fusionNum">--</b></div><div class="legend"><div><span>Technical</span><b id="techPct">--</b></div><div><span>AI Sentiment</span><b id="aiPct">--</b></div><div><span>News & Macro</span><b id="newsPct">--</b></div><h3 id="fusionLabel" class="pos">--</h3></div></div></section>
    </aside>
    <section class="rightMain">
      <div class="kpiStrip" id="kpiStrip"></div>
      <section class="panel chartPanel"><div class="chartTop"><div id="chartTitle">--</div><div class="tvBadge">Trading View</div></div><div class="chartWrap"><div id="chart" class="chart"></div></div></section>
      <section class="bottomCards">
        <div class="panel"><div class="panelHead">TECHNICAL ANALYSIS <span class="info">i</span></div><div class="body" id="technicalBody"></div></div>
        <div class="panel"><div class="panelHead">AI STRATEGIST <span class="info">i</span></div><div class="body"><div class="aiText" id="aiSummary">Loading...</div><div class="recommend">RECOMMENDATION</div><div class="aiText" id="aiRisk">--</div><h3 class="pos">Confidence: <span id="aiConf">--</span>%</h3></div></div>
        <div class="panel"><div class="panelHead">NEWS ANALYSIS <span class="info">i</span></div><div class="body"><div class="newsSent"><div><small>News Sentiment</small><div class="newsScore" id="newsScore">--</div></div><svg class="newsGaugeSvg" viewBox="0 0 96 60"><path class="gaugeTrack" d="M10 50 A38 38 0 0 1 86 50" stroke-width="9"/><path class="gaugeGreen" d="M10 50 A38 38 0 0 1 35 15" stroke-width="9"/><path class="gaugeYellow" d="M35 15 A38 38 0 0 1 61 15" stroke-width="9"/><path class="gaugeRed" d="M61 15 A38 38 0 0 1 86 50" stroke-width="9"/><line id="newsNeedle" class="newsNeedle" x1="48" y1="50" x2="48" y2="21"/><circle cx="48" cy="50" r="3.5" fill="#fff"/><text id="newsGaugeNum" class="newsGaugeNum" x="48" y="42">--</text></svg></div><div id="newsList"></div></div></div>
        <div class="panel"><div class="panelHead">MARKET DRIVERS <span class="info">i</span></div><div class="body" id="driversBody"></div></div>
      </section>
    </section>
  </main>
  <footer class="foot"><span>Last Updated: <b id="updated">--</b></span><div class="bottomTradeControls"><button class="autoOrderBtn off" id="autoOrderToggle" type="button" title="Auto order">AUTO OFF</button><button class="orderBtn" data-popup="/api/login" type="button">Login</button><button class="orderBtn" data-popup="/api/account" type="button">Account</button><button class="orderBtn" data-popup="/api/positions" type="button">Position</button><button class="orderBtn" data-popup="/api/orders" type="button">Order</button><button class="orderBtn regPackBtn" id="regressionPackBtn" type="button">Regression Pack</button></div><span>Disclaimer: AI generated analysis only. Do your own research before investing.</span></footer>
</div>

  <div class="modalOverlay" id="popupOverlay">
    <div class="modalBox">
      <div class="modalHead"><span id="popupTitle">Information</span><button class="modalClose" id="popupClose" type="button">×</button></div>
      <div class="modalBody"><pre id="popupContent">Loading...</pre></div>
    </div>
  </div>
<script>
const ASSETS={GOLD:["GOLD","GOLD (XAUUSD)"],SILVER:["SILVER","SILVER"],WTI:["WTI","WTI OIL"],BRENT:["BRENT","BRENT CRUDE"],BTC:["BTC","BITCOIN"],USTEC100:["US100","US TECH 100"]};
const TFS=["1M","15M","30M","1H","1D"];let asset="GOLD",tf="15M",busy=false;let autoOrderEnabled=false,lastOrderKey="";
function currentTheme(){return localStorage.getItem('marketlens_theme')||'dark'}
function applyTheme(theme){
  const isLight=theme==='light';
  document.body.classList.toggle('light-theme',isLight);
  if(window.themeText)themeText.innerText=isLight?'🌙':'☀';
  try{if(window.chart&&window.Plotly)Plotly.Plots.resize('chart')}catch(e){}
}
function initTheme(){
  applyTheme(currentTheme());
  if(window.themeToggle){
    themeToggle.onclick=()=>{
      const next=document.body.classList.contains('light-theme')?'dark':'light';
      localStorage.setItem('marketlens_theme',next);
      applyTheme(next);
      try{loadData(true)}catch(e){}
    };
  }
}

async function refreshAutoOrderStatus(){
  try{
    const r=await fetch('/api/auto-order/status?_='+Date.now());
    const d=await r.json();
    autoOrderEnabled=!!d.enabled;
    if(window.autoOrderToggle){
      autoOrderToggle.innerText=autoOrderEnabled?'AUTO ON':'AUTO OFF';
      autoOrderToggle.classList.toggle('on',autoOrderEnabled);
      autoOrderToggle.classList.toggle('off',!autoOrderEnabled);
    }
  }catch(e){}
}
async function toggleAutoOrder(){
  try{
    const r=await fetch('/api/auto-order/toggle',{method:'POST'});
    const d=await r.json();
    autoOrderEnabled=!!d.enabled;
    refreshAutoOrderStatus();
    showPopup('Auto Order',d);
  }catch(e){showPopup('Auto Order Error',{error:e.message})}
}
function showPopup(title,data){
  popupTitle.innerText=title||'Information';
  popupContent.textContent=typeof data==='string'?data:JSON.stringify(data,null,2);
  popupOverlay.style.display='flex';
}
async function openApiPopup(title,url){
  showPopup(title,'Loading...');
  try{
    const r=await fetch(url+(url.includes('?')?'&':'?')+'_='+Date.now());
    const d=await r.json();
    showPopup(title,d);
  }catch(e){showPopup(title+' Error',{error:e.message})}
}
function showHtmlPopup(title,html){
  popupTitle.innerText=title||'Information';
  popupContent.innerHTML=html;
  popupOverlay.style.display='flex';
}
function regCls(v){return Number(v)>0?'regGood':Number(v)<0?'regBad':'regNeu'}
function renderRegressionPack(d){
  if(d.error)return `<div class="regBad">${d.error}</div>`;
  const s=d.summary||{};
  const cards=[
    ['Assets',s.assets_tested??0,'regNeu'],['Signals',s.total_signals??0,'regNeu'],
    ['Wins',s.total_wins??0,'regGood'],['Losses',s.total_losses??0,'regBad'],
    ['Win Rate',(s.overall_win_rate_percent??0)+'%',(Number(s.overall_win_rate_percent)>=65?'regGood':Number(s.overall_win_rate_percent)>=45?'regNeu':'regBad')],
    ['Total R',Number(s.total_r_multiple||0).toFixed(3)+'R',regCls(s.total_r_multiple||0)]
  ].map(x=>`<div class="regCard"><small>${x[0]}</small><b class="${x[2]}">${x[1]}</b></div>`).join('');
  const approved=(d.ranking||[]).filter(a=>a.auto_order_allowed).map(a=>`<span class="regPill regGood">✓ ${a.asset_key}</span>`).join('')||'<span class="regPill regBad">No asset approved</span>';
  const watch=(d.ranking||[]).filter(a=>a.health==='AMBER'||String(a.health||'').includes('LOW SAMPLE')).map(a=>`<span class="regPill regNeu">• ${a.asset_key}</span>`).join('')||'<span class="regPill">None</span>';
  const disabled=(d.ranking||[]).filter(a=>a.health==='RED'||a.health==='INSUFFICIENT').map(a=>`<span class="regPill regBad">× ${a.asset_key}</span>`).join('')||'<span class="regPill">None</span>';
  const decisionBox=`<div class="rankBox"><h3>🟢 Auto Trade Approved</h3><div class="regPills">${approved}</div><h3>🟡 Watchlist</h3><div class="regPills">${watch}</div><h3>🔴 Disabled</h3><div class="regPills">${disabled}</div></div>`;
  const ranking=(d.ranking||[]).slice(0,6).map(a=>{
    const wr=Number(a.win_rate_percent||0), tr=Number(a.total_r||0), ac=Number(a.avg_confidence||0);
    const cls=wr>=80?'regGood':wr>=45?'regNeu':'regBad';
    return `<div class="rankItem"><b>#${a.rank||'-'} ${a.asset_key}</b><span class="${cls}">${wr}%</span><span class="${regCls(tr)}">${tr.toFixed(3)}R</span><span>${ac.toFixed(1)}%</span><small>${a.health||''}</small></div>`
  }).join('');
  const rows=(d.assets||[]).sort((a,b)=>(a.rank||99)-(b.rank||99)).map(a=>{
    const wr=Number(a.win_rate_percent||0), tr=Number(a.total_r||0), ac=Number(a.avg_confidence||0);
    const wrCls=wr>=80?'regGood':wr>=45?'regNeu':'regBad';
    const healthCls=(a.health==='GREEN')?'regGood':((a.health==='AMBER'||String(a.health||'').includes('LOW SAMPLE'))?'regNeu':'regBad');
    return `<tr><td><b>#${a.rank||'-'} ${a.asset_key}</b><br><small>${a.name||''}</small></td><td>${a.signals}</td><td class="regGood">${a.wins}</td><td class="regBad">${a.losses}</td><td>${a.stops}</td><td>${a.forced}</td><td class="${wrCls}">${wr}%</td><td class="${regCls(tr)}">${tr.toFixed(3)}R</td><td>${ac.toFixed(1)}%</td><td class="${healthCls}">${a.health||'RED'}</td><td>${a.tradable?'YES':'NO'}</td></tr>`
  }).join('');
  return `<div class="regSummary">${cards}</div>${decisionBox}<div class="rankBox"><h3>🏆 Asset Ranking Today - Production Gate</h3>${ranking}</div><div style="margin:6px 0 10px;color:var(--muted);font-size:11px">${d.note||''}<br>Timeframe: <b>${d.timeframe}</b> | Window: <b>${d.hours} hours</b> | Min Confidence: <b>${s.min_confidence??80}</b> | Auto-trade requires: <b>Win Rate ≥ 80% + Total R ≥ ${(s.min_total_r??1)} + Avg Confidence ≥ ${(s.min_avg_confidence??85)} + (5 Signals OR 3 Wins)</b></div><table class="regTable"><thead><tr><th>Commodity</th><th>Signals</th><th>Win</th><th>Loss</th><th>Stops</th><th>Forced</th><th>Win Rate</th><th>Total R</th><th>Avg Conf</th><th>Health</th><th>Tradable</th></tr></thead><tbody>${rows}</tbody></table>`;
}
async function openRegressionPack(){
  showPopup('Regression Pack','Running 7-day asset health regression...');
  try{
    const r=await fetch(`/api/regression/pack?tf=30M&hours=168&min_confidence=80&_=${Date.now()}`);
    const d=await r.json();
    showHtmlPopup('Regression Pack - Last 7 Days',renderRegressionPack(d));
  }catch(e){showPopup('Regression Pack Error',{error:e.message})}
}
async function tryAutoOrder(d){
  if(!autoOrderEnabled)return;
  const sig=d?.fusion?.signal||'';
  if(!['BUY SIGNAL','SELL SIGNAL'].includes(sig))return;
  const key=[asset,'30M',sig,d?.tech?.entry,d?.tech?.target,d?.tech?.stop].join('|');
  if(key===lastOrderKey)return;
  lastOrderKey=key;
  try{
    const r=await fetch(`/api/place-order?asset=${asset}&tf=30M&execute=true&_=${Date.now()}`);
    const res=await r.json();
    if(res.allowed||res.order)showPopup('Auto Order Result',res);
    else console.log('Auto order guard:',res.guard||res);
  }catch(e){console.log('Auto order error',e)}
}

function init(){initTheme();refreshAutoOrderStatus();if(window.autoOrderToggle)autoOrderToggle.onclick=toggleAutoOrder;if(window.regressionPackBtn)regressionPackBtn.onclick=openRegressionPack;if(window.popupClose)popupClose.onclick=()=>popupOverlay.style.display='none';if(window.popupOverlay)popupOverlay.onclick=e=>{if(e.target===popupOverlay)popupOverlay.style.display='none'};document.querySelectorAll('[data-popup]').forEach(b=>b.onclick=()=>openApiPopup(b.innerText,b.dataset.popup));Object.keys(ASSETS).forEach(k=>assetSelect.innerHTML+=`<option value="${k}">${ASSETS[k][1]}</option>`);assetSelect.value=asset;assetSelect.onchange=()=>{asset=assetSelect.value;loadData(true)};tfTabs.innerHTML=TFS.map(t=>`<button data-tf="${t}">${t}</button>`).join('');tfTabs.onclick=e=>{if(e.target.dataset.tf){tf=e.target.dataset.tf;setTabs();loadData(true)}};setTabs();updateAssetBadge();setInterval(()=>clock.innerText=new Date().toLocaleTimeString(),1000);fetch('/api/hit').then(r=>r.json()).then(x=>{hitCount.innerText=x.hits??'--'}).catch(()=>{});loadStrip();loadData(true);setInterval(()=>loadData(false),30000);setInterval(loadStrip,60000)}
function setTabs(){document.querySelectorAll('#tfTabs button').forEach(b=>b.classList.toggle('active',b.dataset.tf===tf))}
function fmt(x){let n=Number(String(x??'').replace(/,/g,''));return isFinite(n)?n.toLocaleString(undefined,{minimumFractionDigits:3,maximumFractionDigits:3}):'--'}
function zoneText(x){return String(x??'--').split(' - ').map(fmt).join(' - ')}
function pct(x){let n=Number(x||0);return (n>=0?'+':'')+n.toFixed(2)+'%'}
function setBusy(v){busy=v;error.innerText=v?'Loading live data...':''}
async function loadStrip(){try{const r=await fetch('/api/market-strip?_='+Date.now());const d=await r.json();marketStrip.innerHTML=(d.items||[]).map(it=>{if(it.error)return `<div class="tickerCard"><b>${it.asset_key}</b><strong>--</strong><span class="neu">ERR</span></div>`;let c=Number(it.change?.percent||0);return `<div class="tickerCard"><b>${it.name}</b><strong>${fmt(it.price)}</strong><span class="${c>=0?'pos':'neg'}">${pct(c)}</span></div>`}).join('')}catch(e){}}
async function loadData(manual=false){if(busy&&!manual)return;setBusy(true);try{if(refreshFill){refreshFill.style.animation='none';refreshFill.offsetHeight;refreshFill.style.animation='refreshRun 30s linear infinite';}const r=await fetch(`/api/signal?asset=${asset}&tf=${tf}&_=${Date.now()}`);const d=await r.json();if(d.error)throw new Error(d.error);render(d)}catch(e){error.innerText='Error: '+e.message}finally{setBusy(false)}}
function labelColor(label){return label==='BULLISH'?'#20d66b':label==='BEARISH'?'#ff405d':'#f5b02e'}
function rsiZone(r){if(r>=80)return ['EXTREME OVERBOUGHT','#ff405d'];if(r>=70)return ['OVERBOUGHT','#ff405d'];if(r<=20)return ['EXTREME OVERSOLD','#20d66b'];if(r<=30)return ['OVERSOLD','#20d66b'];return ['NORMAL','#f5b02e']}
function rr(entry,target,stop){entry=Number(entry);target=Number(target);stop=Number(stop);let risk=Math.abs(entry-stop),reward=Math.abs(target-entry);return risk>0?'1 : '+(reward/risk).toFixed(1):'--'}
function assetColor(k){return {GOLD:'#FFD700',SILVER:'#C0C0C0',BTC:'#ff405d',WTI:'#ff9900',BRENT:'#ffd000',USTEC100:'#52a8ff'}[k]||'#ffffff'}
function updateAssetBadge(){if(window.assetBadge){assetBadge.innerText=ASSETS[asset]?.[1]||asset;assetBadge.style.color=assetColor(asset)}}
function render(d){updateAssetBadge();let label=d.fusion.label||'NEUTRAL',col=labelColor(label),conf=Number(d.fusion.confidence||0),fusion=Math.round(Math.abs(Number(d.fusion.fusion||0))),rsi=Number(d.tech.rsi||50);let [rz,rzCol]=rsiZone(rsi);signalText.innerText=(d.fusion.signal||'HOLD').replace(' SIGNAL','');signalText.style.color=col;confidenceValue.innerText=Math.round(conf);bars.innerHTML=Array.from({length:18},(_,i)=>`<i style="height:${5+i}px;background:${i<Math.round(conf/100*18)?col:'#1f3048'}"></i>`).join('');entryVal.innerText=zoneText(d.tech.entry_zone||d.tech.entry);targetVal.innerText=fmt(d.tech.target);stopVal.innerText=fmt(d.tech.stop);rrVal.innerText=rr(d.tech.entry,d.tech.target,d.tech.stop);rsiNum.textContent=Math.round(rsi);rsiLabel.textContent=rz;rsiLabel.setAttribute('fill',rzCol);rsiNeedle.style.transform=`rotate(${Math.max(-90,Math.min(90,(rsi/100*180)-90))}deg)`;fusionNum.innerText=fusion;fusionLabel.innerText=label;fusionLabel.style.color=col;let techP=Math.round(d.fusion.tech_percent||0),aiP=Math.round(d.fusion.ai_percent||0),newsP=Math.round(d.fusion.news_percent||0);techPct.innerText=techP+'%';aiPct.innerText=aiP+'%';newsPct.innerText=newsP+'%';fusionDonut.style.background=`conic-gradient(#20d66b 0deg ${techP*3.6}deg,#a855f7 ${techP*3.6}deg ${(techP+aiP)*1.8}deg,#21d4ff ${(techP+aiP)*1.8}deg 360deg)`;updated.innerText=d.updated;chartTitle.innerText=`${d.asset.name} - ${d.tf} - Capital.com - Signal fixed on ${d.signal_tf}`;
let change=d.change||{percent:0,value:0};kpiStrip.innerHTML=[['PRICE',fmt(d.tech.price),'pos'],['24H CHANGE',pct(Number(change.percent||0)),Number(change.percent||0)>=0?'pos':'neg'],['VOLUME',d.chart?.data?.[0]?.close?.length||0,''],['ATR (14)',fmt(d.tech.atr),''],['RSI (14)',rsi.toFixed(2),rsi>=70||rsi<=30?'neg':''],['VOLATILITY',(d.tech.regime?.atr_pct||0).toFixed(2)+'%',''],['TREND',label,label==='BEARISH'?'neg':'pos']].map(x=>`<div class="kpi"><small>${x[0]}</small><b class="${x[2]}">${x[1]}</b></div>`).join('');
let layout=d.chart.layout||{};let isLight=document.body.classList.contains('light-theme');let chartBg=isLight?'#ffffff':'#081523';let chartFont=isLight?'#102033':'#cbd5e1';let gridCol=isLight?'rgba(15,35,65,.12)':'rgba(148,163,184,.12)';let axisCol=isLight?'#c9d6e8':'#203047';layout.template=isLight?'plotly_white':'plotly_dark';layout.paper_bgcolor='rgba(0,0,0,0)';layout.plot_bgcolor=chartBg;layout.font={color:chartFont,size:10};layout.margin={l:56,r:68,t:18,b:58};layout.height=null;layout.autosize=true;layout.legend={orientation:'h',y:1.08,x:0,font:{size:10,color:chartFont}};layout.xaxis={...(layout.xaxis||{}),gridcolor:gridCol,linecolor:axisCol,fixedrange:true,rangebreaks:['1M','15M','30M','1H'].includes(d.tf)?[{bounds:['sat','mon']}]:[]};layout.yaxis={...(layout.yaxis||{}),gridcolor:gridCol,linecolor:axisCol,fixedrange:true};
let baseData=(d.chart.data||[]).slice();
let xs=(baseData[0]&&baseData[0].x)||[];let x0=xs[0],x1=xs[xs.length-1];
let levels=[['TARGET',Number(d.tech.target),'#20d66b','triangle-up'],['ENTRY',Number(d.tech.entry),'#2d8cff','circle'],['STOP LOSS',Number(d.tech.stop),'#ff405d','triangle-down']].filter(v=>isFinite(v[1]));
let candleY=[];try{(baseData[0].high||[]).forEach(x=>candleY.push(Number(x)));(baseData[0].low||[]).forEach(x=>candleY.push(Number(x)));}catch(e){}
levels.forEach(v=>candleY.push(v[1]));let ys=candleY.filter(x=>isFinite(x));if(ys.length){let ymin=Math.min(...ys),ymax=Math.max(...ys),pad=Math.max((ymax-ymin)*0.10,Math.abs(ymax)*0.001);layout.yaxis={...(layout.yaxis||{}),range:[ymin-pad,ymax+pad],autorange:false};}
levels.forEach(v=>{baseData.push({type:'scatter',mode:'lines',x:[x0,x1],y:[v[1],v[1]],name:v[0],line:{color:v[2],width:1.8,dash:'dash'},hoverinfo:'skip'});baseData.push({type:'scatter',mode:'markers+text',x:[x1],y:[v[1]],name:v[0]+' POINT',marker:{color:v[2],size:11,symbol:v[3]},text:[v[0]],textposition:'middle right',textfont:{color:v[2],size:12},showlegend:false,cliponaxis:false,hovertemplate:v[0]+': %{y:.3f}<extra></extra>'})});
Plotly.react('chart',baseData,layout,{displayModeBar:false,responsive:true,scrollZoom:false});setTimeout(()=>Plotly.Plots.resize('chart'),120);
let techTrend=label==='BULLISH'?'Bullish':label==='BEARISH'?'Bearish':'Normal';let techRows=[['RSI (14)',rz,rsi>=70?'neg':rsi<=30?'pos':'neu'],['MACD',d.tech.macd>=0?'Bullish':'Bearish',d.tech.macd>=0?'pos':'neg'],['EMA Trend',techTrend,label==='BULLISH'?'pos':label==='BEARISH'?'neg':'neu'],['ATR',Number(d.tech.regime?.atr_pct||0)>0.8?'High Volatility':'Normal',Number(d.tech.regime?.atr_pct||0)>0.8?'neu':'pos'],['Support','Bullish','pos'],['Resistance',Number(d.tech.price||0)>Number(d.tech.resistance||0)?'Bullish':'Normal',Number(d.tech.price||0)>Number(d.tech.resistance||0)?'pos':'neu']];technicalBody.innerHTML=techRows.map(x=>`<div class="driver"><div class="driverIcon">${x[0].slice(0,3).toUpperCase()}</div><span>${x[0]}</span><b class="${x[2]}">${x[1]}</b></div>`).join('');
aiSummary.innerText=d.ai.summary||'AI summary unavailable.';aiRisk.innerText=d.ai.risk||'No major AI risk returned.';aiConf.innerText=Math.round(conf);newsScore.innerText=(d.news_score>=0?'+':'')+Math.round(d.news_score)+'  '+(d.news_score>=0?'BULLISH':'BEARISH');newsScore.className='newsScore '+(d.news_score>=0?'pos':'neg');let ns=Math.max(-100,Math.min(100,Number(d.news_score||0)));newsGaugeNum.textContent=(ns>=0?'+':'')+Math.round(ns);newsNeedle.style.transform=`rotate(${ns/100*90}deg)`;newsList.innerHTML=(d.news||[]).slice(0,5).map((n,i)=>`<div class="newsItem"><span>${i+1}</span><a href="${n.url||'#'}" target="_blank">${n.headline||''}</a><b class="${i<3?'pos':'neu'}">●</b></div>`).join('')||'<div class="newsItem"><a>No matching live news returned.</a></div>';let er=d.institutional?.event_risk?.risk_penalty||0;let mtf=d.institutional?.mtf?.alignment||0;driversBody.innerHTML=[['USD','USD / Macro',d.news_score>=0?'Bullish':'Bearish',d.news_score>=0?'pos':'neg'],['MTF','MTF Alignment',mtf>0?'Bullish':mtf<0?'Bearish':'Neutral',mtf>0?'pos':mtf<0?'neg':'neu'],['EVT','Event Risk',er>6?'Elevated':'Normal',er>6?'neg':'pos'],['REG','Market Regime',d.tech.regime?.regime||'Normal','neu'],['NEW','News Flow',d.ai.bias||'Neutral',(d.ai.bias||'').includes('BEAR')?'neg':'pos'],['BT','Backtest Win Rate',Math.round((d.tech.backtest?.win_rate||.5)*100)+'%','pos']].map(x=>`<div class="driver"><div class="driverIcon">${x[0]}</div><span>${x[1]}</span><b class="${x[3]}">${x[2]}</b></div>`).join('');tryAutoOrder(d)}
window.addEventListener('resize',()=>{try{Plotly.Plots.resize('chart')}catch(e){}});init();
</script>
</body>
</html>
"""
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=False,
    )
