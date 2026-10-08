"""
UNPLLM - Live data tools (free, keyless public APIs)

Questions about real-time data (weather, air quality, exchange rates)
can't be answered from a static knowledge base, so main.py tries these
tools first. Design choices worth stating in your report:

  * Routing is a cheap keyword/regex gate, NOT an LLM call — a 3.8B model
    on CPU is too slow and too unreliable to classify every query.
  * Results are formatted directly instead of being paraphrased by the
    LLM, so numbers can never be hallucinated and answers are instant.
  * Offline-first behaviour: the last successful reading is cached with a
    timestamp. If the internet is down, you get "last known reading,
    N min ago" instead of an error — or an honest "unavailable" if
    nothing was ever saved.
  * Live data is never written into the vector store (it goes stale
    within hours); it lives in a small JSON cache instead.

APIs (all free, no API key):
  Open-Meteo        weather + geocoding + air quality (non-commercial use)
  Frankfurter       currency conversion (ECB daily reference rates)

Quick self-test on your own machine (needs internet):
    python3 live_tools.py
"""

import json
import re
import time
import requests

LIVE_TOOLS_ENABLED = True  # set False to switch every live tool off instantly
CACHE_FILE = "./live_cache.json"
TIMEOUT_SECONDS = 6
HEADERS = {"User-Agent": "UNPLLM-FinalYearProject/1.0 (student project)"}

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FX_URL = "https://api.frankfurter.dev/v1/latest"

# ---------------------------------------------------------------- cache

def _cache_load():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _cache_get(key):
    return _cache_load().get(key)


def _cache_set(key, text):
    cache = _cache_load()
    cache[key] = {"text": text, "fetched_at": time.time()}
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass  # cache is a nice-to-have, never worth crashing over


def _age(timestamp):
    secs = int(time.time() - timestamp)
    if secs < 90:
        return "moments ago"
    mins = secs // 60
    if mins < 60:
        return f"{mins} min ago"
    hours = mins // 60
    if hours < 48:
        return f"{hours} h ago"
    return f"{hours // 24} days ago"


def _fmt(n):
    return f"{n:,.2f}".rstrip("0").rstrip(".")

# ------------------------------------------------------- city extraction

_PLACE_AFTER_PREP = re.compile(
    r"\b(?:in|at|for|of|near)\s+([A-Za-z][A-Za-z .'\-]*?)"
    r"(?:\s+(?:today|now|right now|currently|tomorrow|tonight|please|pls|thanks|this\s+\w+))?$",
    re.IGNORECASE,
)

_NOISE = set("""
weather temperature forecast humidity raining rainfall air quality aqi pollution level levels
index what whats what's how hows how's is are was be can the a an in at of for near today now
currently current right tonight tomorrow like tell me show give get check please pls does do
did form forms work works happen happens cause causes change changes affect why when which
you your i it there here outside and or about
""".split())

_NOT_PLACES = {"general", "detail", "summer", "winter", "monsoon", "spring", "autumn",
               "today", "tomorrow", "tonight", "now", "the world", "earth", "space"}


def _city_from_remainder(query):
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", query)
    rest = [w for w in words if w.lower() not in _NOISE]
    return " ".join(rest) if 1 <= len(rest) <= 3 else None


def extract_city(query):
    """Pulls a place name out of a question. Returns None if there isn't one."""
    q = query.strip().rstrip("?!. ").strip()
    m = _PLACE_AFTER_PREP.search(q)
    city = m.group(1).strip() if m else _city_from_remainder(q)
    if not city:
        return None
    city = re.sub(r"^the\s+", "", city, flags=re.IGNORECASE).strip(" .'-")
    if not (2 <= len(city) <= 40) or any(ch.isdigit() for ch in city):
        return None
    if city.lower() in _NOT_PLACES:
        return None
    return city

# ----------------------------------------------------------------- tools

_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


def _geocode(city):
    r = requests.get(GEOCODE_URL, params={"name": city, "count": 1},
                     headers=HEADERS, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        raise LookupError(f"no place called {city!r}")
    top = results[0]
    return top["latitude"], top["longitude"], top.get("name", city), top.get("country", "")


def _place_label(name, country):
    return f"{name}, {country}" if country else name


def _weather(city):
    lat, lon, name, country = _geocode(city)
    r = requests.get(WEATHER_URL, params={
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
    }, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    data = r.json()
    cur, units = data["current"], data.get("current_units", {})
    desc = _WMO.get(cur.get("weather_code"), "unknown conditions")
    return (
        f"Current weather in {_place_label(name, country)}: {desc}, "
        f"{cur['temperature_2m']}{units.get('temperature_2m', '°C')}, "
        f"humidity {cur['relative_humidity_2m']}%, "
        f"wind {cur['wind_speed_10m']} {units.get('wind_speed_10m', 'km/h')} "
        f"(as of {cur['time']} local time; source: Open-Meteo)."
    )


def _aqi_category(aqi):
    for limit, label in ((50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"),
                         (200, "Unhealthy"), (300, "Very unhealthy")):
        if aqi <= limit:
            return label
    return "Hazardous"


def _air_quality(city):
    lat, lon, name, country = _geocode(city)
    r = requests.get(AIR_URL, params={
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "current": "us_aqi,pm2_5,pm10",
    }, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    cur = r.json()["current"]
    aqi, pm25, pm10 = cur.get("us_aqi"), cur.get("pm2_5"), cur.get("pm10")
    if aqi is None:
        raise LookupError("no air-quality data for this location")
    parts = [f"US AQI {aqi} ({_aqi_category(aqi)})"]
    if pm25 is not None:
        parts.append(f"PM2.5 {pm25} µg/m³")
    if pm10 is not None:
        parts.append(f"PM10 {pm10} µg/m³")
    return (
        f"Air quality in {_place_label(name, country)}: {', '.join(parts)} "
        f"(as of {cur['time']} local time; model-based estimate from Open-Meteo)."
    )

# currency --------------------------------------------------------------

_CURRENCIES = {
    "us dollar": "USD", "us dollars": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "inr": "INR", "rupee": "INR", "rupees": "INR",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "jpy": "JPY", "yen": "JPY",
    "aud": "AUD", "cad": "CAD", "chf": "CHF", "sgd": "SGD",
    "cny": "CNY", "yuan": "CNY",
}
_CURRENCY_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in _CURRENCIES), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_CONNECTOR_RE = re.compile(r"\b(to|into|in|vs|versus)\b", re.IGNORECASE)
_RATE_WORD_RE = re.compile(r"\b(rate|exchange|convert|conversion)\b", re.IGNORECASE)


def _parse_currency(query):
    """Returns (amount, from_code, to_code) or None if this isn't a conversion question."""
    matches = list(_CURRENCY_RE.finditer(query))
    codes = []
    for m in matches:
        code = _CURRENCIES[m.group(1).lower()]
        if not codes or codes[-1] != code:
            codes.append(code)
    if not codes:
        return None

    before_first = query[:matches[0].start()]
    num = re.search(r"(\d[\d,]*(?:\.\d+)?)", before_first)
    amount = float(num.group(1).replace(",", "")) if num else 1.0

    if len(codes) >= 2:
        if not (_CONNECTOR_RE.search(query) or _RATE_WORD_RE.search(query) or num):
            return None
        return amount, codes[0], codes[1]
    if _RATE_WORD_RE.search(query):  # e.g. "dollar rate today" -> against INR
        return amount, codes[0], ("USD" if codes[0] == "INR" else "INR")
    return None


def _currency(amount, frm, to):
    r = requests.get(FX_URL, params={"base": frm, "symbols": to},
                     headers=HEADERS, timeout=TIMEOUT_SECONDS)
    if r.status_code in (404, 422):
        raise LookupError("unsupported currency")
    r.raise_for_status()
    data = r.json()
    rate = data["rates"][to]
    return (
        f"{_fmt(amount)} {frm} = {_fmt(amount * rate)} {to} "
        f"(rate {rate:.4f}; European Central Bank daily reference rate dated {data['date']} — "
        f"published once per working day, not a live market quote; source: Frankfurter)."
    )

# -------------------------------------------------------------- dispatch

_WEATHER_RE = re.compile(r"\b(weather|temperature|forecast|humidity|raining|rainfall)\b", re.IGNORECASE)
_AQI_RE = re.compile(r"\b(air quality|aqi|air pollution|pollution level)\b", re.IGNORECASE)


def _run(tool, label, key, fetch):
    try:
        text = fetch()
    except LookupError:
        return None  # not a real place / unsupported currency: let normal RAG handle it
    except requests.RequestException:
        cached = _cache_get(key)
        if cached:
            return {"tool": tool, "status": "cached", "text": (
                f"I couldn't reach the live {label} service (you may be offline), so here is "
                f"the last reading I saved ({_age(cached['fetched_at'])}): {cached['text']}")}
        return {"tool": tool, "status": "unavailable", "text": (
            f"I couldn't reach the live {label} service — you may be offline — "
            f"and I don't have an earlier reading saved for this.")}
    except (KeyError, ValueError, TypeError, IndexError):
        return None  # unexpected response shape: fall back to normal RAG
    _cache_set(key, text)
    return {"tool": tool, "status": "live", "text": text}


def try_live_tools(query):
    """
    Returns None if no live tool applies to this question. Otherwise:
      {"tool": "weather" | "air_quality" | "currency",
       "status": "live" | "cached" | "unavailable",
       "text": "<ready-to-show answer>"}
    Never raises — a bug here must not be able to break /generate_rag.
    """
    if not LIVE_TOOLS_ENABLED:
        return None
    try:
        if _AQI_RE.search(query):
            city = extract_city(query)
            if city:
                return _run("air_quality", "air-quality", f"air:{city.lower()}",
                            lambda: _air_quality(city))

        parsed = _parse_currency(query)
        if parsed:
            amount, frm, to = parsed
            return _run("currency", "exchange-rate", f"fx:{amount}:{frm}:{to}",
                        lambda: _currency(amount, frm, to))

        if _WEATHER_RE.search(query):
            city = extract_city(query)
            if city:
                return _run("weather", "weather", f"weather:{city.lower()}",
                            lambda: _weather(city))
    except Exception:
        return None
    return None


if __name__ == "__main__":
    tests = [
        "what's the weather in Delhi today",
        "air quality in Delhi",
        "convert 100 usd to inr",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        print("->", try_live_tools(q))
