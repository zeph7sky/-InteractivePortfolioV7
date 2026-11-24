"""
Portfolio Compounding Simulator - API (FastAPI)

Run locally:   uvicorn api:app --reload
Run on Render: uvicorn api:app --host 0.0.0.0 --port $PORT

Needs etfs.csv in the same folder, with columns: Ticker, ETF Name, Expense Ratio (e.g. "0.0400%").
"""
import math
import os
import threading
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------------------------------------------
CSV_PATH = Path(__file__).parent / "etfs.csv"
FIRST_MONTH = pd.Timestamp(2006, 1, 1)      # earliest month the app supports
MAX_FUNDS = 10                              # most funds one portfolio can hold
PRICE_TTL_SECONDS = 24 * 60 * 60            # reuse downloaded prices for one day

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# --------------------------------------------------------------------------------------
# 2. LOAD THE ETF LIST (etfs.csv)
# --------------------------------------------------------------------------------------
def load_catalog():
    """Read etfs.csv into {ticker: {"name", "fee", "label"}}. fee is a decimal (0.0004) or None."""
    df = pd.read_csv(CSV_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = {"Ticker", "ETF Name", "Expense Ratio"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"etfs.csv is missing column(s): {', '.join(sorted(missing_cols))}")

    df = df.dropna(subset=["Ticker", "ETF Name"]).copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df[df["Ticker"].str.fullmatch(r"[A-Z0-9.\-]+") & (df["Ticker"] != "TICKER")]
    df = df.drop_duplicates(subset="Ticker")

    fee_pct = pd.to_numeric(
        df["Expense Ratio"].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce"
    )

    catalog = {}
    for ticker, name, fee in zip(df["Ticker"], df["ETF Name"].astype(str).str.strip(), fee_pct):
        catalog[ticker] = {
            "name": name,
            "fee": None if pd.isna(fee) else float(fee) / 100.0,
            "label": f"{ticker} — {name}",
        }
    return catalog

CATALOG = load_catalog()

# --------------------------------------------------------------------------------------
# 3. MONTH HELPERS
# --------------------------------------------------------------------------------------
def last_full_month():
    this_month = pd.Timestamp.today().normalize().replace(day=1)
    return this_month - pd.DateOffset(months=1)

def month_labels():
    months = pd.date_range(FIRST_MONTH, last_full_month(), freq="MS")
    return [m.strftime("%m/%Y") for m in months], {m.strftime("%m/%Y"): m for m in months}

# --------------------------------------------------------------------------------------
# 4. PRICE DOWNLOAD WITH A SIMPLE ONE-DAY CACHE
# --------------------------------------------------------------------------------------
_price_cache = {}
_price_lock = threading.Lock()

def fetch_closes(tickers):
    now = time.time()
    with _price_lock:
        stale = [t for t in tickers if t not in _price_cache or now - _price_cache[t][0] > PRICE_TTL_SECONDS]
        if stale:
            raw = yf.download(stale, start="2005-12-01", interval="1mo", auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                raise RuntimeError("Yahoo Finance returned no data.")

            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"].copy()
            else:
                closes = raw[["Close"]].rename(columns={"Close": stale[0]})

            closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).to_period("M").to_timestamp()
            closes = closes[~closes.index.duplicated(keep="last")].sort_index()

            for t in stale:
                if t in closes.columns and not closes[t].dropna().empty:
                    _price_cache[t] = (now, closes[t])

        return pd.DataFrame({t: _price_cache[t][1] for t in tickers if t in _price_cache})

# --------------------------------------------------------------------------------------
# 5. REQUEST SHAPE
# --------------------------------------------------------------------------------------
class FundIn(BaseModel):
    ticker: str
    pct: float = Field(ge=0, le=100)

class SimulationRequest(BaseModel):
    initial: float = Field(0, ge=0, le=10_000_000)
    monthly: float = Field(0, ge=0, le=100_000)
    start: str | None = None
    end: str | None = None
    rebalance: bool = True
    funds: list[FundIn] = Field(min_length=1, max_length=MAX_FUNDS)

def bad_request(message):
    return HTTPException(status_code=400, detail=message)

def num(x, digits=2):
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), digits)

# --------------------------------------------------------------------------------------
# 6. THE APP
# --------------------------------------------------------------------------------------
app = FastAPI(title="Portfolio Compounding Simulator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/funds")
def list_funds():
    labels, _ = month_labels()
    funds = [
        {"ticker": t, "name": info["name"], "label": info["label"], "expense_ratio": info["fee"]}
        for t, info in sorted(CATALOG.items())
    ]
    return {"funds": funds, "months": labels, "max_funds": MAX_FUNDS}

@app.post("/simulate")
def simulate(req: SimulationRequest):
    tickers_in = [f.ticker.strip().upper() for f in req.funds]

    if req.initial == 0 and req.monthly == 0:
        raise bad_request("Please enter at least $1 in either the initial investment or the monthly contribution.")

    seen = set()
    for t in tickers_in:
        if t not in CATALOG:
            raise bad_request(f"'{t}' is not in the supported fund list.")
        if t in seen:
            raise bad_request(f"Duplicate fund: {t} is listed more than once. Please pick a different ETF.")
        seen.add(t)

    total_pct = round(sum(f.pct for f in req.funds), 2)
    if abs(total_pct - 100.0) > 0.005:
        diff = round(abs(100.0 - total_pct), 2)
        if total_pct < 100:
            raise bad_request(f"Your allocation adds up to {total_pct}%. Please add {diff}% more to reach 100%.")
        raise bad_request(f"Your allocation adds up to {total_pct}%. Please reduce it by {diff}% to equal 100%.")

    active_portfolio = {t: f.pct for t, f in zip(tickers_in, req.funds) if f.pct > 0}
    active_tickers = list(active_portfolio)

    labels, label_to_ts = month_labels()
    last_month = last_full_month()

    try:
        prices = fetch_closes(sorted(active_tickers))
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=("Couldn't download price history from Yahoo Finance. This can be a temporary outage or "
                    f"rate limiting. Please try again in a minute. ({e})"),
        )

    no_data = [t for t in active_tickers if t not in prices.columns]
    if no_data:
        raise HTTPException(
            status_code=404,
            detail=f"Yahoo Finance returned no price history for: {', '.join(no_data)}. Please pick a different fund.",
        )

    price_months = pd.date_range("2005-12-01", last_month, freq="MS")
    prices = prices[active_tickers].reindex(price_months)
    returns_df = prices.pct_change(fill_method=None).loc[FIRST_MONTH:]

    first_valid = {t: returns_df[t].first_valid_index() for t in active_tickers}
    unusable = [t for t, d in first_valid.items() if d is None]
    if unusable:
        raise bad_request(f"{', '.join(unusable)} has no usable monthly returns. Please pick a different fund.")

    common_start = max(first_valid.values())

    start_requested = None
    if req.start:
        if req.start not in label_to_ts:
            raise bad_request(f"Invalid start month '{req.start}'. Use MM/YYYY between {labels[0]} and {labels[-1]}.")
        start_requested = label_to_ts[req.start]
    if req.end and req.end not in label_to_ts:
        raise bad_request(f"Invalid end month '{req.end}'. Use MM/YYYY between {labels[0]} and {labels[-1]}.")

    start_ts = common_start if (start_requested is None or start_requested <= common_start) else start_requested
    end_ts = label_to_ts[req.end] if req.end else last_month

    if end_ts < common_start:
        raise bad_request(
            f"The end month ({end_ts:%m/%Y}) is before the earliest date all your funds have data "
            f"({common_start:%m/%Y}). Please choose a later end month."
        )
    if start_ts > end_ts:
        raise bad_request(f"Invalid date range: the start ({start_ts:%m/%Y}) cannot be after the end ({end_ts:%m/%Y}).")

    sliced_returns = returns_df.loc[start_ts:end_ts]
    gaps = [t for t in active_tickers if sliced_returns[t].isna().any()]
    if gaps:
        parts = []
        for t in gaps:
            first, last = returns_df[t].first_valid_index(), returns_df[t].last_valid_index()
            parts.append(f"{t} has data only from {first:%m/%Y} to {last:%m/%Y}")
        raise bad_request("Not enough price history for the selected date range: " + "; ".join(parts) + ".")

    notices = []
    if common_start > FIRST_MONTH and start_ts == common_start:
        late = sorted(((d, t) for t, d in first_valid.items() if d > FIRST_MONTH), reverse=True)
        binding_date, binding_ticker = late[0]
        note = (f"This portfolio starts in {binding_date:%m/%Y} instead of 01/2006 because {binding_ticker} "
                "only has monthly return data from that month.")
