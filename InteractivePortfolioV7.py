# --- 1. IMPORTING TOOLS ---
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf

# --- 2. PAGE SETUP & HEADER ---
st.set_page_config(page_title="Custom Portfolio Compounding Simulator", layout="wide")
st.title("📈 Custom Portfolio Compounding Simulator")
st.write("Search from 100+ pre-2006 ETFs, set your allocations, and see how compounding and fund fees impact your growth.")

# --- 3. LOAD APPROVED ETF LIST & PARSE EXPENSE RATIOS FROM CSV ---
# etfs.csv must sit in the same folder as this file and have the columns:
# Ticker, ETF Name, Expense Ratio (e.g. "0.0400%").
CSV_PATH = Path(__file__).parent / "etfs.csv"

@st.cache_data
def load_etf_catalog():
    df_catalog = pd.read_csv(CSV_PATH)
    df_catalog.columns = [str(c).strip() for c in df_catalog.columns]

    needed = {"Ticker", "ETF Name", "Expense Ratio"}
    missing_cols = needed - set(df_catalog.columns)
    if missing_cols:
        raise ValueError(f"etfs.csv is missing column(s): {', '.join(sorted(missing_cols))}")

    # Drop blank/footer rows, then keep only rows that look like real tickers.
    df_catalog = df_catalog.dropna(subset=["Ticker", "ETF Name"]).copy()
    df_catalog["Ticker"] = df_catalog["Ticker"].astype(str).str.strip().str.upper()
    df_catalog = df_catalog[
        df_catalog["Ticker"].str.fullmatch(r"[A-Z0-9.\-]+") & (df_catalog["Ticker"] != "TICKER")
    ].drop_duplicates(subset="Ticker")

    # "0.0400%" -> 0.0004. Unreadable values become NaN (handled below) instead of crashing.
    fee_pct = pd.to_numeric(
        df_catalog["Expense Ratio"].astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )
    df_catalog["Fee_Decimal"] = fee_pct / 100.0
    df_catalog["Display_Label"] = df_catalog["Ticker"] + " — " + df_catalog["ETF Name"].astype(str).str.strip()
    return df_catalog.set_index("Ticker")[["ETF Name", "Fee_Decimal", "Display_Label"]].to_dict("index")

try:
    FUND_CATALOG = load_etf_catalog()
except FileNotFoundError:
    st.error("❌ Couldn't find **etfs.csv**. Put it in the same folder as this app file (and commit it to GitHub if you deploy).")
    st.stop()
except Exception as e:
    st.error(f"❌ Couldn't read **etfs.csv**: {e}")
    st.stop()

SUPPORTED_TICKERS = sorted(FUND_CATALOG)
DISPLAY_OPTIONS = [FUND_CATALOG[t]["Display_Label"] for t in SUPPORTED_TICKERS]
LABEL_TO_TICKER = {FUND_CATALOG[t]["Display_Label"]: t for t in SUPPORTED_TICKERS}
# Funds whose expense ratio couldn't be read are treated as 0% (and flagged in the sidebar).
FUND_FEE = {t: (0.0 if pd.isna(FUND_CATALOG[t]["Fee_Decimal"]) else FUND_CATALOG[t]["Fee_Decimal"]) for t in SUPPORTED_TICKERS}
MISSING_FEE = {t for t in SUPPORTED_TICKERS if pd.isna(FUND_CATALOG[t]["Fee_Decimal"])}

# --- 4. DATE OPTIONS (01/2006 THROUGH THE LAST *COMPLETE* MONTH) ---
# The current month is excluded because its price is still moving (a partial month).
FIRST_MONTH = pd.Timestamp(2006, 1, 1)
THIS_MONTH = pd.Timestamp.today().normalize().replace(day=1)
LAST_FULL_MONTH = THIS_MONTH - pd.DateOffset(months=1)
MONTHS = pd.date_range(FIRST_MONTH, LAST_FULL_MONTH, freq="MS")
ALL_MONTH_LABELS = [d.strftime("%m/%Y") for d in MONTHS]
LABEL_TO_TS = dict(zip(ALL_MONTH_LABELS, MONTHS))
PRICE_MONTHS = pd.date_range("2005-12-01", LAST_FULL_MONTH, freq="MS")  # Dec 2005 gives Jan 2006 a return

# --- 5. FETCH & CACHE MONTHLY PRICES ---
@st.cache_data(ttl=86400, show_spinner=False)
def get_monthly_closes(tickers_tuple):
    # auto_adjust=True gives dividend-adjusted prices, i.e. total returns.
    raw = yf.download(list(tickers_tuple), start="2005-12-01", interval="1mo", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers_tuple[0]})

    # Normalize every bar to the first day of its month and drop duplicate bars for a month.
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).to_period("M").to_timestamp()
    closes = closes[~closes.index.duplicated(keep="last")].sort_index()
    return closes  # raised errors are NOT cached, so a failed download is retried next run

# --- 6. TRACK NUMBER OF VISIBLE FUND ROWS ---
MAX_FUNDS = 10

if "num_funds" not in st.session_state:
    st.session_state.num_funds = 1

def add_fund():
    if st.session_state.num_funds < MAX_FUNDS:
        st.session_state.num_funds += 1

def remove_fund():
    if st.session_state.num_funds > 1:
        st.session_state.num_funds -= 1

# --- 7. SIDEBAR INPUTS (AMOUNTS, DATES, SEARCHABLE DROPDOWNS & ALLOCATIONS) ---
st.sidebar.header("1. Investment Amounts & Dates")
initial_investment = st.sidebar.number_input("Initial Starting Investment ($)", min_value=0, max_value=10000000, value=0, step=100)
total_monthly = st.sidebar.number_input("Monthly Contribution ($)", min_value=0, max_value=100000, value=360, step=10)

# Dates sit up here (with fixed options) so your choice isn't lost when the fund list changes.
date_col1, date_col2 = st.sidebar.columns(2)
start_label = date_col1.selectbox("From (MM/YYYY)", options=ALL_MONTH_LABELS, index=0)
end_label = date_col2.selectbox("To (MM/YYYY)", options=ALL_MONTH_LABELS, index=len(ALL_MONTH_LABELS) - 1)

st.sidebar.header("2. Choose Funds & Allocation (%)")
st.sidebar.caption("Click any box below and type a ticker or keyword (e.g. *Vanguard*, *Tech*, *Gold*) to search.")

user_selections = []  # (ticker or None, pct)
default_vug_index = SUPPORTED_TICKERS.index("VUG") if "VUG" in SUPPORTED_TICKERS else 0

for i in range(1, st.session_state.num_funds + 1):
    st.sidebar.markdown(f"**Fund {i}**")
    selected_label = st.sidebar.selectbox(
        f"Select ETF {i}",
        options=DISPLAY_OPTIONS,
        # Row 1 starts on VUG; new rows start empty so they can't silently duplicate a fund.
        index=default_vug_index if i == 1 else None,
        placeholder="Search a ticker or name...",
        label_visibility="collapsed",
        key=f"select_{i}",
    )
    ticker_chosen = LABEL_TO_TICKER.get(selected_label)  # None if nothing chosen yet

    pct_input = st.sidebar.number_input(
        f"Allocation % (Fund {i})",
        min_value=0.0,
        max_value=100.0,
        value=100.0 if i == 1 else 0.0,
        step=5.0,
        key=f"pct_{i}",
    )
    user_selections.append((ticker_chosen, pct_input))

btn_col1, btn_col2 = st.sidebar.columns(2)
with btn_col1:
    if st.session_state.num_funds < MAX_FUNDS:
        st.button("➕ Add Fund", on_click=add_fund, width="stretch")
with btn_col2:
    if st.session_state.num_funds > 1:
        st.button("➖ Remove", on_click=remove_fund, width="stretch")

# --- 8. VALIDATION CHECKS (AMOUNTS, DATES, FUNDS & 100% TOTAL) ---
has_error = False

if initial_investment == 0 and total_monthly == 0:
    st.error("⚠️ Please enter at least **$1** in either **Initial Starting Investment** or **Monthly Contribution**.")
    has_error = True

start_ts = LABEL_TO_TS[start_label]
end_ts = LABEL_TO_TS[end_label]
if start_ts > end_ts:
    st.error(f"❌ Invalid date range: **From ({start_label})** cannot be after **To ({end_label})**.")
    has_error = True

seen_tickers = []
for idx, (ticker, pct) in enumerate(user_selections, start=1):
    if ticker is None:
        st.warning(f"👉 Please choose an ETF for **Fund {idx}**, or click **➖ Remove**.")
        has_error = True
    elif ticker in seen_tickers:
        st.error(f"⚠️ Duplicate fund selected: **{ticker}** is listed more than once. Please pick a different ETF.")
        has_error = True
    else:
        seen_tickers.append(ticker)

total_pct = round(sum(pct for _, pct in user_selections), 2)

# Compare with a small tolerance instead of exact float equality.
if abs(total_pct - 100.0) > 0.005:
    st.sidebar.markdown(f"**Current Total:** :red[{total_pct}%]")
    diff = round(abs(100.0 - total_pct), 2)
    if total_pct < 100.0:
        st.error(f"⚠️ Your allocation adds up to **{total_pct}%**. Please add **{diff}%** more to reach **100%**.")
    else:
        st.error(f"⚠️ Your allocation adds up to **{total_pct}%**. Please reduce your allocation by **{diff}%** to equal **100%**.")
    has_error = True
else:
    st.sidebar.markdown("**Current Total:** :green[100%] ✅")

if has_error:
    st.stop()

# Funds actually used (0% rows are ignored), in the order the rows appear.
active_portfolio = {ticker: pct for ticker, pct in user_selections if pct > 0}
active_tickers = list(active_portfolio)

# --- 9. DOWNLOAD RETURNS & CHECK THE DATE RANGE IS COVERED ---
try:
    with st.spinner("Loading historical data since 01/2006..."):
        closes = get_monthly_closes(tuple(sorted(active_tickers)))
except Exception as e:
    st.error(
        "❌ Couldn't download price history from Yahoo Finance. This can be a temporary outage, "
        "rate limiting (common on shared cloud servers), or a missing yfinance install. Try again in a minute."
    )
    st.caption(f"Details: {e}")
    st.stop()

no_data = [t for t in active_tickers if t not in closes.columns or closes[t].dropna().empty]
if no_data:
    st.error(f"❌ Yahoo Finance returned no price history for: **{', '.join(no_data)}**. Please pick a different fund.")
    st.stop()

# Line the prices up on every month from Dec 2005 to the last complete month. This also drops the
# partial current month, and any missing month stays empty (NaN) instead of being silently skipped.
closes = closes[active_tickers].reindex(PRICE_MONTHS)
returns_df = closes.pct_change(fill_method=None).loc[FIRST_MONTH:]

sliced_returns = returns_df.loc[start_ts:end_ts]
gaps = [t for t in active_tickers if sliced_returns[t].isna().any()]
if gaps:
    lines = []
    for t in gaps:
        first, last = returns_df[t].first_valid_index(), returns_df[t].last_valid_index()
        if first is None:
            lines.append(f"**{t}** has no usable monthly returns")
        else:
            lines.append(f"**{t}** has data only from {first:%m/%Y} to {last:%m/%Y}")
    st.error("❌ Not enough price history for the selected date range: " + "; ".join(lines) + ". Please adjust the dates or the fund.")
    st.stop()

# --- 10. BLENDED EXPENSE RATIO & SIDEBAR SUMMARY ---
weighted_fee_decimal = sum((pct / 100.0) * FUND_FEE[t] for t, pct in active_portfolio.items())

st.sidebar.info(f"**Blended Portfolio Expense Ratio:** `{weighted_fee_decimal * 100:.4f}%` / yr")
missing_fee_selected = [t for t in active_tickers if t in MISSING_FEE]
if missing_fee_selected:
    st.sidebar.warning(f"No readable expense ratio in etfs.csv for {', '.join(missing_fee_selected)}; treated as 0%.")

st.sidebar.divider()
st.sidebar.markdown("**Dollar Allocation Split:**")
for ticker, pct in active_portfolio.items():
    init_amt = initial_investment * (pct / 100.0)
    mo_amt = total_monthly * (pct / 100.0)
    if initial_investment > 0:
        st.sidebar.write(f"• **{ticker}**: **${init_amt:,.2f}** start + **${mo_amt:,.2f}** / mo")
    else:
        st.sidebar.write(f"• **{ticker}**: **${mo_amt:,.2f}** / mo")

# --- 11. MONTH-BY-MONTH COMPOUNDING ENGINE (WITH EXPLICIT FEE TRACKING) ---
# Net = real after-fee history. Gross = hypothetical zero-fee version (net return + monthly fee),
# so the gap between them is an estimate of what the expense ratios cost.
net_balances = {t: initial_investment * (pct / 100.0) for t, pct in active_portfolio.items()}
gross_balances = dict(net_balances)
monthly_inputs = {t: total_monthly * (pct / 100.0) for t, pct in active_portfolio.items()}

monthly_records = []
total_invested = initial_investment

for date_ts, row in sliced_returns.iterrows():
    total_invested += total_monthly
    for ticker in active_tickers:
        monthly_fee_rate = FUND_FEE[ticker] / 12.0
        net_r = float(row[ticker])
        gross_r = net_r + monthly_fee_rate  # approximation: add the monthly fee back

        # Deposit at the start of the month, then apply that month's return.
        net_balances[ticker] = (net_balances[ticker] + monthly_inputs[ticker]) * (1 + net_r)
        gross_balances[ticker] = (gross_balances[ticker] + monthly_inputs[ticker]) * (1 + gross_r)

    net_total = sum(net_balances.values())
    gross_total = sum(gross_balances.values())

    row_snapshot = {
        "Date": date_ts,
        "Total Invested": total_invested,
        "Total Portfolio (Net)": round(net_total, 2),
        "Pre-Fee Value (Gross)": round(gross_total, 2),
        "Cumulative Fee Impact": round(max(0.0, gross_total - net_total), 2),
    }
    for ticker in active_tickers:
        row_snapshot[ticker] = round(net_balances[ticker], 2)
    monthly_records.append(row_snapshot)

res = pd.DataFrame(monthly_records)
final_val = res["Total Portfolio (Net)"].iloc[-1]
total_gain = final_val - total_invested
total_fee_impact = res["Cumulative Fee Impact"].iloc[-1]

# --- 12. TOP SUMMARY CARDS (KPIs INCLUDING FEE IMPACT) ---
# total_invested is always > 0 here (the check above requires a lump sum or monthly deposits).
months_label = f"{len(res)} month" + ("" if len(res) == 1 else "s")
col1, col2, col3, col4, col5 = st.columns([1.3, 1, 1, 1, 1])
col1.metric(f"Investment Period ({months_label})", f"{start_label} - {end_label}")
col2.metric("Total Cash Invested", f"${total_invested:,.0f}")
col3.metric("Ending Value (After Fees)", f"${final_val:,.0f}")
col4.metric("Total Compound Gains", f"${total_gain:,.0f}", f"{(total_gain / total_invested) * 100:.1f}%")
col5.metric(
    "Est. Fee Drag Over Time",
    f"-${total_fee_impact:,.0f}",
    f"{weighted_fee_decimal * 100:.3f}% avg fee",
    delta_color="off",
)

# --- 13. INTERACTIVE MONTHLY CHARTS ---
# Same color per fund in both charts (the max of 10 funds matches the 10-color palette).
_palette = px.colors.qualitative.Plotly
COLORS = {t: _palette[i % len(_palette)] for i, t in enumerate(active_tickers)}

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Month-by-Month Compound Growth (Net of Fees)")
    fig_area = px.area(
        res, x="Date", y=active_tickers,
        color_discrete_map=COLORS,
        labels={"value": "Balance ($)", "variable": "Fund", "Date": "Month"},
    )
    fig_area.add_scatter(
        x=res["Date"], y=res["Total Invested"], mode="lines",
        name="Cumulative Cash Invested",
        line=dict(color="black", dash="dash"),
    )
    st.plotly_chart(fig_area, width="stretch")

with chart_col2:
    st.subheader("Ending Allocation Mix")
    fig_pie = px.pie(
        values=[net_balances[t] for t in active_tickers],
        names=active_tickers,
        hole=0.4,
        color=active_tickers,
        color_discrete_map=COLORS,
    )
    st.plotly_chart(fig_pie, width="stretch")

# --- 14. EXPANDABLE MONTH-BY-MONTH LEDGER ---
with st.expander(f"View Full Month-by-Month Compounding Ledger ({months_label})"):
    format_dict = {
        "Date": "{:%m/%Y}",
        "Total Invested": "${:,.0f}",
        "Total Portfolio (Net)": "${:,.2f}",
        "Pre-Fee Value (Gross)": "${:,.2f}",
        "Cumulative Fee Impact": "${:,.2f}",
    }
    for ticker in active_tickers:
        format_dict[ticker] = "${:,.2f}"
    st.dataframe(res.style.format(format_dict), width="stretch")

st.caption(
    "Returns come from Yahoo Finance monthly prices adjusted for dividends and splits, so they are "
    "total returns that already include each fund's real fees. Deposits are made at the start of each "
    "month, and the current (incomplete) month is excluded. The fee-drag figure is only an estimate: "
    "it adds back each fund's CURRENT expense ratio from etfs.csv for every month, but many funds "
    "charged more in the past. For educational purposes only. Historical returns do not guarantee future results."
)
