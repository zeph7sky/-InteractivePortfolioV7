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

# Page styling: smaller summary-card text (wrapped instead of cut off with "...") and no +/- buttons on number boxes.
# To change the size, edit the 1.6rem below (default Streamlit size is about 2.25rem).
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div {
        font-size: 1.6rem;
        overflow: visible;
        text-overflow: clip;
        white-space: normal;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        font-size: 0.9rem;
        overflow: visible;
        text-overflow: clip;
        white-space: normal;
    }
    /* Hide the + / - buttons on number boxes. You can still type a value (or use the up/down arrow keys). */
    [data-testid="stNumberInputStepUp"],
    [data-testid="stNumberInputStepDown"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# The From/To boxes are reserved here (so they appear under the amounts) but are filled in later,
# once we know how far back the chosen funds' data goes. That lets "From" default to the latest
# fund start date automatically.
date_box = st.sidebar.container()

def mark_start_manual():
    # Runs only when the USER changes the From box (not when the app sets it automatically).
    st.session_state["start_manual"] = True

_dates = {"done": False, "start": None, "end": None}

def render_date_inputs(common_start):
    """Draw the From/To boxes. common_start = earliest month every chosen fund has data (or None)."""
    if common_start is not None:
        common_label = common_start.strftime("%m/%Y")
        chosen = st.session_state.get("start_sel")
        # Keep the user's own choice only if it is later than the earliest possible start.
        keep_manual = (
            st.session_state.get("start_manual", False)
            and chosen in LABEL_TO_TS
            and LABEL_TO_TS[chosen] > common_start
        )
        if not keep_manual:
            st.session_state["start_sel"] = common_label
            st.session_state["start_manual"] = False
    else:
        st.session_state.setdefault("start_sel", ALL_MONTH_LABELS[0])
    st.session_state.setdefault("end_sel", ALL_MONTH_LABELS[-1])

    with date_box:
        c1, c2 = st.columns(2)
        _dates["start"] = c1.selectbox("From (MM/YYYY)", options=ALL_MONTH_LABELS, key="start_sel", on_change=mark_start_manual)
        _dates["end"] = c2.selectbox("To (MM/YYYY)", options=ALL_MONTH_LABELS, key="end_sel")
    _dates["done"] = True

def halt():
    """Stop the page, but make sure the date boxes are drawn first so their values aren't lost."""
    if not _dates["done"]:
        render_date_inputs(None)
    st.stop()

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

# Rebalancing switch. When ON, every December the funds are reset to the target % typed above.
rebalance_on = st.sidebar.checkbox(
    "Rebalance every December to my target %",
    value=True,
    help=(
        "Funds grow at different speeds, so their weights drift away from your targets. When this is ON, "
        "at the end of each December the app resets every fund to its target % of the total portfolio "
        "(selling the funds that grew more and buying the ones that lagged). Assumes no trading costs or taxes."
    ),
)

# --- 8. VALIDATION CHECKS (AMOUNTS, DATES, FUNDS & 100% TOTAL) ---
has_error = False

if initial_investment == 0 and total_monthly == 0:
    st.error("⚠️ Please enter at least **$1** in either **Initial Starting Investment** or **Monthly Contribution**.")
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
    halt()

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
    halt()

no_data = [t for t in active_tickers if t not in closes.columns or closes[t].dropna().empty]
if no_data:
    st.error(f"❌ Yahoo Finance returned no price history for: **{', '.join(no_data)}**. Please pick a different fund.")
    halt()

# Line the prices up on every month from Dec 2005 to the last complete month. This also drops the
# partial current month, and any missing month stays empty (NaN) instead of being silently skipped.
closes = closes[active_tickers].reindex(PRICE_MONTHS)
returns_df = closes.pct_change(fill_method=None).loc[FIRST_MONTH:]

# Each fund's first month with a usable return. The portfolio can only start once ALL funds have data,
# i.e. at the latest of these dates.
first_valid = {t: returns_df[t].first_valid_index() for t in active_tickers}
unusable = [t for t, d in first_valid.items() if d is None]
if unusable:
    st.error(f"❌ **{', '.join(unusable)}** has no usable monthly returns. Please pick a different fund.")
    halt()

common_start = max(first_valid.values())

# Draw From/To now. "From" defaults to common_start unless the user picked a later date themselves.
render_date_inputs(common_start)
start_label, end_label = _dates["start"], _dates["end"]
start_ts = LABEL_TO_TS[start_label]
end_ts = LABEL_TO_TS[end_label]

if end_ts < common_start:
    st.error(
        f"❌ **To ({end_label})** is before the earliest date all your funds have data "
        f"(**{common_start:%m/%Y}**). Please choose a later end date."
    )
    st.stop()
if start_ts > end_ts:
    st.error(f"❌ Invalid date range: **From ({start_label})** cannot be after **To ({end_label})**.")
    st.stop()

# Tell the user why the portfolio doesn't start in 01/2006.
if common_start > FIRST_MONTH and start_ts == common_start:
    late_funds = sorted(((d, t) for t, d in first_valid.items() if d > FIRST_MONTH), reverse=True)
    binding_date, binding_ticker = late_funds[0]
    note = (
        f"ℹ️ This portfolio starts in **{binding_date:%m/%Y}** instead of 01/2006 because "
        f"**{binding_ticker}** only has monthly return data from that month."
    )
    others = [f"{t} ({d:%m/%Y})" for d, t in late_funds[1:]]
    if others:
        note += f" Other funds that also start after 01/2006: {', '.join(others)}."
    note += " Every fund is measured over the same period so the comparison stays fair."
    st.info(note)

sliced_returns = returns_df.loc[start_ts:end_ts]
gaps = [t for t in active_tickers if sliced_returns[t].isna().any()]
if gaps:
    lines = []
    for t in gaps:
        first, last = returns_df[t].first_valid_index(), returns_df[t].last_valid_index()
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

# --- 11. MONTH-BY-MONTH COMPOUNDING ENGINE (WITH FEE TRACKING + OPTIONAL REBALANCING) ---
# We keep TWO parallel sets of balances, one per fund:
#   * NET   = the real after-fee history (Yahoo prices already have each fund's fees taken out).
#   * GROSS = a hypothetical zero-fee version (net return + that month's share of the expense ratio).
# The gap between GROSS and NET is the "Est. Fee Drag" shown on the page.
#
# Every month, for every fund i, the balance is updated with:
#       balance_i  =  (balance_i + deposit_i) * (1 + return_i)
# i.e. the deposit goes in at the START of the month, then the month's return is applied.

# Target weight of each fund as a decimal (e.g. 60% -> 0.60). These never change during the test.
target_weights = {t: pct / 100.0 for t, pct in active_portfolio.items()}

# Starting balances: the initial lump sum split by the target weights.
net_balances = {t: initial_investment * w for t, w in target_weights.items()}
gross_balances = dict(net_balances)  # the zero-fee version starts from the same money

# Each month's deposit for each fund = monthly contribution x target weight.
monthly_inputs = {t: total_monthly * w for t, w in target_weights.items()}

monthly_records = []              # one dictionary per month; becomes the ledger table
monthly_portfolio_returns = []    # contribution-adjusted monthly return, used for volatility
total_invested = initial_investment  # running total of cash the user has put in
rebalance_count = 0               # how many times we reset the funds back to target

for date_ts, row in sliced_returns.iterrows():
    # Portfolio value BEFORE this month's deposit and return (needed for the monthly return below).
    prev_net_total = sum(net_balances.values())
    total_invested += total_monthly

    for ticker in active_tickers:
        net_r = float(row[ticker])                     # the fund's actual return this month
        monthly_fee_rate = FUND_FEE[ticker] / 12.0     # one month's share of the yearly expense ratio
        gross_r = net_r + monthly_fee_rate             # add the fee back to estimate the zero-fee return

        # Deposit first, then apply the month's return.
        net_balances[ticker] = (net_balances[ticker] + monthly_inputs[ticker]) * (1 + net_r)
        gross_balances[ticker] = (gross_balances[ticker] + monthly_inputs[ticker]) * (1 + gross_r)

    net_total = sum(net_balances.values())
    gross_total = sum(gross_balances.values())

    # Portfolio return for the month = ending value / (starting value + this month's deposit) - 1.
    # Dividing by the deposit too keeps new contributions from looking like gains.
    # (Rebalancing below only moves money between funds, so it never changes these totals.)
    start_base = prev_net_total + total_monthly
    port_ret = net_total / start_base - 1 if start_base > 0 else float("nan")
    monthly_portfolio_returns.append(port_ret)

    # ---- ANNUAL REBALANCING (end of December) ----
    # Funds grow at different speeds, so their weights drift away from the targets. At the end of each
    # December we reset every fund to its target share of the CURRENT total:
    #       new_balance_i = total_portfolio_value x target_weight_i
    # Example: target 60/40, and after December's return A = $7,200 and B = $2,800 (total $10,000).
    # Reset -> A = $6,000 and B = $4,000: we sell $1,200 of A and buy $1,200 of B. The total stays $10,000.
    # Net and gross are rebalanced separately, each to its own total, so the fee comparison stays fair.
    # With a single fund there is nothing to rebalance. Assumes no trading costs and no taxes.
    rebalanced_this_month = False
    if rebalance_on and date_ts.month == 12 and len(active_tickers) > 1:
        for ticker in active_tickers:
            net_balances[ticker] = net_total * target_weights[ticker]
            gross_balances[ticker] = gross_total * target_weights[ticker]
        rebalanced_this_month = True
        rebalance_count += 1

    # Save this month's snapshot (fund balances shown AFTER any rebalancing that happened).
    row_snapshot = {
        "Date": date_ts,
        "Total Invested": total_invested,
        "Total Portfolio (Net)": round(net_total, 2),
        "Pre-Fee Value (Gross)": round(gross_total, 2),
        "Cumulative Fee Impact": round(max(0.0, gross_total - net_total), 2),
        "Portfolio Return (Month)": port_ret,
        "Rebalanced?": "Yes" if rebalanced_this_month else "",
    }
    for ticker in active_tickers:
        row_snapshot[ticker] = round(net_balances[ticker], 2)
    monthly_records.append(row_snapshot)

res = pd.DataFrame(monthly_records)
final_val = res["Total Portfolio (Net)"].iloc[-1]
total_gain = final_val - total_invested
total_fee_impact = res["Cumulative Fee Impact"].iloc[-1]

# Annualized volatility = standard deviation of the monthly returns x sqrt(12).
# It needs at least 2 months of returns.
monthly_ret_series = pd.Series(monthly_portfolio_returns).dropna()
annualized_vol = monthly_ret_series.std() * (12 ** 0.5) if len(monthly_ret_series) >= 2 else None

# "Growth of $1": what $1 would have become using the same monthly portfolio returns.
# Because each month's return already adjusts for your deposits, this curve is NOT distorted by
# when or how much you contributed. Both new numbers below are built from it.
n_months = len(monthly_ret_series)
growth_curve = pd.concat([pd.Series([1.0]), (1 + monthly_ret_series).cumprod()], ignore_index=True)
# (position 0 is the starting point; position k is the end of the k-th month)

# Annualized growth rate = the steady yearly rate that would produce the same total growth:
#     (final growth of $1) ^ (12 / number of months) - 1
# Shown only for 12+ months, because stretching a few months into a yearly rate is misleading.
annual_growth = growth_curve.iloc[-1] ** (12 / n_months) - 1 if n_months >= 12 else None

# Biggest drop (maximum drawdown) = the largest fall from a previous high point to a later low:
#     drawdown at each point = growth_curve / highest value so far - 1
# The most negative drawdown is the biggest drop. It is measured on month-end values, so a
# dip inside a single month is not captured.
drawdowns = growth_curve / growth_curve.cummax() - 1
trough_pos = int(drawdowns.idxmin())
max_drawdown = float(drawdowns.iloc[trough_pos])                          # 0 or negative
peak_pos = int(growth_curve.iloc[: trough_pos + 1][::-1].idxmax())        # the high just before the low

def _drawdown_date(pos):
    # position 0 = before the first month; position k = end of the k-th month in the ledger
    return "the start of the period" if pos == 0 else f"{res['Date'].iloc[pos - 1]:%m/%Y}"

# --- 12. TOP SUMMARY CARDS (KPIs INCLUDING FEE IMPACT) ---
# total_invested is always > 0 here (the check above requires a lump sum or monthly deposits).
months_label = f"{len(res)} month" + ("" if len(res) == 1 else "s")
# Row 1: the money numbers.
row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
row1_col1.metric("Total Cash Invested", f"${total_invested:,.0f}")
row1_col2.metric("Ending Value (After Fees)", f"${final_val:,.0f}")
row1_col3.metric("Total Compound Gains", f"${total_gain:,.0f}", f"{(total_gain / total_invested) * 100:.1f}%")
row1_col4.metric(
    "Annualized Growth Rate",
    f"{annual_growth * 100:+.1f}%" if annual_growth is not None else "N/A",
    help=(
        "The steady yearly growth rate (compounded) that would give the same total growth, after fees. "
        "It adjusts for when you made deposits, so it is not skewed by contributing more in good or bad years. "
        f"Based on {n_months} monthly returns. Shown only for periods of 12 months or longer."
    ),
)

# Row 2: period, fees and risk.
row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
row2_col1.metric(f"Investment Period ({months_label})", f"{start_label} - {end_label}")
row2_col2.metric(
    "Est. Fee Drag Over Time",
    f"-${total_fee_impact:,.0f}",
    f"{weighted_fee_decimal * 100:.3f}% avg fee",
    delta_color="off",
)
row2_col3.metric(
    "Annualized Volatility",
    f"{annualized_vol * 100:.1f}%" if annualized_vol is not None else "N/A",
    help=(
        "Standard deviation of the portfolio's monthly returns (after fees, adjusted for your deposits) "
        f"multiplied by the square root of 12. Based on {n_months} monthly returns. "
        "Higher means a bumpier ride. Needs at least 2 months of data."
    ),
)
row2_col4.metric(
    "Biggest Drop (Max Drawdown)",
    "0.0%" if max_drawdown > -0.0005 else f"{max_drawdown * 100:.1f}%",
    help=(
        "The largest fall in the portfolio's value from a previous high to a later low, measured on "
        "month-end values and adjusted for your deposits, so new contributions can't hide a loss. "
        "A dip inside a single month is not captured."
    ),
)

# Say when the biggest drop happened (or that there wasn't one).
if max_drawdown > -0.0005:
    st.caption("Biggest drop: none. The portfolio never fell below a previous high in this period.")
else:
    st.caption(
        f"Biggest drop: {abs(max_drawdown) * 100:.1f}%, from the high in {_drawdown_date(peak_pos)} "
        f"to the low in {_drawdown_date(trough_pos)}."
    )

# One-line reminder of the rebalancing setting used for the numbers above.
if len(active_tickers) == 1:
    st.caption("Rebalancing: not applicable (only one fund selected).")
elif rebalance_on:
    st.caption(f"Rebalancing: ON. Funds were reset to your target % at the end of each December ({rebalance_count} time{'s' if rebalance_count != 1 else ''} in this period).")
else:
    st.caption("Rebalancing: OFF. Fund weights were left to drift with market performance.")

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
        "Portfolio Return (Month)": "{:.2%}",
    }
    for ticker in active_tickers:
        format_dict[ticker] = "${:,.2f}"
    st.dataframe(res.style.format(format_dict), width="stretch")

st.caption(
    "When rebalancing is on, funds are reset to your target % each December with no trading costs or taxes. "
    "Returns come from Yahoo Finance monthly prices adjusted for dividends and splits, so they are "
    "total returns that already include each fund's real fees. Deposits are made at the start of each "
    "month, and the current (incomplete) month is excluded. The fee-drag figure is only an estimate: "
    "it adds back each fund's CURRENT expense ratio from etfs.csv for every month, but many funds "
    "charged more in the past. Annualized volatility is the standard deviation of monthly portfolio returns "
    "times the square root of 12 (past variability, not a forecast). Annualized growth rate and biggest drop "
    "are based on month-end values and adjusted for your deposits. For educational purposes only. Historical returns do not guarantee future results."
)
