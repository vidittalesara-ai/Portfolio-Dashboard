# portfolio_dashboard.py
import re
import io
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# pdf parsing
import pdfplumber

# finance data
import yfinance as yf

st.set_page_config(layout="wide", page_title="3Sigma Partners — Portfolio Dashboard")

# ---------- Utility functions ----------
DATE_PATTERNS = [
    r'\d{2}-[A-Za-z]{3}-\d{4}',  # 05-Sep-2022
    r'\d{2}-[A-Za-z]{3}-\d{2}',  # 05-Sep-22
    r'\d{2}/\d{2}/\d{4}',        # 05/09/2022
    r'\d{2}/\d{2}/\d{2}',        # 05/09/22
]

def parse_date(s):
    # try multiple formats
    for fmt in ("%d-%b-%Y","%d-%b-%y","%d/%m/%Y","%d/%m/%y","%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except:
            pass
    # fallback: try month name spelled
    try:
        return datetime.strptime(s, "%d-%B-%Y").date()
    except:
        return None

def xnpv(rate, cashflows):
    # cashflows: list of (date (datetime.date), amount)
    if rate <= -1.0:
        return None
    d0 = cashflows[0][0]
    return sum([cf / ((1 + rate) ** (( (d - d0).days) / 365.0)) for (d, cf) in cashflows])

def xirr(cashflows, guess=0.12):
    # cashflows: list of (date, amount); must include at least one negative (investment) and one positive (value)
    # use Newton-Raphson
    if len(cashflows) < 2:
        return np.nan
    # ensure sorted by date
    cashflows = sorted(cashflows, key=lambda x: x[0])
    # if all flows are same sign, return nan
    signs = set(np.sign([cf for (_, cf) in cashflows]))
    if len(signs) == 1:
        return np.nan

    def f(r):
        return xnpv(r, cashflows)

    r = guess
    for i in range(100):
        # derivative approximation
        f0 = f(r)
        dr = 1e-6
        f1 = f(r + dr)
        if f0 is None or f1 is None:
            return np.nan
        deriv = (f1 - f0) / dr
        if deriv == 0:
            break
        new_r = r - f0 / deriv
        if abs(new_r - r) < 1e-6:
            r = new_r
            break
        r = new_r
    return r

# ---------- CAS parser ----------
def extract_text_from_pdf(file_bytes):
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

# Identify transaction lines using regex heuristics:
TXN_LINE_RE = re.compile(
    r'(?P<date>\d{2}[-/][A-Za-z]{3}[-/]\d{2,4}|\d{2}/\d{2}/\d{2,4})\s+'
    r'(?P<amount>-?[\d,]+\.\d{2})\s+'
    r'(?P<nav>[\d,]+\.\d+)\s*'
    r'(?P<units>[-\d,]+\.\d+)\s*(?P<txn>.+)$'
)

# Some CAS variants don't include nav/units in same columns; fallback regex:
ALT_TXN_RE = re.compile(
    r'(?P<date>\d{2}[-/][A-Za-z]{3}[-/]\d{2,4})\s+'
    r'(?P<amount>-?[\d,\,]+\.\d{2})\s*(?P<rest>.+)$'
)

def parse_cas_text(text):
    """
    Returns:
      - transactions: DataFrame with columns [date, amount, nav, units, txn_desc, folio, scheme]
      - holdings: DataFrame summarising closing units, nav on date, market value, total_cost
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    transactions = []
    current_scheme = None
    folio = None
    closing_info = []

    # identify "Folio No" and "NAV on" and "Closing Unit Balance" lines to capture holdings meta
    for i, line in enumerate(lines):
        # detect scheme header pattern: lines that have ISIN or ' - Direct - Growth' or 'Fund' frequently
        if len(line) > 30 and ('ISIN' in line or 'Fund' in line or 'Plan' in line) and not line.startswith('Date '):
            current_scheme = line
            # try extract folio nearby
            # find folio in next 5 lines
            folio = None
            for j in range(i, min(i+6, len(lines))):
                if 'Folio No' in lines[j]:
                    folio = lines[j].split(':')[-1].strip()
                    break

        m = TXN_LINE_RE.match(line)
        if m:
            dt = parse_date(m.group('date'))
            amt = float(m.group('amount').replace(',', ''))
            nav = float(m.group('nav').replace(',', ''))
            units = float(m.group('units').replace(',', ''))
            txn = m.group('txn').strip()
            transactions.append({
                'scheme': current_scheme,
                'folio': folio,
                'date': dt,
                'amount': amt,
                'nav': nav,
                'units': units,
                'txn_desc': txn
            })
            continue
        # fallback match
        m2 = ALT_TXN_RE.match(line)
        if m2:
            dt = parse_date(m2.group('date'))
            amt = float(m2.group('amount').replace(',', ''))
            rest = m2.group('rest').strip()
            # attempt to extract nav/units from rest by searching for two numbers
            nums = re.findall(r'([\d,]+\.\d+)', rest)
            nav = float(nums[0].replace(',', '')) if len(nums) > 0 else np.nan
            units = float(nums[1].replace(',', '')) if len(nums) > 1 else np.nan
            txn = rest
            transactions.append({
                'scheme': current_scheme,
                'folio': folio,
                'date': dt,
                'amount': amt,
                'nav': nav,
                'units': units,
                'txn_desc': txn
            })
            continue

        # capture closing unit / market value lines
        if line.startswith('NAV on'):
            # sample: NAV on 06-Aug-2025: INR 31.8576 Market Value on 06-Aug-2025: INR 25,254.38
            nav_match = re.search(r'NAV on\s+(?P<nav_date>[\d-A-Za-z]+):\s*INR\s*(?P<nav>[\d,\.]+)', line)
            mv_match = re.search(r'Market Value on\s+[\d-A-Za-z]+:\s*INR\s*(?P<mv>[\d,\,\.]+)', line)
            if nav_match:
                nav_date = nav_match.group('nav_date')
                nav = float(nav_match.group('nav').replace(',', ''))
                mv = float(mv_match.group('mv').replace(',', '')) if mv_match else np.nan
                closing_info.append({'scheme': current_scheme, 'folio': folio, 'nav_on': nav_date, 'nav': nav, 'market_value': mv})
        if line.startswith('Closing Unit Balance:'):
            # Closing Unit Balance: 792.727 Total Cost Value: 15,752.62
            m = re.search(r'Closing Unit Balance:\s*([-\d,\.]+)\s+Total Cost Value:\s*([-\d,\,\.]+)', line)
            if m:
                units = float(m.group(1).replace(',', ''))
                cost = float(m.group(2).replace(',', ''))
                if closing_info:
                    closing_info[-1].update({'closing_units': units, 'total_cost_value': cost})
    tx_df = pd.DataFrame(transactions)
    hold_df = pd.DataFrame(closing_info)
    return tx_df, hold_df

# ---------- benchmark mapping ----------
# ticker mapping using yfinance (India indexes often available as ETFs). We'll map to commonly used tickers:
BENCHMARKS = {
    "Nifty 500": "^NSEI",  # substitute: we will use Nifty 50 for price series in example; user can edit tickers
    "Nifty 50": "^NSEI",
    "Nifty Next 50 (Junior Nifty)": "^CNX100",  # placeholder; user may need to adjust to a valid ticker or ETF ticker
    "Nifty Midcap": "NIFTY_MIDCAP_100.NS",  # placeholder — sample code will accept any yfinance ticker user prefers
    "Nifty Smallcap": "NIFTY_SMALLCAP_100.NS"
}

def fetch_benchmark_series(ticker, start_date, end_date):
    # wrap yfinance
    try:
        series = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if series.empty:
            return None
        return series['Adj Close'].rename(ticker)
    except Exception as e:
        return None

# ---------- UI ----------
st.markdown("<h1 style='color:#0b3954'>3Sigma Partners — Portfolio Dashboard</h1>", unsafe_allow_html=True)
st.sidebar.header("Upload & Settings")

uploaded = st.sidebar.file_uploader("Upload CAMS CAS PDF", type=['pdf'])
benchmark_choice = st.sidebar.selectbox("Default benchmark (for comparisons)", ["Nifty 500", "Nifty 50", "Nifty Midcap", "Nifty Smallcap", "Nifty Next 50 (Junior Nifty)"])
compute_btn = st.sidebar.button("Parse & Build Dashboard")

st.sidebar.markdown("---")
st.sidebar.markdown("Palette: Blues / Grey / Golden")

if uploaded is None:
    st.info("Upload your CAMS Consolidated Account Statement (PDF) to start. Example CAS patterns detected in your file include `SIP Purchase` / `Redemption` transaction lines. :contentReference[oaicite:2]{index=2}")
    st.stop()

if compute_btn:
    with st.spinner("Parsing CAS PDF..."):
        raw_text = extract_text_from_pdf(uploaded.read())
        tx_df, hold_df = parse_cas_text(raw_text)

    if tx_df.empty:
        st.error("No transactions parsed — CAS format may differ. Try a different CAS or let me know and I'll adapt parser.")
        st.write("First 2000 characters of extracted text to help debugging:")
        st.code(raw_text[:2000])
        st.stop()

    # tidy
    tx_df['date'] = pd.to_datetime(tx_df['date'])
    tx_df['amount'] = pd.to_numeric(tx_df['amount'])
    tx_df['units'] = pd.to_numeric(tx_df['units'], errors='coerce')
    tx_df['nav'] = pd.to_numeric(tx_df['nav'], errors='coerce')
    tx_df['scheme'] = tx_df['scheme'].fillna('Unknown Scheme')

    # derive holdings from last known NAV * units
    grouped = tx_df.groupby('scheme').agg({'units': 'sum'})
    # last NAV per scheme
    last_nav = tx_df.sort_values('date').groupby('scheme')['nav'].last()
    holdings = grouped.join(last_nav).reset_index().rename(columns={'units': 'total_units', 'nav': 'last_nav'})
    holdings['market_value'] = holdings['total_units'] * holdings['last_nav']

    # create contributions timeseries (monthly)
    tx_invest = tx_df[tx_df['amount'] > 0].copy()  # contributions positive amounts (SIP/Purchase)
    tx_withdraw = tx_df[tx_df['amount'] < 0].copy()  # redemptions negative

    monthly_invest = tx_invest.set_index('date').resample('M').amount.sum().fillna(0).cumsum()
    invested_monthly = tx_invest.set_index('date').resample('M').amount.sum().fillna(0)

    # compute portfolio value timeseries — approximate using spot market values (we only have closing market values on statement date)
    # We'll create a simple monthly portfolio value by summing scheme market_value where possible, and forward fill
    # Build a synthetic monthly date index from first tx to today
    start = tx_df['date'].min().date()
    end = datetime.today().date()
    months = pd.date_range(start=start, end=end, freq='M')

    # For each month, compute market value by summing units * last_nav (approx). More sophisticated approach requires historical NAV series per scheme.
    portfolio_vals = []
    for m in months:
        # include all transactions up to month-end, compute units held then and multiply by last known NAV (approx)
        upto = tx_df[tx_df['date'] <= m]
        holdings_up = upto.groupby('scheme').units.sum()
        # last known nav up to that month for each scheme (approx)
        navs = upto.groupby('scheme').nav.last().fillna(0)
        mv = (holdings_up * navs).sum()
        portfolio_vals.append({'month': m, 'portfolio_value': mv, 'invested_cumulative': upto[upto['amount']>0].amount.sum()})
    pv_df = pd.DataFrame(portfolio_vals)

    # Compute XIRR per scheme:
    scheme_xirr = {}
    for scheme, group in tx_df.groupby('scheme'):
        # cashflows: investments = negative (outflow), redemptions and current market value = positive
        cfs = []
        for _, r in group.sort_values('date').iterrows():
            amt = r['amount']
            date = r['date'].date()
            # in CAS amount is positive for purchase, negative for redemption
            # For XIRR convention: investments are negative cashflows
            if amt > 0:
                cfs.append((date, -amt))
            else:
                cfs.append((date, -amt))  # redemption positive
        # append a final positive cashflow = current market value at today's date
        # find last nav for scheme
        current_value = holdings.loc[holdings['scheme'] == scheme, 'market_value'].values
        if len(current_value) > 0:
            current_value = float(current_value[0])
            cfs.append((datetime.today().date(), current_value))
        xr = xirr(cfs)
        scheme_xirr[scheme] = xr

    holdings['xirr'] = holdings['scheme'].map(scheme_xirr)

    # portfolio XIRR: combine all transactions into cashflow list
    pf_cfs = []
    for _, r in tx_df.sort_values('date').iterrows():
        amt = r['amount']
        dt = r['date'].date()
        pf_cfs.append((dt, -amt if amt > 0 else -amt))
    # final current portfolio value as positive
    pf_current_value = holdings['market_value'].sum()
    pf_cfs.append((datetime.today().date(), float(pf_current_value)))
    portfolio_xirr = xirr(pf_cfs)

    # Benchmarks: fetch series and compute XIRR for benchmark with same cashflow dates (i.e., if you invested same amounts on same dates into benchmark)
    # Create helper to compute "what if invested into benchmark" XIRR
    def compute_benchmark_xirr_for_cashflows(ticker, cashflows):
        # cashflows: list of (date, contributed_amount as positive)
        # approach: for each contribution date, buy units = amount / price_on_date; then final value = units * last_price
        # we fetch price for each unique date and last price
        dates = sorted({cf[0] for cf in cashflows})
        # fetch series from earliest date - 5 days to today
        earliest = min(dates)
        start_fetch = earliest - pd.Timedelta(days=7)
        end_fetch = datetime.today().date()
        s = fetch_benchmark_series(ticker, start_fetch, end_fetch)
        if s is None or s.empty:
            return None, None
        s = s.dropna()
        # price on or before a date: use last available price on or before that date
        unit_map = {}
        for d in dates:
            # yfinance series index is Timestamp; find last index <= d
            sidx = s[s.index.date <= d]
            if sidx.empty:
                price = None
            else:
                price = float(sidx.iloc[-1])
            unit_map[d] = price

        # build cashflows for XIRR: investments negative, final value positive
        cfs = []
        total_units = 0.0
        for (d, amt) in cashflows:
            price = unit_map.get(d)
            if (price is None) or price == 0:
                # cannot compute
                return None, None
            units_bought = amt / price
            total_units += units_bought
            cfs.append((d, -amt))
        last_price = float(s.iloc[-1])
        final_value = total_units * last_price
        cfs.append((datetime.today().date(), final_value))
        xr = xirr(cfs)
        return xr, s

    # Build UI tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Holdings", "Transactions", "Benchmark Comparison", "Allocations"])

    # ---------- Overview ----------
    with tab1:
        st.subheader("Portfolio timeline vs Benchmark (monthly)")
        # Build monthly invested series and portfolio series from pv_df
        pv_df['month_str'] = pv_df['month'].dt.to_period('M').astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pv_df['month'], y=pv_df['invested_cumulative'],
                                 mode='lines+markers', name='Cumulative Invested', line=dict(width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=pv_df['month'], y=pv_df['portfolio_value'],
                                 mode='lines+markers', name='Portfolio Value', line=dict(width=3)))
        # fetch benchmark series for same dates and compute "if invested monthly into benchmark"
        benchmark_label = benchmark_choice
        bench_ticker = BENCHMARKS.get(benchmark_label, "^NSEI")
        # Build contributions as monthly sums for the months used (date and amount)
        monthly_contribs = tx_invest.set_index('date').resample('M').amount.sum().reset_index()
        cashflows_for_bench = [(row['date'].date(), float(row['amount'])) for _, row in monthly_contribs.iterrows() if row['amount'] > 0]

        bench_xirr, bench_series = compute_benchmark_xirr_for_cashflows(bench_ticker, cashflows_for_bench)
        bench_values = None
        if bench_series is not None:
            # compute hypothetical benchmark value after each month by accumulating units
            units = 0.0
            by_month_vals = []
            for _, row in monthly_contribs.iterrows():
                dt = row['date'].date()
                amt = float(row['amount'])
                # price on or before dt
                price_on = bench_series[bench_series.index.date <= dt]
                price_on = float(price_on.iloc[-1]) if not price_on.empty else np.nan
                if not np.isnan(price_on):
                    units += amt / price_on
                val = units * float(bench_series.iloc[-1])  # value at last price
                by_month_vals.append((row['date'], val))
            if by_month_vals:
                bm_df = pd.DataFrame(by_month_vals, columns=['date', 'benchmark_value'])
                fig.add_trace(go.Scatter(x=bm_df['date'], y=bm_df['benchmark_value'],
                                         mode='lines+markers', name=f'{benchmark_label} (hypothetical)'))
        fig.update_layout(
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title='Month',
            yaxis_title='INR',
            colorway=["#0b3954", "#8aa2b8", "#d4af37"]  # blue, grey-blue, golden
        )
        st.plotly_chart(fig, use_container_width=True)

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Invested (₹)", f"{tx_invest['amount'].sum():,.0f}")
        col2.metric("Current Portfolio Value (₹)", f"{pf_current_value:,.0f}")
        col3.metric("Portfolio XIRR (ann.)", f"{portfolio_xirr*100:.2f}%" if pd.notna(portfolio_xirr) else "N/A")
        if bench_xirr:
            col4.metric(f"{benchmark_label} XIRR (for same cashflows)", f"{bench_xirr*100:.2f}%")
        else:
            col4.metric(f"{benchmark_label} XIRR (for same cashflows)", "N/A")

    # ---------- Holdings ----------
    with tab2:
        st.subheader("Holdings & XIRR per sc
