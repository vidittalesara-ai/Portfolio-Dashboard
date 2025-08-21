import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from dateutil import parser
import numpy as np

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CAS PDF", type=["pdf"])

if uploaded_file:
    # For now, mock parsed data (since parsing CAS PDF directly is complex)
    # Replace with actual parsing logic when integrated
    data = {
        "Fund": ["Axis Midcap", "Axis Smallcap", "Canara Robeco FlexiCap", "Motilal Oswal", "PPFAS"],
        "Cost Value": [134705, 86811, 48132, 274924, 45010],
        "Market Value": [215346, 135361, 71722, 322779, 72729],
        "Category": ["Midcap", "Smallcap", "FlexiCap", "Largecap", "Multicap"],
        "Sector": ["Midcap", "Smallcap", "Diversified", "Largecap", "Diversified"]
    }
    df = pd.DataFrame(data)

    # --- XIRR Calculation (mocked for demo) ---
    def xirr(cashflows, dates):
        def npv(rate):
            return sum([cf / ((1 + rate) ** ((d - dates[0]).days / 365)) for cf, d in zip(cashflows, dates)])

        try:
            return np.irr([cf for cf in cashflows])
        except:
            return None

    portfolio_xirr = 0.145  # Mocked 14.5%

    st.header("Portfolio Dashboard")
    st.subheader("XIRR Analysis")
    st.metric("Portfolio XIRR", f"{portfolio_xirr*100:.2f}%")

    # --- Sector Allocation ---
    st.subheader("Sector Allocation")
    sector_alloc = df.groupby("Sector")["Market Value"].sum()
    fig1, ax1 = plt.subplots()
    ax1.pie(sector_alloc, labels=sector_alloc.index, autopct="%1.1f%%")
    ax1.axis('equal')
    st.pyplot(fig1)

    # --- Market Cap Allocation ---
    st.subheader("Market Cap Allocation")
    mcap_alloc = df.groupby("Category")["Market Value"].sum()
    fig2, ax2 = plt.subplots()
    ax2.pie(mcap_alloc, labels=mcap_alloc.index, autopct="%1.1f%%")
    ax2.axis('equal')
    st.pyplot(fig2)

    # --- Benchmark Comparison ---
    st.subheader("Portfolio vs Benchmark")
    benchmark = st.selectbox("Select Benchmark", ["Nifty 50", "Nifty Midcap", "Nifty Smallcap", "Nifty 500"])

    benchmark_returns = {"Nifty 50": 0.11, "Nifty Midcap": 0.13, "Nifty Smallcap": 0.15, "Nifty 500": 0.12}
    bench_xirr = benchmark_returns[benchmark]

    st.metric("Benchmark XIRR", f"{bench_xirr*100:.2f}%")

    comparison = pd.DataFrame({
        "Portfolio": [portfolio_xirr],
        benchmark: [bench_xirr]
    })
    st.bar_chart(comparison.T)

else:
    st.info("Please upload your CAS PDF to generate the dashboard.")
