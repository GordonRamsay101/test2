import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime

# Set up the app
st.set_page_config(page_title="Quant Finance AI", layout="wide")
st.title("📈 Belfort AI")

# User input
symbol = st.text_input("Enter Stock Symbol (e.g., SPX, GOOGL, DJI):", "AAPL").upper()
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if st.button("Run Analysis"):
    with st.spinner("Fetching data..."):
        df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found.")
        st.stop()

    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Indicators
    df["SMA_7"] = df["Close"].rolling(window=7).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["Daily Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily Return"].rolling(window=30).std()
    df["Resistance"] = df["High"].rolling(window=7).max()
    df["Support"] = df["Low"].rolling(window=30).min()

    # ========== Trading Logic ==========
    def make_decision(data):
        if len(data) < 30:
            return "HOLD", None, None, None, None, []

        latest = data.iloc[-1]
        conditions = []

        try:
            sma_7, sma_30 = float(latest["SMA_7"]), float(latest["SMA_30"])
            vol, close, support = float(latest["Volatility"]), float(latest["Close"]), float(latest["Support"])
        except:
            return "HOLD", None, None, None, None, []

        conditions.append("✅ SMA over 7 days > SMA over 30 days" if sma_7 > sma_30 else "❌ SMA over 7 days ≤ SMA over 30 days")
        conditions.append("✅ Volatility < 2%" if vol < 0.02 else "❌ Volatility ≥ 2%")
        conditions.append("✅ Close > Support" if close > support else "❌ Close ≤ Support")

        if all(c.startswith("✅") for c in conditions):
            sl = support
            rr_2_tp = close + 2 * (close - sl)
            rr_3_tp = close + 3 * (close - sl)
            rr_best = 3 if rr_3_tp - close > rr_2_tp - close else 2
            tp = rr_3_tp if rr_best == 3 else rr_2_tp
            return "BUY", close, tp, sl, rr_best, conditions

        sl = close * 0.98
        tp = close + 2 * (close - sl)
        return "HOLD", close, tp, sl, 2, conditions

    decision, entry, tp, sl, rr, checks = make_decision(df)

    # ========== Display Decision ==========
    st.subheader("📌 Trading Decision")
    for check in checks:
        st.markdown(f"- {check}")

    # Format dollar amounts with commas and no decimals
    entry_fmt = f"${entry:,.0f}" if entry else None
    tp_fmt = f"${tp:,.0f}" if tp else None
    sl_fmt = f"${sl:,.0f}" if sl else None

    if decision == "BUY":
        st.success(f"💰 **BUY**")
        st.markdown(f"**RR**: {rr}:1")
        st.markdown(f"**Entry**: {entry_fmt}")
        st.markdown(f"**TP**: {tp_fmt}")
        st.markdown(f"**SL**: {sl_fmt}")
    else:
        st.warning(f"⚠️ HOLD! Not optimal to buy.")
        st.markdown(f"**RR**: {rr}:1")
        if entry:
            st.markdown(f"**Hypothetical Entry**: {entry_fmt}")
            st.markdown(f"**TP**: {tp_fmt}")
            st.markdown(f"**SL**: {sl_fmt}")

    # ========== Stock Chart ==========
    st.subheader("📉 Stock Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_7"], name="SMA 7", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_30"], name="SMA 30", line=dict(color="red", dash="dash")))
    fig.update_layout(title=f"{symbol} Price", xaxis_title="Date", yaxis_title="Price",
                      width=1100, height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ========== TradingView Embed ==========
    st.subheader("📺 Live Chart")
    # Adjust symbol for TradingView: use "TVC:DJI" if symbol == DJI, else "NASDAQ:<symbol>"
    if symbol == "DJI":
        tradingview_symbol = "TVC:DJI"
    else:
        tradingview_symbol = f"NASDAQ:{symbol}"
    tradingview_html = f"""
    <iframe src="https://www.tradingview.com/embed-widget/symbol-overview/?locale=en#%7B%22symbols%22%3A%5B%5B%22{tradingview_symbol}%22%5D%5D%2C%22width%22%3A%22100%25%22%2C%22height%22%3A300%2C%22locale%22%3A%22en%22%2C%22colorTheme%22%3A%22dark%22%7D"
    width="100%" height="300" frameborder="0"></iframe>
    """
    st.markdown(tradingview_html, unsafe_allow_html=True)

    # ========== Forecasting ==========
    df_model = df.dropna().copy()
    df_model["days_since_start"] = (df_model.index - df_model.index.min()).days
    daily_avg_close = df_model[["days_since_start", "Close"]].rename(columns={"Close": "close"})

    X = daily_avg_close["days_since_start"].values.reshape(-1, 1)
    y = daily_avg_close["close"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    X_future = np.arange(X.max() + 1, X.max() + 366).reshape(-1, 1)
    y_future_pred = model.predict(poly.transform(X_future))

    # Plot forecast
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=X.flatten(), y=y, name="Historical Avg Close", line=dict(color="green")))
    fig_forecast.add_trace(go.Scatter(x=X_future.flatten(), y=y_future_pred,
                                      name="Predicted Avg Close", line=dict(color="blue", dash="dash")))
    fig_forecast.update_layout(title="Historical and Predicted Stock Prices",
                               xaxis_title="Days Since Start",
                               yaxis_title="Average Closing Price",
                               width=1100, height=600, hovermode="x unified")
    st.subheader("📦 Forecast & Backtesting View")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ========== Model Metrics ==========
    st.subheader("📊 Model Metrics")
    st.write(f"R² Score: {r2_score(y, model.predict(poly.transform(X))):.4f}")
    st.write(f"Adjusted R²: {1 - (1 - r2_score(y, model.predict(poly.transform(X)))) * ((len(y) - 1) / (len(y) - X_poly_train.shape[1] - 1)):.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y, model.predict(poly.transform(X)))):.4f}")
    st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y, model.predict(poly.transform(X))):.4f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {np.mean(np.abs((y - model.predict(poly.transform(X))) / y)) * 100:.2f}%")

# ========== Footer ==========
footer = """
<style>
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #eeeeee;
color: #000000;
text-align: center;
padding: 5px 0;
}
</style>
<div class="footer">
<p>Developed by Sami El-Khoury Awaragi, Camelia Ladjeroud, Azan Niazi & Tristan Thai</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
