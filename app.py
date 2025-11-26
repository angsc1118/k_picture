import matplotlib
matplotlib.use('Agg')

import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- 頁面設定 ---
st.set_page_config(page_title="股票籌碼分析儀", layout="wide")
st.title("📊 股票技術分析 + 籌碼分佈圖")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("參數設定")
    ticker = st.text_input("股票代號", value="3167.TW").upper()
    period = st.selectbox("資料區間", ["3mo", "6mo", "1y"], 1)
    st.info("💡 橘色線 = 分價量最大堆積價（POC）")

# --- 下載資料 ---
@st.cache_data(ttl=300)
def load_data(symbol, period):
    df = yf.download(symbol, period=period, auto_adjust=False, multi_level_index=False)

    if df.empty:
        return None

    # 防止 column 是 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    return df


# --- 作圖 ---
def plot_chart(df, symbol):

    # === 計算布林通道 ===
    close = df['Close'].astype(float)
    df['MA20'] = close.rolling(20).mean()
    df['STD20'] = close.rolling(20).std()
    df['BB_Upper'] = df['MA20'] + 2 * df['STD20']
    df['BB_Lower'] = df['MA20'] - 2 * df['STD20']

    # === 分價量計算 ===
    price_bins = np.linspace(df['Low'].min(), df['High'].max(), 80)
    hist_vol, bin_edges = np.histogram(
        df['Close'], bins=price_bins, weights=df['Volume']
    )

    # 計算 POC
    max_vol_idx = np.argmax(hist_vol)
    poc_price = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx + 1]) / 2

    # === 分價量轉可視化資料 ===
    vp_y = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin 中心
    vp_scaled = hist_vol / hist_vol.max() * (df['High'].max() - df['Low'].min()) * 0.15

    vp_series = pd.Series(vp_scaled + df['Low'].min(), index=vp_y)

    # === mplfinance 風格 ===
    mc = mpf.make_marketcolors(up='r', down='g')
    style = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

    add_plots = [
        mpf.make_addplot(df['BB_Upper'], color='grey', linestyle='--'),
        mpf.make_addplot(df['BB_Lower'], color='grey', linestyle='--'),
        mpf.make_addplot(vp_series, type='bar', width=0.7, color='skyblue', alpha=0.3)
    ]

    # === 繪製圖表（最穩定版）===
    fig, axes = mpf.plot(
        df,
        type='candle',
        volume=True,
        style=style,
        mav=(5, 20, 60),
        addplot=add_plots,
        figsize=(14, 8),
        returnfig=True
    )

    # === 加 POC 水平線 ===
    axes[0].axhline(poc_price, color='orange', linewidth=2)

    return fig, poc_price


# --- 主程式 ---
df = load_data(ticker, period)

if df is None or len(df) < 20:
    st.warning("資料不足或代號錯誤，請確認（例如：2330.TW）")
else:
    last_close = df['Close'].iloc[-1]

    col1, col2 = st.columns(2)
    col1.metric("最新收盤", f"{last_close:.2f}")

    fig, poc = plot_chart(df, ticker)

    # === 使用 BytesIO 安全輸出到 Streamlit Cloud ===
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)

    st.image(buf, use_container_width=True)
    col2.metric("POC（最大籌碼堆積價）", f"{poc:.2f}")

    plt.close(fig)
    buf.close()
