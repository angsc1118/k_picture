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

    # === 分價量（Volume Profile）計算 ===
    price_bins = np.linspace(df['Low'].min(), df['High'].max(), 60)
    hist_vol, bin_edges = np.histogram(
        df['Close'], bins=price_bins, weights=df['Volume']
    )

    max_vol_idx = np.argmax(hist_vol)
    poc_price = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx + 1]) / 2

    # === 生成與 df 相同長度的 volume profile "柱狀" addplot ===
    # 右側 volume profile 以 bar 的方式，在價格附近畫出 dummy data
    vp_display = pd.Series(index=df.index, dtype=float)
    vp_display[:] = np.nan  # 先全 NaN

    # 把 hist_vol 正規化後，分布插入到接近價格的位置中
    vp_scaled = hist_vol / hist_vol.max() * (df['High'].max() - df['Low'].min()) * 0.10

    # 將 volume profile 映射到 K 線的最後一根附近，使其看起來像右側分價量
    vp_display.iloc[-len(vp_scaled):] = df['Close'].iloc[-1] + vp_scaled

    # === mplfinance style ===
    mc = mpf.make_marketcolors(up='r', down='g')
    style = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

    add_plots = [
        mpf.make_addplot(df['BB_Upper'], color='grey', linestyle='--'),
        mpf.make_addplot(df['BB_Lower'], color='grey', linestyle='--'),

        # 右側 volume profile（不會觸發錯誤）
        mpf.make_addplot(vp_display, type='bar', width=0.3, color='skyblue', alpha=0.3)
    ]

    # === 畫圖 ===
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

    # === POC 水平線 ===
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

    # === 使用 BytesIO 輸出 ===
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)

    st.image(buf, use_container_width=True)
    col2.metric("POC（最大籌碼堆積價）", f"{poc:.2f}")

    plt.close(fig)
    buf.close()
