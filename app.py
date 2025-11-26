import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 頁面設定 ---
st.set_page_config(page_title="股票籌碼分析儀", layout="wide")

st.title("📊 股票技術分析 + 籌碼分布圖 (Volume Profile)")
st.markdown("輸入台股代號 (如 `8155.TW`)，查看 K 線、布林通道與最大籌碼堆積價位 (POC)。")

# --- 側邊欄輸入 ---
with st.sidebar:
    st.header("參數設定")
    ticker = st.text_input("股票代號", value="8155.TW").upper()
    period = st.selectbox("資料區間", options=["3mo", "6mo", "1y", "2y"], index=1)
    st.info("💡 橘色橫線 = 最大成交量價位 (POC)\n\n藍色長條 = 分價量表")

# --- 核心函數：下載數據 (加入快取機制) ---
@st.cache_data(ttl=3600)
def load_data(symbol, time_period):
    try:
        df = yf.download(symbol, period=time_period)
        
        # 處理 yfinance 新版多層索引問題
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.index = pd.to_datetime(df.index)
        
        if df.empty:
            return None
        return df
    except Exception as e:
        return None

# --- 核心函數：繪圖邏輯 ---
def plot_chart(df, symbol):
    # 1. 計算布林通道
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (2 * df['STD20'])
    df['BB_Lower'] = df['MA20'] - (2 * df['STD20'])

    # 2. 計算分價量 (Volume Profile)
    price_bins = np.linspace(df['Low'].min(), df['High'].max(), num=100)
    hist_vol, bin_edges = np.histogram(
        df['Close'].values, 
        bins=price_bins, 
        weights=df['Volume'].values
    )
    
    # 找出 POC
    max_vol_idx = np.argmax(hist_vol)
    poc_price = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx+1]) / 2

    # 3. 設定 mplfinance 風格
    mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
    s  = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle=':', y_on_right=True)

    # 附加圖層 (布林通道)
    apds = [
        mpf.make_addplot(df['BB_Upper'], color='grey', linestyle='--', width=0.8),
        mpf.make_addplot(df['BB_Lower'], color='grey', linestyle='--', width=0.8),
    ]

    # 4. 繪製主圖 (returnfig=True 是關鍵)
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=s,
        title=f"\n{symbol} Volume Profile & POC Analysis",
        ylabel='Price (TWD)',
        volume=True,
        mav=(5, 20, 60),
        addplot=apds,
        figsize=(18, 10), # 加大尺寸適應網頁
        tight_layout=True,
        panel_ratios=(2, 1),
        returnfig=True 
    )

    # 5. 手動疊加籌碼分布圖
    ax_main = axes[0]
    ax_vp = ax_main.twiny() # 建立雙軸
    
    # 繪製背景藍條
    ax_vp.barh(
        y=bin_edges[:-1],           
        width=hist_vol,             
        height=np.diff(bin_edges),  
        align='edge',
        color='skyblue',
        alpha=0.25,                 
        zorder=0                    
    )
    
    # 繪製橘色 POC 線
    ax_main.axhline(
        y=poc_price, 
        color='orange', 
        linewidth=2.5, 
        linestyle='-',
        label='POC Price'
    )
    
    # 版面微調
    ax_vp.set_xlim(0, max(hist_vol) * 3) 
    ax_vp.axis('off') # 隱藏上方刻度

    # 標示文字
    ax_main.text(
        x=df.index[-1], 
        y=poc_price, 
        s=f' POC: {poc_price:.2f}', 
        color='orange', 
        fontweight='bold', 
        verticalalignment='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.8)
    )

    return fig, poc_price

# --- 主程式邏輯 ---
if ticker:
    with st.spinner(f"正在下載 {ticker} 數據並計算籌碼分佈..."):
        df = load_data(ticker, period)
        
    if df is not None and len(df) > 20:
        # 顯示最新數據摘要
        last_close = df['Close'].iloc[-1]
        last_date = df.index[-1].strftime('%Y-%m-%d')
        
        col1, col2 = st.columns(2)
        col1.metric("收盤價", f"{last_close:.2f}", f"日期: {last_date}")
        
        # 繪圖並顯示
        fig, poc = plot_chart(df, ticker)
        st.pyplot(fig) # Streamlit 專用的繪圖指令
        
        col2.metric("最大籌碼堆積價位 (POC)", f"{poc:.2f}")
        
    else:
        st.error(f"找不到 {ticker} 的資料，或是資料不足 (需大於20筆交易日)。請確認代號是否正確 (例如台股需加 .TW)。")
