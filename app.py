import matplotlib
# 1. 強制使用非互動式後端 (防止 Server 崩潰)
matplotlib.use('Agg') 

import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io # 新增：用於記憶體緩衝
from PIL import Image # 新增：用於解除圖片限制

# 2. 解除 PIL 的圖片大小限制 (防止 DecompressionBombError)
Image.MAX_IMAGE_PIXELS = None 

# --- 頁面設定 ---
st.set_page_config(page_title="股票籌碼分析儀", layout="wide")

st.title("📊 股票技術分析 + 籌碼分布圖")

# --- 側邊欄輸入 ---
with st.sidebar:
    st.header("參數設定")
    ticker = st.text_input("股票代號", value="3167.TW").upper()
    period = st.selectbox("資料區間", options=["3mo", "6mo", "1y"], index=1)
    st.info("💡 橘色橫線 = 最大成交量價位 (POC)")

# --- 函數：下載數據 ---
@st.cache_data(ttl=3600)
def load_data(symbol, time_period):
    try:
        # 加入 auto_adjust=False 防止格式警告
        df = yf.download(symbol, period=time_period, auto_adjust=False, multi_level_index=False)
        
        # 二次防護：如果還是多層索引，手動處理
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.index = pd.to_datetime(df.index)
        
        if df.empty:
            return None
        return df
    except Exception as e:
        return None

# --- 函數：繪圖邏輯 ---
def plot_chart(df, symbol):
    # 1. 計算布林通道
    close_price = df['Close'].astype(float)
    df['MA20'] = close_price.rolling(window=20).mean()
    df['STD20'] = close_price.rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (2 * df['STD20'])
    df['BB_Lower'] = df['MA20'] - (2 * df['STD20'])

    # 2. 計算分價量 (Volume Profile)
    try:
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), num=100)
        hist_vol, bin_edges = np.histogram(
            df['Close'].values, 
            bins=price_bins, 
            weights=df['Volume'].values
        )
        
        max_vol_idx = np.argmax(hist_vol)
        poc_price = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx+1]) / 2
    except:
        return None, 0

    # 3. 設定 mplfinance 風格
    mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
    s  = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle=':', y_on_right=True)

    apds = [
        mpf.make_addplot(df['BB_Upper'], color='grey', linestyle='--', width=0.8),
        mpf.make_addplot(df['BB_Lower'], color='grey', linestyle='--', width=0.8),
    ]

    # 4. 繪製主圖
    # 注意：這裡的 figsize 設定適中即可
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=s,
        title=f"\n{symbol} Analysis",
        ylabel='Price',
        volume=True,
        mav=(5, 20, 60),
        addplot=apds,
        figsize=(14, 8), 
        tight_layout=True,
        panel_ratios=(2, 1),
        returnfig=True 
    )

    # 5. 疊加籌碼分布
    ax_main = axes[0]
    ax_vp = ax_main.twiny()
    
    ax_vp.barh(
        y=bin_edges[:-1],           
        width=hist_vol,             
        height=np.diff(bin_edges),  
        align='edge',
        color='skyblue',
        alpha=0.25,                 
        zorder=0                    
    )
    
    ax_main.axhline(
        y=poc_price, 
        color='orange', 
        linewidth=2.5, 
        linestyle='-',
        label='POC'
    )
    
    ax_vp.set_xlim(0, max(hist_vol) * 3) 
    ax_vp.axis('off')

    ax_main.text(
        x=df.index[-1], 
        y=poc_price, 
        s=f' POC: {poc_price:.2f}', 
        color='orange', 
        fontweight='bold', 
        verticalalignment='bottom'
    )

    return fig, poc_price

# --- 主程式 ---
if ticker:
    df = load_data(ticker, period)
        
    if df is not None and len(df) > 20:
        last_close = df['Close'].iloc[-1]
        col1, col2 = st.columns(2)
        col1.metric("最新收盤", f"{last_close:.2f}")
        
        # 繪圖
        fig, poc = plot_chart(df, ticker)
        
        if fig:
            # ========================================================
            # 【核心修正】不使用 st.pyplot(fig) 
            # 改用 BytesIO + 指定 DPI，徹底解決 DecompressionBombError
            # ========================================================
            buf = io.BytesIO()
            # dpi=100 確保圖片大小適中，不會超過 PIL 限制
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            st.image(buf, use_container_width=True) # 顯示圖片
            
            col2.metric("最大籌碼堆積 (POC)", f"{poc:.2f}")
            
            # 釋放記憶體
            plt.close(fig) 
            buf.close()
        else:
            st.error("繪圖失敗，請檢查數據源。")
        
    else:
        st.warning("找不到資料或資料不足。請確認代號 (如 2330.TW)。")
