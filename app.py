import matplotlib
# 【重要】強制使用無介面後端，防止 Streamlit 崩潰
matplotlib.use('Agg')

import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# 解除圖片像素限制
Image.MAX_IMAGE_PIXELS = None

# --- 頁面設定 ---
st.set_page_config(page_title="股票籌碼分析儀", layout="wide")
st.title("📊 股票技術分析 + 籌碼分布圖 (Volume Profile)")

# --- 側邊欄 ---
with st.sidebar:
    st.header("參數設定")
    ticker = st.text_input("股票代號", value="3167.TW").upper()
    period = st.selectbox("資料區間", ["3mo", "6mo", "1y"], index=1)
    st.info("💡 確保代號正確，例如台股需加 .TW")

# --- 1. 穩健的資料下載函數 ---
@st.cache_data(ttl=600)
def get_data(symbol, period):
    try:
        # 下載資料，強制關閉多層索引
        df = yf.download(symbol, period=period, auto_adjust=False, progress=False)
        
        # 資料清洗：處理 yfinance 的多層欄位問題
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 確保是日期索引
        df.index = pd.to_datetime(df.index)
        
        # 檢查資料是否為空或過少
        if df.empty or len(df) < 20:
            return None
            
        return df
    except Exception as e:
        st.error(f"資料下載失敗: {e}")
        return None

# --- 2. 核心繪圖函數 ---
def create_plot(df, symbol):
    # --- 計算技術指標 ---
    close = df['Close']
    df['MA20'] = close.rolling(20).mean()
    df['STD20'] = close.rolling(20).std()
    df['BB_Up'] = df['MA20'] + 2 * df['STD20']
    df['BB_Lo'] = df['MA20'] - 2 * df['STD20']

    # --- 計算籌碼分布 (Volume Profile) ---
    # 這是畫出「橫向」柱狀圖的數學核心
    price_bins = np.linspace(df['Low'].min(), df['High'].max(), 80)
    hist_vol, bin_edges = np.histogram(df['Close'], bins=price_bins, weights=df['Volume'])
    
    # 找出最大量價位 (POC)
    max_idx = np.argmax(hist_vol)
    poc_price = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2

    # --- 設定 mplfinance 風格 ---
    mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
    s  = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle=':')

    # 附加圖層：布林通道
    apds = [
        mpf.make_addplot(df['BB_Up'], color='grey', linestyle='--', width=0.8),
        mpf.make_addplot(df['BB_Lo'], color='grey', linestyle='--', width=0.8)
    ]

    # --- 【關鍵修改】開始繪圖 ---
    # 注意：這裡將 tight_layout 設為 False，避免與後面的 savefig 衝突
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=s,
        title=f"\n{symbol} Volume Profile",
        ylabel='Price',
        volume=True,
        mav=(5, 20, 60),
        addplot=apds,
        figsize=(14, 8),
        panel_ratios=(2, 1),
        returnfig=True, # 必須為 True 才能讓我們手動畫籌碼圖
        tight_layout=False 
    )

    # --- 手動繪製橫向籌碼圖 ---
    ax_main = axes[0] # 主圖 (K線圖)
    ax_vp = ax_main.twiny() # 建立雙軸 (共用 Y 軸，獨立 X 軸)

    # 畫出藍色橫條
    ax_vp.barh(
        y=bin_edges[:-1],
        width=hist_vol,
        height=np.diff(bin_edges),
        align='edge',
        color='skyblue',
        alpha=0.3,
        zorder=0 # 放在 K 線後面
    )
    
    # 畫出 POC 橘色線
    ax_main.axhline(poc_price, color='orange', linewidth=2.5)
    
    # 調整籌碼圖範圍 (避免蓋住 K 線)
    ax_vp.set_xlim(0, max(hist_vol) * 4) # 設定為最大量的4倍，讓柱子只佔畫面 1/4
    ax_vp.axis('off') # 隱藏上方刻度

    # 標示價格文字
    ax_main.text(
        df.index[-1], poc_price, f' POC: {poc_price:.2f}', 
        color='orange', fontweight='bold', va='bottom'
    )

    return fig, poc_price

# --- 3. 主程式執行邏輯 ---
if ticker:
    st.write(f"正在分析: **{ticker}** ...")
    
    df = get_data(ticker, period)
    
    if df is not None:
        # 顯示最新價格
        last_price = df['Close'].iloc[-1]
        col1, col2 = st.columns(2)
        col1.metric("最新收盤價", f"{last_price:.2f}")
        
        try:
            # 產生圖表物件
            fig, poc = create_plot(df, ticker)
            col2.metric("最大籌碼堆積 (POC)", f"{poc:.2f}")

            # --- 【最後一哩路】將圖表轉為圖片顯示 ---
            buf = io.BytesIO()
            # 這裡使用 bbox_inches='tight' 來裁切多餘白邊，確保內容完整
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight') 
            buf.seek(0)
            
            # 顯示圖片
            st.image(buf, use_container_width=True)
            
            # 清理記憶體
            plt.close(fig)
            buf.close()
            
        except Exception as e:
            st.error(f"繪圖發生錯誤: {e}")
            st.write("建議：嘗試縮短查詢區間，或更換股票代號。")
    else:
        st.error(f"找不到 {ticker} 的資料。")
        st.warning("常見原因：\n1. 股票代號錯誤 (台股請加 .TW)\n2. 該股票近期無交易\n3. Yahoo Finance 暫時連線不穩")
