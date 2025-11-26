import matplotlib
# 1. 強制後端 (必須在最前面)
matplotlib.use('Agg')

import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# 2. 解除像素限制
Image.MAX_IMAGE_PIXELS = None

# --- 頁面設定 ---
st.set_page_config(page_title="股票籌碼分析除錯版", layout="wide")
st.title("📊 股票籌碼分析 (V4.0 除錯模式)")

# --- 側邊欄 ---
with st.sidebar:
    st.header("設定")
    ticker = st.text_input("股票代號", value="2330.TW").upper() # 改用台積電測試，確保一定有量
    period = st.selectbox("區間", ["3mo", "6mo"], index=0)
    st.warning("若看到文字但沒看到圖，請檢查下方錯誤訊息")

# --- 下載數據 ---
@st.cache_data(ttl=60) # 縮短快取方便測試
def get_data_debug(symbol, p):
    st.write(f"📡 正在連接 Yahoo Finance 下載 {symbol}...")
    try:
        df = yf.download(symbol, period=p, progress=False, auto_adjust=False)
        
        if df.empty:
            st.error("❌ 下載成功但資料為空 (Empty DataFrame)")
            return None
            
        # 處理 MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        st.success(f"✅ 數據獲取成功: 共 {len(df)} 筆交易日")
        # 顯示前幾筆資料確認數據真的存在
        with st.expander("查看原始數據 (Debug)"):
            st.dataframe(df.head())
            
        return df
    except Exception as e:
        st.error(f"❌ 數據下載崩潰: {e}")
        return None

# --- 繪圖邏輯 ---
def create_chart_debug(df, symbol):
    st.write("🎨 開始繪圖計算...")
    
    # 指標計算
    try:
        # 確保是 Series 運算
        close = df['Close']
        df['MA20'] = close.rolling(20).mean()
        df['STD20'] = close.rolling(20).std()
        df['BB_Up'] = df['MA20'] + 2 * df['STD20']
        df['BB_Lo'] = df['MA20'] - 2 * df['STD20']
        
        # 籌碼計算
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), 80)
        hist, edges = np.histogram(df['Close'], bins=price_bins, weights=df['Volume'])
        max_idx = np.argmax(hist)
        poc = (edges[max_idx] + edges[max_idx+1]) / 2
        st.write(f"🔢 POC 計算完成: {poc:.2f}")
    except Exception as e:
        st.error(f"❌ 指標計算錯誤: {e}")
        return None

    # 設定風格
    mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    apds = [
        mpf.make_addplot(df['BB_Up'], color='grey', linestyle='--'),
        mpf.make_addplot(df['BB_Lo'], color='grey', linestyle='--')
    ]

    st.write("🖌️ 正在呼叫 mplfinance plot...")
    
    # 建立圖表 (關鍵：關閉 tight_layout)
    try:
        fig, axes = mpf.plot(
            df,
            type='candle',
            style=s,
            volume=True,
            addplot=apds,
            mav=(5, 20),
            figsize=(12, 8), # 縮小一點確保安全
            returnfig=True,
            tight_layout=False 
        )
    except Exception as e:
        st.error(f"❌ mpf.plot 失敗: {e}")
        return None

    # 疊加籌碼圖
    try:
        ax_main = axes[0]
        ax_vp = ax_main.twiny()
        
        ax_vp.barh(
            y=edges[:-1],
            width=hist,
            height=np.diff(edges),
            align='edge',
            color='skyblue',
            alpha=0.3
        )
        
        # 畫 POC
        ax_main.axhline(poc, color='orange', linewidth=2)
        
        # 設定範圍
        ax_vp.set_xlim(0, max(hist) * 4)
        ax_vp.axis('off')
        
    except Exception as e:
        st.error(f"❌ 疊圖失敗: {e}")
        # 就算疊圖失敗，我們也試著回傳 fig，至少看得到 K 線
        pass 

    return fig

# --- 主執行區 ---
if ticker:
    df = get_data_debug(ticker, period)
    
    if df is not None:
        fig = create_chart_debug(df, ticker)
        
        if fig:
            st.write("💾 正在轉換圖片 (Buffer)...")
            try:
                buf = io.BytesIO()
                # 【絕對關鍵】移除 bbox_inches='tight'，這是最可能導致圖片空白的原因
                fig.savefig(buf, format='png', dpi=100) 
                buf.seek(0)
                
                st.write("🖼️ 準備顯示圖片...")
                st.image(buf, use_container_width=True)
                st.success("✨ 圖片顯示程序完成")
                
                # 清理
                plt.close(fig)
                buf.close()
            except Exception as e:
                st.error(f"❌ 圖片儲存/顯示失敗: {e}")
