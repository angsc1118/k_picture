import matplotlib
# 1. 強制後端，防止 Streamlit 崩潰
matplotlib.use('Agg')

import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# 解除像素限制
Image.MAX_IMAGE_PIXELS = None

# --- 頁面設定 ---
st.set_page_config(page_title="專業籌碼分析圖", layout="wide")
st.title("📊 專業股票技術分析 + 籌碼分布 (Volume Profile)")
st.markdown("風格仿照專業看盤軟體，優化視覺體驗。")

# --- 側邊欄 ---
with st.sidebar:
    st.header("設定")
    # 預設使用您提供的例子 3167
    ticker = st.text_input("股票代號", value="3167.TW").upper()
    period = st.selectbox("資料區間", ["3mo", "6mo", "1y"], index=1)
    st.info("💡 橘色粗線為最大籌碼堆積價位 (POC)")

# --- 資料下載 (含快取) ---
@st.cache_data(ttl=300)
def get_data_final(symbol, p):
    try:
        df = yf.download(symbol, period=p, progress=False, auto_adjust=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df if len(df) > 20 else None
    except:
        return None

# --- 核心繪圖邏輯 (美化版) ---
def create_chart_aesthetic(df, symbol):
    # 1. 計算指標
    close = df['Close']
    df['MA5'] = close.rolling(5).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    df['STD20'] = close.rolling(20).std()
    df['BB_Up'] = df['MA20'] + 2 * df['STD20']
    df['BB_Lo'] = df['MA20'] - 2 * df['STD20']
    
    # 2. 計算籌碼分布
    price_bins = np.linspace(df['Low'].min(), df['High'].max(), 100) # 切細一點，100格
    hist, edges = np.histogram(df['Close'], bins=price_bins, weights=df['Volume'])
    max_idx = np.argmax(hist)
    poc = (edges[max_idx] + edges[max_idx+1]) / 2

    # --- 3. 美化樣式設定 (關鍵) ---
    # 定義更鮮豔的台股紅綠色
    my_colors = mpf.make_marketcolors(
        up='#FF3333', down='#00B060', # 鮮紅、鮮綠
        edge='inherit', wick='inherit', volume='inherit'
    )
    # 定義風格：背景白、格線極淡的虛線
    my_style = mpf.make_mpf_style(
        base_mpf_style='yahoo', 
        marketcolors=my_colors, 
        gridstyle=':', gridcolor='#E0E0E0', # 極淡灰色虛線格網
        y_on_right=True
    )
    
    # 設定均線顏色 (藍、橘、紫)
    mav_colors = ['#1f77b4', '#ff7f0e', '#9467bd']

    # 布林通道設定 (灰色虛線)
    apds = [
        mpf.make_addplot(df['BB_Up'], color='grey', linestyle='--', width=1, alpha=0.7),
        mpf.make_addplot(df['BB_Lo'], color='grey', linestyle='--', width=1, alpha=0.7)
    ]

    # --- 4. 繪製主圖 ---
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=my_style,
        volume=True,
        addplot=apds,
        mav=(5, 20, 60),
        mavcolors=mav_colors, # 套用自訂均線色
        figsize=(16, 9),      # 16:9 寬螢幕比例
        panel_ratios=(2, 1),
        title=dict(title=f"\n{symbol} 籌碼分布分析圖", color='black', size=14),
        ylabel='價格 (TWD)',
        ylabel_lower='成交量',
        returnfig=True,
        tight_layout=True     # 重新啟用緊湊佈局，讓邊距更好看
    )

    # --- 5. 疊加美化後的籌碼圖 ---
    ax_main = axes[0]
    ax_vp = ax_main.twiny()
    
    # 繪製橫向長條圖 (加入邊框細節)
    ax_vp.barh(
        y=edges[:-1],
        width=hist,
        height=np.diff(edges)*0.9, # 高度乘以 0.9 讓條與條之間有一點點縫隙
        align='edge',
        color='skyblue',    # 主體顏色
        alpha=0.35,         # 透明度
        edgecolor='#87CEEB',# 加上淺藍色邊框，增加層次感
        linewidth=0.5,      # 邊框寬度
        zorder=0
    )
    
    # 繪製 POC 橘色粗線
    ax_main.axhline(poc, color='#FF8C00', linewidth=2.5, alpha=0.9, zorder=10)
    
    # 加入 POC 標籤色塊
    ax_main.text(
        df.index[-1], poc, f' POC: {poc:.2f} ',
        color='white',
        fontweight='bold',
        backgroundcolor='#FF8C00', # 橘色背景色塊
        verticalalignment='center',
        horizontalalignment='left',
        zorder=11
    )
    
    # 設定籌碼圖範圍與隱藏刻度
    ax_vp.set_xlim(0, max(hist) * 3.5)
    ax_vp.axis('off')
    
    return fig, poc, df['Close'].iloc[-1]

# --- 主程式 ---
if ticker:
    with st.spinner(f"正在分析 {ticker}，請稍候..."):
        df = get_data_final(ticker, period)
        
    if df is not None:
        try:
            fig, poc_price, last_price = create_chart_aesthetic(df, ticker)
            
            # 顯示數據指標
            col1, col2 = st.columns(2)
            col1.metric("最新收盤價", f"{last_price:.2f}")
            col2.metric("最大籌碼堆積價 (POC)", f"{poc_price:.2f}", delta_color="off")
            
            # 轉換並顯示圖片
            buf = io.BytesIO()
            # 這裡我們依賴 mplfinance 內部的 tight_layout，所以存檔時不加 bbox_inches
            fig.savefig(buf, format='png', dpi=120) 
            buf.seek(0)
            st.image(buf, use_container_width=True)
            
            plt.close(fig)
            buf.close()
        except Exception as e:
            st.error(f"繪圖過程發生錯誤: {e}")
    else:
        st.error(f"無法取得 {ticker} 的資料，請確認代號或網路連線。")
