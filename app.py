import matplotlib
# 1. 強制後端 (必須在最前面，防止 Streamlit 在雲端崩潰)
matplotlib.use('Agg')

import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import os
import requests
from PIL import Image

# 解除圖片像素限制
Image.MAX_IMAGE_PIXELS = None

# --- 頁面設定 ---
st.set_page_config(page_title="專業籌碼分析圖 (V5.2)", layout="wide")
st.title("📊 專業股票技術分析 + 籌碼分布 (Volume Profile)")

# ==========================================
# 解決中文亂碼問題：自動下載並加載中文字體
# ==========================================
@st.cache_resource
def get_chinese_font():
    font_path = "NotoSansTC-Regular.otf"
    # 如果字體檔案不存在，則下載
    if not os.path.exists(font_path):
        # 使用 Adobe 開源繁體中文字體
        url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/TraditionalChinese/SourceHanSansTC-Regular.otf"
        
        try:
            # 下載字體
            r = requests.get(url)
            with open(font_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            # 如果下載失敗，就不勉強，避免程式崩潰
            pass
            
    # 建立字體屬性物件
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    else:
        # 回退使用預設字體
        return fm.FontProperties()

# 獲取字體物件
prop = get_chinese_font()

# ==========================================
# 側邊欄設定
# ==========================================
with st.sidebar:
    st.header("參數設定")
    user_input = st.text_input("股票代號", value="8299", help="輸入代號即可，例如 8299").strip()
    period = st.selectbox("資料區間", ["3mo", "6mo", "1y"], index=1)
    st.info("💡 亮黃色橫線 = 最大籌碼堆積價位 (POC)")
    
    st.divider()
    # 送出按鍵
    run_button = st.button("🚀 送出並開始分析", type="primary")

# ==========================================
# 核心邏輯函數
# ==========================================

# 1. 智慧搜尋股票代號
def smart_download(input_ticker, p, status_container):
    input_ticker = input_ticker.upper()
    targets = []

    # 邏輯判斷
    if ".TW" in input_ticker or ".TWO" in input_ticker:
        # 使用者已經指定了市場，直接搜尋
        targets = [input_ticker]
    elif input_ticker.isdigit():
        # 如果是純數字，先試 TW (上市)，再試 TWO (上櫃)
        targets = [f"{input_ticker}.TW", f"{input_ticker}.TWO"]
    else:
        # 其他情況 (如美股代號)
        targets = [input_ticker]

    # 迴圈測試代號
    for t in targets:
        status_container.text(f"🔍 正在搜尋代號: {t} ...")
        try:
            df = yf.download(t, period=p, progress=False, auto_adjust=False)
            
            # 檢查資料有效性
            if not df.empty and len(df) > 10:
                # 處理 MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index)
                
                status_container.text(f"✅ 成功獲取資料: {t}")
                return df, t # 回傳資料與該有效代號
        except:
            continue
            
    return None, None

# 2. 繪圖函數
def create_chart_final(df, symbol):
    # --- 計算指標 ---
    close = df['Close']
    df['MA5'] = close.rolling(5).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    df['STD20'] = close.rolling(20).std()
    df['BB_Up'] = df['MA20'] + 2 * df['STD20']
    df['BB_Lo'] = df['MA20'] - 2 * df['STD20']
    
    # --- 計算籌碼 POC ---
    price_bins = np.linspace(df['Low'].min(), df['High'].max(), 100)
    hist, edges = np.histogram(df['Close'], bins=price_bins, weights=df['Volume'])
    max_idx = np.argmax(hist)
    poc = (edges[max_idx] + edges[max_idx+1]) / 2

    # --- 風格與顏色 ---
    # 自訂鮮豔紅綠
    mc = mpf.make_marketcolors(up='#FF3333', down='#00B060', edge='inherit', wick='inherit', volume='inherit')
    # 格線樣式
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle=':', gridcolor='#D0D0D0', y_on_right=True)
    
    # 均線顏色
    mav_colors = ['#1f77b4', '#ff7f0e', '#9467bd']
    
    # 布林通道
    apds = [
        mpf.make_addplot(df['BB_Up'], color='grey', linestyle='--', width=1, alpha=0.6),
        mpf.make_addplot(df['BB_Lo'], color='grey', linestyle='--', width=1, alpha=0.6)
    ]

    # --- 繪圖 (關閉 tight_layout 以便手動調整間距) ---
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=s,
        volume=True,
        addplot=apds,
        mav=(5, 20, 60),
        mavcolors=mav_colors,
        figsize=(16, 10),
        panel_ratios=(2, 1), # 上下圖比例 2:1
        returnfig=True,
        tight_layout=False 
    )

    # --- 【關鍵修正】拉開 K線圖 與 交易量圖 的距離 ---
    fig.subplots_adjust(hspace=0.4) 

    # --- 手動設定標題 (使用中文字體) ---
    ax_main = axes[0]
    ax_vol = axes[2]
    
    ax_main.set_title(f"{symbol} 籌碼分布與技術分析", fontproperties=prop, fontsize=22, pad=20)
    ax_main.set_ylabel("價格 (Price)", fontproperties=prop, fontsize=12)
    ax_vol.set_ylabel("成交量 (Volume)", fontproperties=prop, fontsize=12)

    # --- 疊加 POC (亮黃色) ---
    ax_vp = ax_main.twiny()
    
    ax_vp.barh(
        y=edges[:-1],
        width=hist,
        height=np.diff(edges)*0.9,
        align='edge',
        color='skyblue',
        alpha=0.3,
        edgecolor='#87CEEB',
        linewidth=0.5,
        zorder=0
    )
    
    # 【POC 亮黃色修正】
    ax_main.axhline(poc, color='#FFFF00', linewidth=3.0, alpha=1.0, zorder=10, linestyle='-')
    
    # POC 文字標籤
    ax_main.text(
        df.index[-1], poc, f' POC: {poc:.2f} ',
        color='black',
        fontweight='bold',
        backgroundcolor='#FFFF00',
        verticalalignment='center',
        zorder=11
    )
    
    ax_vp.set_xlim(0, max(hist) * 3.5)
    ax_vp.axis('off')
    
    return fig, poc, df['Close'].iloc[-1]

# ==========================================
# 主程式 (按下按鈕後才執行)
# ==========================================
if run_button:
    # 建立進度提示區
    status_box = st.empty()
    status_box.text("🚀 系統啟動...")
    
    if not user_input:
        status_box.error("請輸入股票代號！")
    else:
        # 1. 執行智慧搜尋
        df, valid_symbol = smart_download(user_input, period, status_box)
        
        if df is None:
            status_box.empty() # 清空進度文字
            st.error(f"❌ 查無資料：已嘗試搜尋 '{user_input}.TW' 與 '{user_input}.TWO' 皆無結果。")
            st.warning("請確認代號是否正確，或該股票是否已下市。")
        else:
            status_box.text(f"🎨 正在繪製 {valid_symbol} 圖表...")
            
            try:
                fig, poc_price, last_price = create_chart_final(df, valid_symbol)
                
                status_box.text("✅ 繪圖完成，渲染圖片中...")
                
                # --- 版面配置：置中圖片 ---
                c1, c2, c3 = st.columns([1, 10, 1]) 
                
                with c2:
                    # 顯示數據指標
                    m1, m2 = st.columns(2)
                    m1.metric("最新收盤價", f"{last_price:.2f}")
                    m2.metric("最大籌碼堆積 (POC)", f"{poc_price:.2f}")
                    
                    # 顯示圖片
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=120)
                    buf.seek(0)
                    
                    st.image(buf, use_container_width=True)
                
                status_box.success(f"✨ 分析完成！代號: {valid_symbol}")
                
                # 釋放記憶體
                plt.close(fig)
                buf.close()
                
            except Exception as e:
                status_box.error("❌ 程式發生意外錯誤")
                st.error(f"錯誤詳情: {e}")
