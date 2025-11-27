import matplotlib
# 1. 強制後端 (必須在最前面)
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

# 解除像素限制
Image.MAX_IMAGE_PIXELS = None

# --- 頁面設定 ---
st.set_page_config(page_title="專業籌碼分析圖 (V5.1)", layout="wide")
st.title("📊 專業股票技術分析 + 籌碼分布 (Volume Profile)")

# ==========================================
# 解決中文亂碼問題：自動下載並加載中文字體
# ==========================================
@st.cache_resource
def get_chinese_font():
    font_path = "NotoSansTC-Regular.ttf"
    # 如果字體檔案不存在，則下載 (使用 Google Fonts 或其他穩定源)
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
        # 為了相容性，這裡使用一個較小的字體檔連結示範，或直接使用系統字體
        # 這裡改用一個較穩定的開源字體連結 (NotoSans TC Regular)
        url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/TraditionalChinese/SourceHanSansTC-Regular.otf"
        
        try:
            with st.spinner("正在下載中文字體以解決亂碼問題..."):
                r = requests.get(url)
                with open(font_path, 'wb') as f:
                    f.write(r.content)
    except:
        pass
        
    # 建立字體屬性物件
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    else:
        # 如果下載失敗，回退到系統預設 (可能會亂碼，但程式不會崩潰)
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
    
    # 新增送出按鍵
    run_button = st.button("送出並開始分析", type="primary")

# ==========================================
# 核心邏輯函數
# ==========================================

# 1. 智慧搜尋股票代號
def smart_download(input_ticker, p, status_container):
    input_ticker = input_ticker.upper()
    
    # 如果使用者已經自己打了 .TW 或 .TWO，就直接用
    if ".TW" in input_ticker:
        targets = [input_ticker]
    elif input_ticker.isdigit():
        # 如果是純數字，先試 TW，再試 TWO
        targets = [f"{input_ticker}.TW", f"{input_ticker}.TWO"]
    else:
        # 其他情況 (如美股) 直接搜尋
        targets = [input_ticker]

    for t in targets:
        status_container.text(f"🔍 正在嘗試搜尋: {t} ...")
        try:
            df = yf.download(t, period=p, progress=False, auto_adjust=False)
            
            # 檢查是否真的有資料
            if not df.empty and len(df) > 10:
                # 處理 MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index)
                
                status_container.text(f"✅ 成功找到資料: {t}")
                return df, t # 回傳資料與正確的代號
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
    
    # 均線顏色 (藍、橘、紫)
    mav_colors = ['#1f77b4', '#ff7f0e', '#9467bd']
    
    # 布林通道 (灰色虛線)
    apds = [
        mpf.make_addplot(df['BB_Up'], color='grey', linestyle='--', width=1, alpha=0.6),
        mpf.make_addplot(df['BB_Lo'], color='grey', linestyle='--', width=1, alpha=0.6)
    ]

    # --- 繪圖 (關閉 tight_layout 以便我們手動調整間距) ---
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=s,
        volume=True,
        addplot=apds,
        mav=(5, 20, 60),
        mavcolors=mav_colors,
        figsize=(16, 10), # 拉高一點高度
        panel_ratios=(2, 1), # 主圖:成交量 = 2:1
        returnfig=True,
        tight_layout=False # 關閉自動佈局，讓我們手動調整間距
    )

    # --- 【關鍵修正】拉開 K線圖 與 交易量圖 的距離 ---
    # hspace 控制垂直間距 (預設通常是 0)
    fig.subplots_adjust(hspace=0.4) 

    # --- 手動設定標題 (使用中文字體) ---
    ax_main = axes[0]
    ax_vol = axes[2] # mplfinance 中，成交量通常是 ax[2] (因為 ax[1] 是副軸)
    
    ax_main.set_title(f"{symbol} 籌碼分布與技術分析", fontproperties=prop, fontsize=20, pad=20)
    ax_main.set_ylabel("價格 (Price)", fontproperties=prop, fontsize=12)
    ax_vol.set_ylabel("成交量 (Volume)", fontproperties=prop, fontsize=12)

    # --- 疊加 POC (亮黃色) ---
    # 建立雙軸畫籌碼
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
    ax_main.axhline(poc, color='#FFFF00', linewidth=2.5, alpha=1.0, zorder=10, linestyle='-')
    
    # POC 文字標籤
    ax_main.text(
        df.index[-1], poc, f' POC: {poc:.2f} ',
        color='black',
        fontweight='bold',
        backgroundcolor='#FFFF00', # 背景也用亮黃
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
    # 建立一個進度提示區塊
    status_box = st.empty()
    status_box.text("🚀 系統啟動，準備處理...")
    
    if not user_input:
        st.error("請輸入股票代號！")
    else:
        # 1. 下載與搜尋
        df, valid_symbol = smart_download(user_input, period, status_box)
        
        if df is None:
            status_box.text("❌ 處理結束")
            st.error(f"找不到代號 '{user_input}' (已嘗試 .TW 與 .TWO)。請確認代號是否正確。")
        else:
            status_box.text(f"🎨 正在繪製 {valid_symbol} 的圖表 (計算 POC 中)...")
            
            try:
                fig, poc_price, last_price = create_chart_final(df, valid_symbol)
                
                # 2. 顯示數據
                status_box.text("✅ 繪圖完成，正在渲染圖片...")
                
                # 使用 Columns 置中顯示圖片
                # c1 (左邊界), c2 (中間內容), c3 (右邊界)
                c1, c2, c3 = st.columns([1, 10, 1]) 
                
                with c2:
                    # 顯示 Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("最新收盤價", f"{last_price:.2f}")
                    m2.metric("最大籌碼堆積 (POC)", f"{poc_price:.2f}")
                    
                    # 顯示圖片
                    buf = io.BytesIO()
                    # bbox_inches='tight' 有時會導致截圖不全，但因為我們手動調整了 hspace，
                    # 這裡可以試著不加 tight，或者小心使用。
                    # 安全起見，這裡不加 tight，保留完整畫布
                    fig.savefig(buf, format='png', dpi=120)
                    buf.seek(0)
                    
                    st.image(buf, use_container_width=True)
                
                status_box.success(f"✨ 分析完成！代號: {valid_symbol}")
                
                plt.close(fig)
                buf.close()
                
            except Exception as e:
                status_box.error("❌ 程式發生意外錯誤")
                st.error(f"錯誤詳情: {e}")
