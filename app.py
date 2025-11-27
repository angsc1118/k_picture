import matplotlib
# 1. 強制後端 (防止 Streamlit 崩潰)
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
st.set_page_config(page_title="專業籌碼分析 (精確演算版)", layout="wide")
st.title("📊 專業股票技術分析 + 精確籌碼分布 (Volume Profile)")
st.markdown("""
此版本實作了 **Tick-by-Tick 精確級距** 與 **K棒均勻分佈演算法**。
不再使用模糊的區間概算，而是模擬真實交易所掛單價位，提供最精準的支撐壓力分析。
""")

# ==========================================
# 0. 中文字體處理
# ==========================================
@st.cache_resource
def get_chinese_font():
    font_path = "NotoSansTC-Regular.otf"
    if not os.path.exists(font_path):
        url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/TraditionalChinese/SourceHanSansTC-Regular.otf"
        try:
            r = requests.get(url)
            with open(font_path, 'wb') as f:
                f.write(r.content)
        except:
            pass
    return fm.FontProperties(fname=font_path) if os.path.exists(font_path) else fm.FontProperties()

prop = get_chinese_font()

# ==========================================
# 1. 核心演算法：精確籌碼計算 (Method B + C)
# ==========================================

# (Method B) 定義台股價格跳動單位 (Tick Size)
def get_tw_tick(price):
    if price < 10: return 0.01
    elif price < 50: return 0.05
    elif price < 100: return 0.1
    elif price < 500: return 0.5
    elif price < 1000: return 1.0
    else: return 5.0

# (Method B) 產生符合交易所規則的價格網格
def generate_tick_bins(low_price, high_price):
    # 確保範圍稍微擴大一點點，以免最高價被切掉
    current = low_price
    bins = [current]
    
    # 為了防止無窮迴圈(極端狀況)，設一個安全上限
    max_steps = 10000 
    steps = 0
    
    while current < high_price and steps < max_steps:
        tick = get_tw_tick(current)
        current = round(current + tick, 2) # 修正浮點數誤差
        bins.append(current)
        steps += 1
        
    return np.array(bins)

# (Method C) 均勻分佈成交量演算法
def calculate_precise_volume_profile(df):
    min_p = df['Low'].min()
    max_p = df['High'].max()
    
    # 1. 產生真實價格網格 (Edges)
    edges = generate_tick_bins(min_p, max_p)
    
    # 初始化籌碼桶 (Volume Buckets)
    # bucket 數量比 edge 少 1
    vol_hist = np.zeros(len(edges) - 1)
    
    # 2. 逐日分配成交量 (Uniform Distribution)
    # 雖然用迴圈較慢，但對於 1年約 250 筆資料來說，計算時間 < 0.05秒，可忽略
    lows = df['Low'].values
    highs = df['High'].values
    vols = df['Volume'].values
    
    for i in range(len(df)):
        day_low = lows[i]
        day_high = highs[i]
        day_vol = vols[i]
        
        if day_vol == 0: continue
        
        # 找出當日股價範圍涵蓋了哪些 Bins
        # searchsorted: 找出 day_low 在 edges 中的插入位置
        # side='right' - 1 確保包含 day_low 所在的那個 bin
        start_idx = np.searchsorted(edges, day_low, side='right') - 1
        # side='left' 確保包含 day_high 截止的那個 bin
        end_idx = np.searchsorted(edges, day_high, side='left')
        
        # 邊界保護 (防止當日股價超出歷史範圍，雖然理論上不可能)
        start_idx = max(0, start_idx)
        end_idx = min(len(vol_hist), end_idx)
        
        # 特殊情況：High == Low (十字線或一字線)，或者範圍極小
        if end_idx <= start_idx:
            end_idx = start_idx + 1
            
        # 計算涵蓋的格數
        num_bins = end_idx - start_idx
        
        # 平均分配成交量 (Method C 核心)
        if num_bins > 0:
            avg_vol = day_vol / num_bins
            vol_hist[start_idx:end_idx] += avg_vol
            
    return vol_hist, edges

# ==========================================
# 2. 繪圖與數據處理
# ==========================================

# 智慧搜尋
def smart_download(input_ticker, p, status_container):
    input_ticker = input_ticker.upper()
    targets = [input_ticker] if (".TW" in input_ticker or ".TWO" in input_ticker) else [f"{input_ticker}.TW", f"{input_ticker}.TWO"]
    if not input_ticker.isdigit() and ".TW" not in input_ticker: targets = [input_ticker] # 美股容錯

    for t in targets:
        status_container.text(f"🔍 搜尋: {t} ...")
        try:
            df = yf.download(t, period=p, progress=False, auto_adjust=False)
            if not df.empty and len(df) > 10:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index)
                return df, t
        except: continue
    return None, None

# 繪圖主函數
def create_chart_precise(df, symbol):
    # 指標
    close = df['Close']
    df['MA5'] = close.rolling(5).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    df['STD20'] = close.rolling(20).std()
    df['BB_Up'] = df['MA20'] + 2 * df['STD20']
    df['BB_Lo'] = df['MA20'] - 2 * df['STD20']
    
    # --- 呼叫精確演算法計算 POC ---
    hist, edges = calculate_precise_volume_profile(df)
    
    # 找出 POC
    max_idx = np.argmax(hist)
    poc = (edges[max_idx] + edges[max_idx+1]) / 2

    # 風格設定
    mc = mpf.make_marketcolors(up='#FF3333', down='#00B060', edge='inherit', wick='inherit', volume='inherit')
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle=':', gridcolor='#D0D0D0', y_on_right=True)
    mav_colors = ['#1f77b4', '#ff7f0e', '#9467bd']
    
    apds = [
        mpf.make_addplot(df['BB_Up'], color='grey', linestyle='--', width=1, alpha=0.6),
        mpf.make_addplot(df['BB_Lo'], color='grey', linestyle='--', width=1, alpha=0.6)
    ]

    # 繪圖
    fig, axes = mpf.plot(
        df, type='candle', style=s, volume=True, addplot=apds,
        mav=(5, 20, 60), mavcolors=mav_colors,
        figsize=(16, 10), panel_ratios=(2, 1),
        returnfig=True, tight_layout=False
    )
    
    # 拉開間距
    fig.subplots_adjust(hspace=0.4) 

    # 設定標題
    ax_main = axes[0]
    ax_vol = axes[2]
    ax_main.set_title(f"{symbol} 精確籌碼分布分析", fontproperties=prop, fontsize=22, pad=20)
    ax_main.set_ylabel("價格", fontproperties=prop, fontsize=12)
    ax_vol.set_ylabel("成交量", fontproperties=prop, fontsize=12)

    # --- 疊加 POC (使用計算出的精確 Bins) ---
    ax_vp = ax_main.twiny()
    ax_vp.barh(
        y=edges[:-1],
        width=hist,
        height=np.diff(edges)*0.9, # 留一點縫隙
        align='edge',
        color='skyblue', alpha=0.3, edgecolor='#87CEEB', linewidth=0.5, zorder=0
    )
    
    # POC 黃色亮線
    ax_main.axhline(poc, color='#FFFF00', linewidth=3.0, alpha=1.0, zorder=10)
    
    # POC 標籤
    ax_main.text(
        df.index[-1], poc, f' POC: {poc:.2f} ',
        color='black', fontweight='bold', backgroundcolor='#FFFF00',
        verticalalignment='center', zorder=11
    )
    
    ax_vp.set_xlim(0, max(hist) * 3.5)
    ax_vp.axis('off')
    
    return fig, poc, df['Close'].iloc[-1]

# ==========================================
# 3. 側邊欄與執行
# ==========================================
with st.sidebar:
    st.header("參數設定")
    user_input = st.text_input("股票代號", value="8299").strip()
    period = st.selectbox("資料區間", ["3mo", "6mo", "1y"], index=1)
    st.info("💡 採用 TWSE 真實跳動級距演算法")
    st.divider()
    run_button = st.button("🚀 開始精確分析", type="primary")

if run_button:
    status_box = st.empty()
    status_box.text("🚀 初始化演算法...")
    
    if not user_input:
        status_box.error("請輸入代號")
    else:
        df, valid_symbol = smart_download(user_input, period, status_box)
        
        if df is None:
            status_box.empty()
            st.error(f"❌ 查無資料: {user_input}")
        else:
            status_box.text(f"🧮 正在進行 Tick-by-Tick 籌碼運算 ({len(df)} 筆資料)...")
            
            try:
                fig, poc_price, last_price = create_chart_precise(df, valid_symbol)
                
                status_box.text("✅ 運算完成，渲染圖表中...")
                c1, c2, c3 = st.columns([1, 10, 1])
                with c2:
                    m1, m2 = st.columns(2)
                    m1.metric("最新收盤", f"{last_price:.2f}")
                    m2.metric("精確 POC 價位", f"{poc_price:.2f}")
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=120)
                    buf.seek(0)
                    st.image(buf, use_container_width=True)
                
                status_box.success(f"✨ 分析完成: {valid_symbol}")
                plt.close(fig)
                buf.close()
                
            except Exception as e:
                status_box.error("運算錯誤")
                st.error(str(e))
