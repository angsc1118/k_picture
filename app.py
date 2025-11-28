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
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D 
import io
import os
import requests
from PIL import Image

# 解除圖片像素限制
Image.MAX_IMAGE_PIXELS = None

# --- 頁面設定 ---
st.set_page_config(page_title="專業籌碼分析 Pro", layout="wide")
st.title("📊 專業股票技術分析 + 精確籌碼分布 (Visual Optimized)")
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
</style>
此版本已修復 **Streamlit API 棄用警告** (use_container_width -> width)，確保在最新環境下執行順暢。
""", unsafe_allow_html=True)

# ==========================================
# 0. 中文字體處理 (強制註冊版)
# ==========================================
@st.cache_resource
def get_chinese_font():
    font_path = "NotoSansTC-Regular.otf"
    if not os.path.exists(font_path):
        url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/TraditionalChinese/SourceHanSansTC-Regular.otf"
        try:
            r = requests.get(url, timeout=10)
            with open(font_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            st.warning(f"字體下載失敗: {e}")
    
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        return fm.FontProperties(fname=font_path)
    else:
        return fm.FontProperties()

prop = get_chinese_font()
font_name = prop.get_name() 

# ==========================================
# 1. 核心演算法：精確籌碼計算
# ==========================================

def get_tw_tick(price):
    if price < 10: return 0.01
    elif price < 50: return 0.05
    elif price < 100: return 0.1
    elif price < 500: return 0.5
    elif price < 1000: return 1.0
    else: return 5.0

def generate_tick_bins(low_price, high_price):
    current = low_price
    bins = [current]
    max_steps = 15000 
    steps = 0
    while current < high_price and steps < max_steps:
        tick = get_tw_tick(current)
        current = round(current + tick, 2)
        bins.append(current)
        steps += 1
    return np.array(bins)

def calculate_precise_volume_profile(df):
    min_p = df['Low'].min()
    max_p = df['High'].max()
    edges = generate_tick_bins(min_p, max_p)
    vol_hist = np.zeros(len(edges) - 1)
    
    lows = df['Low'].values
    highs = df['High'].values
    vols = df['Volume'].values
    
    for i in range(len(df)):
        day_low = lows[i]
        day_high = highs[i]
        day_vol = vols[i]
        if day_vol == 0: continue
        
        start_idx = np.searchsorted(edges, day_low, side='right') - 1
        end_idx = np.searchsorted(edges, day_high, side='left')
        
        start_idx = max(0, start_idx)
        end_idx = min(len(vol_hist), end_idx)
        if end_idx <= start_idx: end_idx = start_idx + 1
            
        num_bins = end_idx - start_idx
        if num_bins > 0:
            vol_hist[start_idx:end_idx] += day_vol / num_bins
            
    return vol_hist, edges

# ==========================================
# 2. 繪圖與數據處理
# ==========================================

def smart_download(input_ticker, p, status_container):
    input_ticker = input_ticker.upper()
    targets = [input_ticker] if (".TW" in input_ticker or ".TWO" in input_ticker) else [f"{input_ticker}.TW", f"{input_ticker}.TWO"]
    if not input_ticker.isdigit() and ".TW" not in input_ticker: targets = [input_ticker]

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

def create_chart_precise(df, symbol):
    # 指標計算
    close = df['Close']
    df['MA5'] = close.rolling(5).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    df['STD20'] = close.rolling(20).std()
    df['BB_Up'] = df['MA20'] + 2 * df['STD20']
    df['BB_Lo'] = df['MA20'] - 2 * df['STD20']
    
    last_ma5 = df['MA5'].iloc[-1]
    last_ma20 = df['MA20'].iloc[-1]
    last_ma60 = df['MA60'].iloc[-1]

    # 精確籌碼運算
    hist, edges = calculate_precise_volume_profile(df)
    max_idx = np.argmax(hist)
    poc = (edges[max_idx] + edges[max_idx+1]) / 2

    # --- 視覺風格定義 ---
    mc = mpf.make_marketcolors(
        up='#D32F2F', down='#00796B', 
        edge='inherit', wick='inherit', 
        volume={'up': '#D32F2F', 'down': '#00796B'}
    )
    
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo', 
        marketcolors=mc, 
        gridstyle=':', 
        gridcolor='#E0E0E0', 
        facecolor='#FAFAFA', 
        figcolor='#FFFFFF', 
        y_on_right=True,
        rc={
            'font.family': font_name, 
            'axes.unicode_minus': False,
            'axes.labelsize': 12,
            'axes.titlesize': 18
        }
    )
    
    mav_colors = ['#1f77b4', '#ff7f0e', '#9467bd']
    
    apds = [
        mpf.make_addplot(df['BB_Up'], color='slategrey', linestyle='--', width=0.8, alpha=0.5),
        mpf.make_addplot(df['BB_Lo'], color='slategrey', linestyle='--', width=0.8, alpha=0.5)
    ]

    # --- 繪圖設定 (直式長方形) ---
    fig, axes = mpf.plot(
        df, type='candle', style=s, volume=True, addplot=apds,
        mav=(5, 20, 60), mavcolors=mav_colors,
        figsize=(12, 16),      # 寬12 高16
        panel_ratios=(2.5, 1), 
        returnfig=True, 
        tight_layout=False,    
        scale_padding={'left': 0.1, 'top': 1, 'right': 1.2, 'bottom': 1}
    )
    
    # 強制拉開間距
    fig.subplots_adjust(hspace=0.3)
    
    ax_main = axes[0]
    ax_vol = axes[2]
    
    # 標題設定
    ax_main.set_title(f"{symbol} 專業技術分析", fontproperties=prop, fontsize=22, weight='bold', pad=20)
    ax_main.set_ylabel("價格", fontproperties=prop, fontsize=12)
    ax_vol.set_ylabel("成交量", fontproperties=prop, fontsize=12)

    ax_main.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.2f}'))
    ax_vol.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # VP (Volume Profile)
    ax_vp = ax_main.twiny()
    max_hist = max(hist)
    ax_vp.set_xlim(0, max_hist * 3.0) 
    
    ax_vp.barh(
        y=edges[:-1], width=hist, height=np.diff(edges)*0.9,
        align='edge', color='#708090', alpha=0.15, edgecolor=None, zorder=0
    )
    ax_vp.axis('off')

    # POC
    ax_main.axhline(poc, color='white', linewidth=3.5, alpha=0.8, zorder=9)
    ax_main.axhline(poc, color='#FF6D00', linewidth=2.0, alpha=1.0, zorder=10)
    
    ax_main.text(
        df.index[-1], poc, f' POC: {poc:.2f} ',
        color='white', fontweight='bold', backgroundcolor='#FF6D00',
        fontsize=10, verticalalignment='center', zorder=11,
        bbox=dict(facecolor='#FF6D00', edgecolor='white', boxstyle='round,pad=0.3')
    )

    # Legend
    legend_elements = [
        Line2D([0], [0], color=mav_colors[0], lw=2, label=f'MA5: {last_ma5:.2f}'),
        Line2D([0], [0], color=mav_colors[1], lw=2, label=f'MA20: {last_ma20:.2f}'),
        Line2D([0], [0], color=mav_colors[2], lw=2, label=f'MA60: {last_ma60:.2f}')
    ]
    ax_main.legend(
        handles=legend_elements, loc='upper left', 
        fontsize=11, framealpha=0.9, edgecolor='#CCCCCC'
    )

    return fig, poc, df['Close'].iloc[-1]

# ==========================================
# 3. 側邊欄與執行
# ==========================================
with st.sidebar:
    st.header("參數設定")
    user_input = st.text_input("股票代號", value="2330").strip()
    period = st.selectbox("資料區間", ["3mo", "6mo", "1y"], index=1)
    st.info("💡 系統狀態：API 相容性修正 (V7.1)")
    st.divider()
    run_button = st.button("🚀 開始分析", type="primary")

if run_button:
    status_box = st.empty()
    status_box.text("🚀 初始化...")
    fig = None 
    buf = None 
    
    if not user_input:
        status_box.error("請輸入代號")
    else:
        df, valid_symbol = smart_download(user_input, period, status_box)
        
        if df is None:
            status_box.empty()
            st.error(f"❌ 查無資料: {user_input}")
        else:
            status_box.text(f"🧮 正在運算精確籌碼...")
            
            try:
                fig, poc_price, last_price = create_chart_precise(df, valid_symbol)
                
                status_box.text("✅ 運算完成，渲染中...")
                
                c1, c2, c3 = st.columns([1, 8, 1]) 
                with c2:
                    m1, m2 = st.columns(2)
                    m1.metric("最新收盤", f"{last_price:.2f}")
                    m2.metric("精確 POC 價位", f"{poc_price:.2f}")
                    
                    st.markdown("---")

                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight') 
                    buf.seek(0)
                    
                    # ⚠️ 修正：依指示將 use_container_width=True 替換為 width="stretch"
                    st.image(buf, width="stretch")
                
                status_box.success(f"✨ 分析完成: {valid_symbol}")
                
            except Exception as e:
                status_box.error("運算錯誤")
                st.error(f"Error details: {e}")
            
            finally:
                if fig is not None:
                    plt.close(fig)
                    plt.close('all')
                if buf is not None:
                    buf.close()
