import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="QuantVision — Stock Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ULTRA MODERN CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── RESET & BASE ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050810 !important;
    font-family: 'Outfit', sans-serif !important;
    color: #e2e8f0 !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,102,241,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(16,185,129,0.08) 0%, transparent 60%),
        #050810 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── SIDEBAR ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(10,12,25,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Outfit', sans-serif !important;
    color: #e2e8f0 !important;
}

/* ── HIDE DEFAULT ELEMENTS ─────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
}

/* ── SELECTBOX ────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(15,18,35,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.3s ease !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.2) !important;
}

/* ── TABS ──────────────────────────────────────────────────── */
[data-testid="stTabs"] {
    background: transparent !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,18,35,0.6) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 12px !important;
    color: rgba(226,232,240,0.5) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.25) !important;
    color: #818cf8 !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.2) !important;
}

/* ── METRIC CARDS ─────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(15,18,35,0.7) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(99,102,241,0.4) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.15) !important;
}
[data-testid="metric-container"] label {
    color: rgba(226,232,240,0.5) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

/* ── BUTTONS ─────────────────────────────────────────────── */
.stButton > button {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 12px !important;
    color: rgba(129,140,248,0.7) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.25) !important;
    border-color: rgba(99,102,241,0.55) !important;
    box-shadow: 0 0 22px rgba(99,102,241,0.2) !important;
    transform: translateY(-1px) !important;
    color: #c7d2fe !important;
}
/* Active / primary button — selected stock */
.stButton > button[kind="primary"] {
    background: rgba(99,102,241,0.35) !important;
    border: 1px solid rgba(129,140,248,0.7) !important;
    color: #e0e7ff !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.3) !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: rgba(99,102,241,0.45) !important;
    box-shadow: 0 0 30px rgba(99,102,241,0.4) !important;
}

/* ── CHAT INPUT ───────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: rgba(15,18,35,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"] input {
    background: transparent !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── CHAT MESSAGES ────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: rgba(15,18,35,0.6) !important;
    border: 1px solid rgba(99,102,241,0.1) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
    margin-bottom: 8px !important;
}

/* ── DIVIDER ─────────────────────────────────────────────── */
hr { border-color: rgba(99,102,241,0.15) !important; }

/* ── FORCE SIDEBAR OPEN ───────────────────────────────────── */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ── RADIO BUTTONS (stock selector) ──────────────────────── */
[data-testid="stRadio"] > div {
    gap: 6px !important;
    flex-direction: column !important;
}
[data-testid="stRadio"] label {
    background: rgba(15,18,35,0.6) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: rgba(226,232,240,0.7) !important;
}
[data-testid="stRadio"] label:hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #c7d2fe !important;
    transform: translateX(3px) !important;
}
[data-testid="stRadio"] label[data-checked="true"] {
    background: rgba(99,102,241,0.25) !important;
    border-color: rgba(129,140,248,0.6) !important;
    color: #e0e7ff !important;
    box-shadow: 0 0 16px rgba(99,102,241,0.2) !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
}

/* ── SELECT SLIDER ────────────────────────────────────────── */
.stSlider > div > div {
    background: rgba(99,102,241,0.2) !important;
}
.stSlider [data-testid="stThumbValue"] {
    background: #818cf8 !important;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ── SCROLLBAR ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99,102,241,0.3);
    border-radius: 2px;
}

/* ── PLOTLY CHARTS ────────────────────────────────────────── */
.js-plotly-plot .plotly { border-radius: 16px !important; }
</style>
""", unsafe_allow_html=True)

# ── CUSTOM HTML COMPONENTS ────────────────────────────────────
def header_html(ticker, company):
    logos = {
        "AAPL": "🍎", "TSLA": "⚡", "RELIANCE": "🔷"
    }
    colors = {
        "AAPL": "#818cf8",
        "TSLA": "#34d399",
        "RELIANCE": "#f59e0b"
    }
    col = colors.get(ticker, "#818cf8")
    logo = logos.get(ticker, "📈")
    st.markdown(f"""
    <div style="
        display: flex; align-items: center; gap: 20px;
        padding: 28px 32px;
        background: rgba(15,18,35,0.6);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 20px;
        margin-bottom: 24px;
        backdrop-filter: blur(20px);
        position: relative; overflow: hidden;
        animation: fadeSlideIn 0.6s ease forwards;
    ">
        <div style="
            width: 64px; height: 64px;
            background: rgba(99,102,241,0.15);
            border: 2px solid {col};
            border-radius: 18px;
            display: flex; align-items: center;
            justify-content: center;
            font-size: 28px;
            box-shadow: 0 0 30px {col}40;
        ">{logo}</div>
        <div>
            <div style="
                font-size: 28px; font-weight: 800;
                color: #f1f5f9;
                letter-spacing: -0.02em;
                font-family: 'Outfit', sans-serif;
            ">{company}</div>
            <div style="
                font-size: 13px; font-weight: 500;
                color: {col}; letter-spacing: 0.1em;
                text-transform: uppercase;
                font-family: 'JetBrains Mono', monospace;
            ">{ticker} · LIVE ANALYSIS</div>
        </div>
        <div style="
            margin-left: auto;
            font-size: 11px; color: rgba(226,232,240,0.3);
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
        ">
            QuantVision AI<br>
            <span style="color: #34d399;">● ACTIVE</span>
        </div>
    </div>
    <style>
    @keyframes fadeSlideIn {{
        from {{ opacity: 0; transform: translateY(-12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """, unsafe_allow_html=True)

def section_title(icon, title, subtitle=""):
    st.markdown(f"""
    <div style="
        display: flex; align-items: center; gap: 12px;
        margin: 24px 0 16px 0;
        animation: fadeIn 0.5s ease forwards;
    ">
        <div style="
            width: 36px; height: 36px;
            background: rgba(99,102,241,0.15);
            border: 1px solid rgba(99,102,241,0.3);
            border-radius: 10px;
            display: flex; align-items: center;
            justify-content: center; font-size: 16px;
        ">{icon}</div>
        <div>
            <div style="
                font-size: 16px; font-weight: 700;
                color: #f1f5f9;
                font-family: 'Outfit', sans-serif;
            ">{title}</div>
            {'<div style="font-size:12px;color:rgba(226,232,240,0.4);font-family:Outfit,sans-serif;">' + subtitle + '</div>' if subtitle else ''}
        </div>
    </div>
    <style>
    @keyframes fadeIn {{
        from {{ opacity: 0; }} to {{ opacity: 1; }}
    }}
    </style>
    """, unsafe_allow_html=True)

def signal_badge(label, value, color):
    st.markdown(f"""
    <div style="
        display: inline-flex; align-items: center;
        gap: 8px; padding: 8px 16px;
        background: {color}18;
        border: 1px solid {color}40;
        border-radius: 100px;
        margin: 4px;
        animation: popIn 0.4s cubic-bezier(0.34,1.56,0.64,1) forwards;
    ">
        <div style="
            width: 8px; height: 8px;
            background: {color};
            border-radius: 50%;
            box-shadow: 0 0 8px {color};
        "></div>
        <span style="
            font-size: 12px; font-weight: 600;
            color: {color};
            font-family: 'Outfit', sans-serif;
            letter-spacing: 0.04em;
        ">{label}: {value}</span>
    </div>
    <style>
    @keyframes popIn {{
        from {{ opacity:0; transform: scale(0.8); }}
        to   {{ opacity:1; transform: scale(1); }}
    }}
    </style>
    """, unsafe_allow_html=True)

def prediction_card(direction, confidence, accuracy):
    is_up   = direction == "UP"
    color   = "#34d399" if is_up else "#f87171"
    bg      = "rgba(52,211,153,0.08)" if is_up \
              else "rgba(248,113,113,0.08)"
    arrow   = "↑" if is_up else "↓"
    st.markdown(f"""
    <div style="
        padding: 28px;
        background: {bg};
        border: 1px solid {color}30;
        border-radius: 20px;
        text-align: center;
        position: relative; overflow: hidden;
        animation: glowPulse 3s ease-in-out infinite;
    ">
        <div style="
            font-size: 56px; font-weight: 900;
            color: {color};
            font-family: 'Outfit', sans-serif;
            line-height: 1;
            text-shadow: 0 0 40px {color}60;
            animation: arrowBounce 2s ease-in-out infinite;
        ">{arrow}</div>
        <div style="
            font-size: 22px; font-weight: 800;
            color: {color}; margin: 8px 0 4px;
            font-family: 'Outfit', sans-serif;
        ">{direction}</div>
        <div style="
            font-size: 13px;
            color: rgba(226,232,240,0.5);
            font-family: 'JetBrains Mono', monospace;
        ">{confidence:.1f}% confidence · {accuracy:.1%} model acc</div>
    </div>
    <style>
    @keyframes glowPulse {{
        0%,100% {{ box-shadow: 0 0 20px {color}15; }}
        50%      {{ box-shadow: 0 0 40px {color}30; }}
    }}
    @keyframes arrowBounce {{
        0%,100% {{ transform: translateY(0); }}
        50%      {{ transform: translateY(-6px); }}
    }}
    </style>
    """, unsafe_allow_html=True)

def risk_card(level, reason):
    cfg = {
        "HIGH RISK":   ("#f87171", "🔴", "rgba(248,113,113,0.08)"),
        "MEDIUM RISK": ("#fbbf24", "🟡", "rgba(251,191,36,0.08)"),
        "LOW RISK":    ("#34d399", "🟢", "rgba(52,211,153,0.08)"),
    }
    color, icon, bg = cfg.get(
        level, ("#818cf8","⚪","rgba(99,102,241,0.08)")
    )
    st.markdown(f"""
    <div style="
        padding: 20px 24px;
        background: {bg};
        border: 1px solid {color}30;
        border-radius: 16px;
        animation: slideUp 0.5s ease forwards;
    ">
        <div style="
            font-size: 18px; font-weight: 700;
            color: {color}; margin-bottom: 6px;
            font-family: 'Outfit', sans-serif;
        ">{icon} {level}</div>
        <div style="
            font-size: 13px;
            color: rgba(226,232,240,0.55);
            font-family: 'Outfit', sans-serif;
            line-height: 1.5;
        ">{reason}</div>
    </div>
    <style>
    @keyframes slideUp {{
        from {{ opacity:0; transform:translateY(10px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    </style>
    """, unsafe_allow_html=True)

# ── DATA HELPERS ──────────────────────────────────────────────
FILE_MAP = {
    "AAPL": "AAPL", "TSLA": "TSLA", "RELIANCE": "RELIANCE"
}
COMPANY = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla Inc.",
    "RELIANCE": "Reliance Industries"
}

def add_extra_features(df):
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]
    df["SMA_10"]         = c.rolling(10).mean()
    df["SMA_20"]         = c.rolling(20).mean()
    df["EMA_10"]         = c.ewm(span=10).mean()
    df["price_to_sma10"] = c / df["SMA_10"] - 1
    df["price_to_sma20"] = c / df["SMA_20"] - 1
    df["sma10_to_sma20"] = df["SMA_10"] / df["SMA_20"] - 1
    df["SMA_cross"]      = (df["SMA_10"]>df["SMA_20"]).astype(int)
    df["ATR_pct"]        = (h-l).rolling(14).mean() / c
    df["volume_ratio"]   = v / v.rolling(20).mean()
    df["volume_change"]  = v.pct_change()
    df["return_1d"]      = c.pct_change(1)
    df["return_3d"]      = c.pct_change(3)
    df["return_5d"]      = c.pct_change(5)
    df["return_10d"]     = c.pct_change(10)
    df["lag_ret_1"]      = df["return_1d"].shift(1)
    df["lag_ret_2"]      = df["return_1d"].shift(2)
    df["lag_ret_3"]      = df["return_1d"].shift(3)
    rh = h.rolling(20).max()
    rl = l.rolling(20).min()
    df["dist_high"]      = (c - rh) / c
    df["dist_low"]       = (c - rl) / c
    op = df["Open"] if "Open" in df.columns else c
    df["body_size"]      = abs(c - op) / c
    df["upper_wick"]     = (h - c.clip(lower=op)) / c
    df["lower_wick"]     = (c.clip(upper=op) - l) / c
    df["ROC_5"]          = c.pct_change(5)
    df["ROC_10"]         = c.pct_change(10)
    daily = np.sign(df["return_1d"].fillna(0))
    streak, count = [], 0
    for d in daily:
        count = (max(1,count+1) if d>0 else
                 min(-1,count-1) if d<0 else 0)
        streak.append(count)
    df["streak"] = streak
    df.drop(columns=["SMA_10","SMA_20","EMA_10"],
            inplace=True, errors="ignore")
    return df

@st.cache_data
def load_all(ticker):
    fname = FILE_MAP[ticker]
    paths = {
        "base": f"data/{fname}.csv",
        "ind":  f"data/{fname}_indicators.csv",
        "pat":  f"data/{fname}_patterns.csv",
        "tre":  f"data/{fname}_trend.csv"
    }
    out = {}
    for k, p in paths.items():
        if os.path.exists(p):
            d = pd.read_csv(p)
            d.columns = d.columns.str.strip()
            if k == "base":
                d.columns = d.columns.str.title()
            dc = [c for c in d.columns if c.lower()=="date"]
            if dc:
                d.rename(columns={dc[0]:"Date"},inplace=True)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = ['_'.join(
                    [str(x) for x in c if str(x)!='']
                ).strip() for c in d.columns]
            d["Date"] = pd.to_datetime(d["Date"])
            out[k] = d
    return out

def get_prediction(ticker):
    path = f"models/{ticker}_model.pkl"
    if not os.path.exists(path):
        return None, None, None
    try:
        saved    = joblib.load(path)
        model    = saved["model"]
        scaler   = saved.get("scaler")
        features = saved["features"]
        acc      = saved.get("accuracy", 0)
        fname    = FILE_MAP[ticker]
        dfs, suffixes = [], ["","_indicators","_patterns","_trend"]
        for s in suffixes:
            p = f"data/{fname}{s}.csv"
            if os.path.exists(p):
                d = pd.read_csv(p)
                d.columns = d.columns.str.strip()
                if s == "":
                    d.columns = d.columns.str.title()
                dc = [c for c in d.columns if c.lower()=="date"]
                if dc:
                    d.rename(columns={dc[0]:"Date"},inplace=True)
                d["Date"] = pd.to_datetime(
                    d["Date"]
                ).dt.strftime("%Y-%m-%d")
                dfs.append(d)
        if not dfs: return None, None, None
        df = dfs[0]
        for d in dfs[1:]:
            df = df.merge(d, on="Date", how="left")
        cc = [c for c in df.columns if c.lower()=="close"]
        if len(cc) > 1:
            df.drop(columns=cc[1:],inplace=True,errors="ignore")
        df = df.sort_values("Date").reset_index(drop=True)
        df = add_extra_features(df)
        avail  = [f for f in features if f in df.columns]
        latest = df[avail].iloc[[-1]].fillna(0)
        if scaler:
            latest = pd.DataFrame(
                scaler.transform(latest), columns=avail
            )
        pred  = model.predict(latest)[0]
        proba = model.predict_proba(latest)[0]
        return ("UP" if pred==1 else "DOWN"), max(proba)*100, acc
    except Exception:
        return None, None, None

def get_risk(ticker):
    fname = FILE_MAP[ticker]
    ind   = f"data/{fname}_indicators.csv"
    tre   = f"data/{fname}_trend.csv"
    risks = []
    if os.path.exists(ind):
        d   = pd.read_csv(ind)
        d.columns = d.columns.str.strip()
        row = d.iloc[-1]
        rsi = row.get("RSI", 50)
        if pd.notna(rsi):
            r = float(rsi)
            if r > 70: risks.append(f"RSI overbought ({r:.0f})")
            elif r < 30: risks.append(f"RSI oversold ({r:.0f})")
        ms = str(row.get("MACD_signal",""))
        if "Bearish" in ms: risks.append("MACD bearish")
    if os.path.exists(tre):
        d   = pd.read_csv(tre)
        row = d.iloc[-1]
        if int(row.get("trend_direction",0)) == -1:
            risks.append("Downtrend active")
    level = ("HIGH RISK" if len(risks)>=2 else
             "MEDIUM RISK" if len(risks)==1 else "LOW RISK")
    return level, (", ".join(risks) if risks
                   else "All indicators normal")

# ── PLOTLY THEME ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,18,35,0.6)",
    font=dict(family="Outfit", color="#94a3b8", size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(
        gridcolor="rgba(99,102,241,0.08)",
        showgrid=True, zeroline=False,
        tickfont=dict(size=11, color="#64748b")
    ),
    yaxis=dict(
        gridcolor="rgba(99,102,241,0.08)",
        showgrid=True, zeroline=False,
        tickfont=dict(size=11, color="#64748b")
    ),
    legend=dict(
        bgcolor="rgba(15,18,35,0.8)",
        bordercolor="rgba(99,102,241,0.2)",
        borderwidth=1,
        font=dict(size=12)
    )
)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="
        padding: 20px 0 16px;
        font-size: 20px; font-weight: 800;
        color: #f1f5f9; letter-spacing: -0.02em;
        font-family: 'Outfit', sans-serif;
    ">
        QuantVision
        <span style="
            font-size: 11px; font-weight: 500;
            color: #818cf8; letter-spacing: 0.1em;
            display: block; margin-top: 2px;
        ">AI STOCK ANALYSIS</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:rgba(226,232,240,0.4);
        font-family:'Outfit',sans-serif;letter-spacing:0.08em;
        text-transform:uppercase;margin-bottom:8px;">
        Select Company
    </div>""", unsafe_allow_html=True)

    ticker = st.radio(
        "",
        ["AAPL", "TSLA", "RELIANCE"],
        format_func=lambda x: {
            "AAPL":     "🍎  Apple Inc.",
            "TSLA":     "⚡  Tesla Inc.",
            "RELIANCE": "🔷  Reliance Industries"
        }[x],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:rgba(226,232,240,0.4);
        font-family:'Outfit',sans-serif;letter-spacing:0.08em;
        text-transform:uppercase;margin-bottom:8px;">
        Timeframe
    </div>""", unsafe_allow_html=True)

    tf = st.selectbox(
        "", ["1M","3M","6M","1Y","3Y"],
        index=3, label_visibility="collapsed"
    )
    tf_days = {"1M":30,"3M":90,"6M":180,"1Y":365,"3Y":1095}
    days    = tf_days[tf]

    st.markdown("---")

    # Prediction in sidebar
    dirn, conf, acc = get_prediction(ticker)
    level, reason   = get_risk(ticker)

    if dirn:
        col  = "#34d399" if dirn=="UP" else "#f87171"
        icon = "↑" if dirn=="UP" else "↓"
        st.markdown(f"""
        <div style="
            padding: 16px;
            background: {col}10;
            border: 1px solid {col}25;
            border-radius: 14px; text-align: center;
            margin-bottom: 12px;
        ">
            <div style="font-size:32px;color:{col};
                font-weight:900;font-family:'Outfit',sans-serif;
                animation: bounce 2s infinite;">
                {icon} {dirn}
            </div>
            <div style="font-size:11px;
                color:rgba(226,232,240,0.45);
                font-family:'JetBrains Mono',monospace;
                margin-top:4px;">
                {conf:.1f}% confidence
            </div>
        </div>
        <style>
        @keyframes bounce {{
            0%,100% {{ transform:translateY(0); }}
            50%      {{ transform:translateY(-4px); }}
        }}
        </style>
        """, unsafe_allow_html=True)

    risk_colors = {
        "HIGH RISK":   "#f87171",
        "MEDIUM RISK": "#fbbf24",
        "LOW RISK":    "#34d399"
    }
    rc = risk_colors.get(level, "#818cf8")
    st.markdown(f"""
    <div style="
        padding: 12px 16px;
        background: {rc}10;
        border: 1px solid {rc}25;
        border-radius: 12px;
    ">
        <div style="font-size:13px;font-weight:600;
            color:{rc};font-family:'Outfit',sans-serif;">
            {level}
        </div>
        <div style="font-size:11px;
            color:rgba(226,232,240,0.4);
            font-family:'Outfit',sans-serif;
            margin-top:3px;line-height:1.4;">
            {reason}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:rgba(226,232,240,0.25);
        font-family:'Outfit',sans-serif;line-height:1.8;">
        ML Stock Analysis Project<br>
        Group-13
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT ──────────────────────────────────────────────
data = load_all(ticker)
base = data.get("base")
ind  = data.get("ind")
pat  = data.get("pat")
tre  = data.get("tre")

if base is not None:
    cutoff = base["Date"].max() - pd.Timedelta(days=days)
    base_f = base[base["Date"] >= cutoff].copy()
else:
    base_f = None

if ind is not None:
    cutoff = ind["Date"].max() - pd.Timedelta(days=days)
    ind_f  = ind[ind["Date"] >= cutoff].copy()
else:
    ind_f = None

st.markdown("<br>", unsafe_allow_html=True)

# Header
header_html(ticker, COMPANY[ticker])

# ── TOP KPI ROW ───────────────────────────────────────────────
if base_f is not None and len(base_f) > 1:
    curr  = base_f["Close"].iloc[-1]
    prev  = base_f["Close"].iloc[-2]
    chg   = curr - prev
    pct   = chg / prev * 100
    high  = base_f["High"].max()
    low   = base_f["Low"].min()
    vol   = base_f["Volume"].mean()
    ret5  = (curr - base_f["Close"].iloc[-6]) / base_f["Close"].iloc[-6] * 100 \
            if len(base_f) > 5 else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Current Price",    f"{curr:.2f}",
              f"{chg:+.2f} ({pct:+.1f}%)")
    c2.metric(f"{tf} High",       f"{high:.2f}")
    c3.metric(f"{tf} Low",        f"{low:.2f}")
    c4.metric("Avg Daily Volume", f"{vol/1e6:.2f}M")
    c5.metric("5-Day Return",     f"{ret5:+.2f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Price",
    "⚡  Indicators",
    "🔍  Patterns",
    "📈  Trend",
    "🤖  AI Chat"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PRICE CHART
# ══════════════════════════════════════════════════════════════
with tab1:
    if base_f is not None:
        section_title("📊", "Price Action",
                      f"{COMPANY[ticker]} · {tf} candlestick")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=base_f["Date"],
            open=base_f["Open"], high=base_f["High"],
            low=base_f["Low"],   close=base_f["Close"],
            name="Price",
            increasing=dict(
                line=dict(color="#34d399", width=1),
                fillcolor="rgba(52,211,153,0.7)"
            ),
            decreasing=dict(
                line=dict(color="#f87171", width=1),
                fillcolor="rgba(248,113,113,0.7)"
            )
        ))

        # Bollinger Bands overlay
        if ind_f is not None:
            for col, clr, nm in [
                ("BB_upper","rgba(99,102,241,0.6)","BB Upper"),
                ("BB_lower","rgba(99,102,241,0.6)","BB Lower"),
                ("BB_middle","rgba(99,102,241,0.3)","BB Mid")
            ]:
                if col in ind_f.columns:
                    fig.add_trace(go.Scatter(
                        x=ind_f["Date"], y=ind_f[col],
                        name=nm,
                        line=dict(color=clr,width=1,dash="dot"),
                        showlegend=True
                    ))

        layout = PLOTLY_LAYOUT.copy()
        layout.update(dict(
            height=480,
            xaxis_rangeslider_visible=False,
            title=dict(
                text=f"{ticker} · {tf} Price Action",
                font=dict(size=14,color="#94a3b8")
            )
        ))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        # Volume
        section_title("📦", "Volume", "Trading activity")
        colors_vol = [
            "#34d399" if c >= o else "#f87171"
            for c, o in zip(
                base_f["Close"], base_f["Open"]
            )
        ]
        fig_v = go.Figure(go.Bar(
            x=base_f["Date"], y=base_f["Volume"],
            marker_color=colors_vol, opacity=0.7,
            name="Volume"
        ))
        lv = PLOTLY_LAYOUT.copy()
        lv.update(height=200)
        fig_v.update_layout(**lv)
        st.plotly_chart(fig_v, use_container_width=True)

    else:
        st.error("Price data not found — run data_collection.py")

# ══════════════════════════════════════════════════════════════
# TAB 2 — INDICATORS
# ══════════════════════════════════════════════════════════════
with tab2:
    if ind_f is not None:
        section_title("⚡", "Technical Indicators",
                      "RSI · MACD · Bollinger Bands")

        # Signal badges
        latest = ind_f.iloc[-1]
        rsi_v  = latest.get("RSI", 50)
        sig_v  = latest.get("indicator_signal", 0)
        macd_v = latest.get("MACD_signal", "Neutral")
        roc_v  = latest.get("ROC", 0)

        cols_badges = st.columns(4)
        with cols_badges[0]:
            rc = ("#f87171" if float(rsi_v)>70 else
                  "#34d399" if float(rsi_v)<30 else "#818cf8")
            signal_badge(
                "RSI", f"{float(rsi_v):.1f}", rc
            )
        with cols_badges[1]:
            mc = "#34d399" if "Bull" in str(macd_v) \
                 else "#f87171" if "Bear" in str(macd_v) \
                 else "#818cf8"
            signal_badge("MACD", str(macd_v), mc)
        with cols_badges[2]:
            roc_c = "#34d399" if float(roc_v)>0 else "#f87171"
            signal_badge("ROC", f"{float(roc_v):.2f}", roc_c)
        with cols_badges[3]:
            ov_c = ("#34d399" if sig_v==1 else
                    "#f87171" if sig_v==-1 else "#818cf8")
            ov_l = ("Bullish" if sig_v==1 else
                    "Bearish" if sig_v==-1 else "Neutral")
            signal_badge("Overall", ov_l, ov_c)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # RSI
        with col1:
            if "RSI" in ind_f.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_hrect(
                    y0=70, y1=100,
                    fillcolor="rgba(248,113,113,0.05)",
                    line_width=0
                )
                fig_rsi.add_hrect(
                    y0=0, y1=30,
                    fillcolor="rgba(52,211,153,0.05)",
                    line_width=0
                )
                fig_rsi.add_trace(go.Scatter(
                    x=ind_f["Date"], y=ind_f["RSI"],
                    fill="tozeroy",
                    fillcolor="rgba(129,140,248,0.08)",
                    line=dict(color="#818cf8", width=2),
                    name="RSI"
                ))
                fig_rsi.add_hline(
                    y=70, line_color="#f87171",
                    line_dash="dot", line_width=1
                )
                fig_rsi.add_hline(
                    y=30, line_color="#34d399",
                    line_dash="dot", line_width=1
                )
                lr = PLOTLY_LAYOUT.copy()
                lr.update(dict(
                    height=300,
                    title=dict(
                        text="RSI (14)",
                        font=dict(size=13,color="#94a3b8")
                    ),
                    yaxis=dict(range=[0,100])
                ))
                fig_rsi.update_layout(**lr)
                st.plotly_chart(
                    fig_rsi, use_container_width=True
                )

        # MACD
        with col2:
            if "MACD" in ind_f.columns:
                fig_m = go.Figure()
                if "MACD_hist" in ind_f.columns:
                    hist_colors = [
                        "#34d399" if v >= 0 else "#f87171"
                        for v in ind_f["MACD_hist"]
                    ]
                    fig_m.add_trace(go.Bar(
                        x=ind_f["Date"],
                        y=ind_f["MACD_hist"],
                        marker_color=hist_colors,
                        opacity=0.5, name="Histogram"
                    ))
                fig_m.add_trace(go.Scatter(
                    x=ind_f["Date"], y=ind_f["MACD"],
                    line=dict(color="#818cf8",width=2),
                    name="MACD"
                ))
                if "MACD_signal_line" in ind_f.columns:
                    fig_m.add_trace(go.Scatter(
                        x=ind_f["Date"],
                        y=ind_f["MACD_signal_line"],
                        line=dict(color="#f59e0b",width=1.5),
                        name="Signal"
                    ))
                lm = PLOTLY_LAYOUT.copy()
                lm.update(dict(
                    height=300,
                    title=dict(
                        text="MACD (12,26,9)",
                        font=dict(size=13,color="#94a3b8")
                    )
                ))
                fig_m.update_layout(**lm)
                st.plotly_chart(
                    fig_m, use_container_width=True
                )

        # ROC full width
        if "ROC" in ind_f.columns:
            section_title("📐","Rate of Change","Momentum speed")
            fig_roc = go.Figure()
            roc_col = [
                "#34d399" if v>=0 else "#f87171"
                for v in ind_f["ROC"]
            ]
            fig_roc.add_trace(go.Bar(
                x=ind_f["Date"], y=ind_f["ROC"],
                marker_color=roc_col, opacity=0.7,
                name="ROC"
            ))
            fig_roc.add_hline(
                y=0, line_color="rgba(226,232,240,0.2)",
                line_width=1
            )
            lr2 = PLOTLY_LAYOUT.copy()
            lr2.update(height=220)
            fig_roc.update_layout(**lr2)
            st.plotly_chart(fig_roc, use_container_width=True)

    else:
        st.error(
            "Indicator data not found — "
            "run indicator_analysis.py"
        )

# ══════════════════════════════════════════════════════════════
# TAB 3 — PATTERNS
# ══════════════════════════════════════════════════════════════
with tab3:
    if pat is not None:
        section_title("🔍","Pattern Detection",
                      "Chart formations detected by ML")

        PNAMES = {
            0: "No Pattern",
            1: "Double Bottom",
            2: "Head & Shoulders",
            3: "Descending Triangle"
        }
        PSIGS = {
            0: ("Neutral", "#818cf8"),
            1: ("Bullish reversal", "#34d399"),
            2: ("Bearish reversal", "#f87171"),
            3: ("Bearish breakdown", "#f87171")
        }

        latest_pat = pat.iloc[-1]
        plabel     = int(latest_pat.get("pattern_label",0))
        pname      = PNAMES[plabel]
        psig, pcol = PSIGS[plabel]

        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown(f"""
            <div style="
                padding: 28px; text-align: center;
                background: {pcol}10;
                border: 1px solid {pcol}30;
                border-radius: 20px;
                animation: popIn 0.5s
                    cubic-bezier(0.34,1.56,0.64,1) forwards;
            ">
                <div style="font-size:40px;margin-bottom:12px;">
                    {'📈' if plabel==1 else '📉' if plabel>1 else '⚪'}
                </div>
                <div style="font-size:18px;font-weight:700;
                    color:{pcol};
                    font-family:'Outfit',sans-serif;">
                    {pname}
                </div>
                <div style="font-size:12px;
                    color:rgba(226,232,240,0.5);
                    font-family:'Outfit',sans-serif;
                    margin-top:6px;">
                    {psig}
                </div>
            </div>
            <style>
            @keyframes popIn {{
                from{{opacity:0;transform:scale(0.85);}}
                to{{opacity:1;transform:scale(1);}}
            }}
            </style>
            """, unsafe_allow_html=True)

        with col2:
            # Pattern distribution pie
            counts   = pat["pattern_label"].value_counts()
            plabels  = [PNAMES.get(int(i),str(i))
                        for i in counts.index]
            pcolors  = ["#818cf8","#34d399","#f87171","#fbbf24"]
            fig_pie  = go.Figure(go.Pie(
                labels=plabels, values=counts.values,
                hole=0.6,
                marker=dict(
                    colors=pcolors[:len(counts)],
                    line=dict(
                        color="rgba(5,8,16,0.8)", width=2
                    )
                ),
                textfont=dict(family="Outfit"),
                hovertemplate="%{label}<br>%{value} rows<extra></extra>"
            ))
            lp = PLOTLY_LAYOUT.copy()
            lp.update(dict(
                height=260,
                showlegend=True,
                title=dict(
                    text="Pattern Distribution",
                    font=dict(size=13,color="#94a3b8")
                )
            ))
            fig_pie.update_layout(**lp)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Pattern over time
        if base_f is not None:
            pat_copy = pat.copy()
            pat_copy["Date"] = pd.to_datetime(pat_copy["Date"])
            cutoff   = pat_copy["Date"].max() - pd.Timedelta(days=days)
            pat_f    = pat_copy[pat_copy["Date"]>=cutoff]
            merged   = base_f.merge(
                pat_f[["Date","pattern_label"]],
                on="Date", how="left"
            ).fillna(0)

            section_title("📅","Patterns Over Time",
                          "Highlighted on price chart")
            fig_pt = go.Figure()
            fig_pt.add_trace(go.Scatter(
                x=merged["Date"], y=merged["Close"],
                line=dict(color="#818cf8",width=1.5),
                name="Price"
            ))
            for pid, pclr, pn in [
                (1,"#34d399","Double Bottom"),
                (2,"#f87171","Head & Shoulders"),
                (3,"#fbbf24","Desc. Triangle")
            ]:
                mask = merged["pattern_label"] == pid
                if mask.any():
                    fig_pt.add_trace(go.Scatter(
                        x=merged["Date"][mask],
                        y=merged["Close"][mask],
                        mode="markers",
                        marker=dict(
                            color=pclr, size=10,
                            symbol="diamond",
                            line=dict(color="white",width=1)
                        ),
                        name=pn
                    ))
            lpt = PLOTLY_LAYOUT.copy()
            lpt.update(height=300)
            fig_pt.update_layout(**lpt)
            st.plotly_chart(fig_pt, use_container_width=True)

    else:
        st.error(
            "Pattern data not found — "
            "run pattern_analysis.py"
        )

# ══════════════════════════════════════════════════════════════
# TAB 4 — TREND
# ══════════════════════════════════════════════════════════════
with tab4:
    if tre is not None and base_f is not None:
        section_title("📈","Trend Analysis",
                      "Support · Resistance · Direction")

        tre_copy = tre.copy()
        tre_copy["Date"] = pd.to_datetime(tre_copy["Date"])
        cutoff   = tre_copy["Date"].max() - pd.Timedelta(days=days)
        tre_f    = tre_copy[tre_copy["Date"] >= cutoff]

        latest_t = tre.iloc[-1]
        td       = latest_t.get("trend_direction", 0)
        sp       = float(latest_t.get("support_price", 0))
        rp       = float(latest_t.get("resistance_price", 0))
        tname    = ("Uptrend" if td==1 else
                    "Downtrend" if td==-1 else "Sideways")
        tcol     = ("#34d399" if td==1 else
                    "#f87171" if td==-1 else "#fbbf24")

        c1, c2, c3 = st.columns(3)
        c1.metric("Trend Direction", tname)
        c2.metric("Support Level",   f"{sp:.2f}")
        c3.metric("Resistance Level",f"{rp:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatter(
            x=base_f["Date"], y=base_f["Close"],
            name="Price",
            line=dict(color="#818cf8", width=2),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.04)"
        ))
        if "support_price" in tre_f.columns:
            fig_tr.add_trace(go.Scatter(
                x=tre_f["Date"], y=tre_f["support_price"],
                name="Support",
                line=dict(color="#34d399",dash="dash",width=1.5)
            ))
        if "resistance_price" in tre_f.columns:
            fig_tr.add_trace(go.Scatter(
                x=tre_f["Date"],
                y=tre_f["resistance_price"],
                name="Resistance",
                line=dict(color="#f87171",dash="dash",width=1.5)
            ))

        # Shade trend direction
        if "trend_direction" in tre_f.columns:
            up_mask   = tre_f["trend_direction"] == 1
            down_mask = tre_f["trend_direction"] == -1
            for dt, mask, col, nm in [
                (tre_f, up_mask,
                 "rgba(52,211,153,0.05)", "Uptrend zone"),
                (tre_f, down_mask,
                 "rgba(248,113,113,0.05)", "Downtrend zone")
            ]:
                if mask.any() and "support_price" in dt.columns:
                    fig_tr.add_trace(go.Scatter(
                        x=dt["Date"][mask],
                        y=dt["resistance_price"][mask],
                        fill="tonexty",
                        fillcolor=col,
                        line=dict(width=0),
                        showlegend=False,
                        name=nm
                    ))

        lt = PLOTLY_LAYOUT.copy()
        lt.update(height=460)
        fig_tr.update_layout(**lt)
        st.plotly_chart(fig_tr, use_container_width=True)

        # Trend distribution bar
        if "trend_direction" in tre.columns:
            section_title("📊","Trend History Distribution","")
            tc = tre["trend_direction"].value_counts()
            tl = [("Uptrend" if i==1 else
                   "Downtrend" if i==-1 else "Sideways")
                  for i in tc.index]
            tv = [("#34d399" if i==1 else
                   "#f87171" if i==-1 else "#fbbf24")
                  for i in tc.index]
            fig_tb = go.Figure(go.Bar(
                x=tl, y=tc.values,
                marker_color=tv, opacity=0.8
            ))
            ltb = PLOTLY_LAYOUT.copy()
            ltb.update(height=220)
            fig_tb.update_layout(**ltb)
            st.plotly_chart(fig_tb, use_container_width=True)

    else:
        st.error(
            "Trend data not found — "
            "run trend_analysis.py"
        )

# ══════════════════════════════════════════════════════════════
# TAB 5 — AI CHATBOT
# ══════════════════════════════════════════════════════════════
with tab5:
    section_title("🤖","AI Stock Assistant",
                  f"Ask anything about {ticker}")

    # Import chatbot
    import sys
    sys.path.append(
        os.path.join(os.path.dirname(__file__), "..")
    )

    try:
        from chatbot.chatbot import get_response
        chatbot_ready = True
    except ImportError:
        chatbot_ready = False

    if chatbot_ready:
        # Quick buttons row
        st.markdown("""
        <div style="font-size:12px;
            color:rgba(226,232,240,0.4);
            font-family:'Outfit',sans-serif;
            letter-spacing:0.06em;text-transform:uppercase;
            margin-bottom:12px;">
            Quick Questions
        </div>
        """, unsafe_allow_html=True)

        qb1, qb2, qb3, qb4 = st.columns(4)
        if "chat" not in st.session_state:
            st.session_state.chat = []

        quick_qs = [
            (qb1, f"Will {ticker} go up tomorrow?",
             "Tomorrow's Prediction"),
            (qb2, f"Is {ticker} high risk?",
             "Risk Assessment"),
            (qb3, f"What indicators show for {ticker}?",
             "Indicators"),
            (qb4, f"What trend is {ticker} showing?",
             "Trend Info"),
        ]
        for col, q, label in quick_qs:
            with col:
                if st.button(label):
                    r = get_response(q, ticker)
                    st.session_state.chat += [
                        {"role":"user",    "content":q},
                        {"role":"assistant","content":r}
                    ]
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input
        if prompt := st.chat_input(
            f"Ask me anything about {ticker}..."
        ):
            r = get_response(prompt, ticker)
            st.session_state.chat += [
                {"role":"user",     "content":prompt},
                {"role":"assistant","content":r}
            ]
            st.rerun()

        if not st.session_state.chat:
            st.markdown(f"""
            <div style="
                text-align: center; padding: 40px 20px;
                color: rgba(226,232,240,0.3);
                font-family: 'Outfit', sans-serif;
            ">
                <div style="font-size: 40px;
                    margin-bottom: 12px;">🤖</div>
                <div style="font-size: 16px;
                    font-weight: 600; margin-bottom: 6px;
                    color: rgba(226,232,240,0.5);">
                    Ask me about {ticker}
                </div>
                <div style="font-size: 13px;">
                    Try: "Will {ticker} go up tomorrow?"
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error(
            "Chatbot not found! "
            "Make sure chatbot/chatbot.py exists."
        )
        st.info(
            "Get chatbot.py from Member 5 and place it "
            "in the chatbot/ folder."
        )