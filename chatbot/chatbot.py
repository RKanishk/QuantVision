import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── FILE MAP (must match model_training.py exactly) ───────────
FILE_MAP = {
    "AAPL":     "AAPL",
    "TSLA":     "TSLA",
    "RELIANCE": "RELIANCE"
}

# ── EXACT SAME feature engineering as model_training.py ───────
# KEY FIX: chatbot must build features EXACTLY the same way
# the model was trained — otherwise feature count mismatch
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
    df["SMA_cross"]      = (
        df["SMA_10"] > df["SMA_20"]
    ).astype(int)

    tr               = h - l
    df["ATR_pct"]    = tr.rolling(14).mean() / c
    df["volume_ratio"]  = v / v.rolling(20).mean()
    df["volume_change"] = v.pct_change()
    df["return_1d"]  = c.pct_change(1)
    df["return_3d"]  = c.pct_change(3)
    df["return_5d"]  = c.pct_change(5)
    df["return_10d"] = c.pct_change(10)
    df["lag_ret_1"]  = df["return_1d"].shift(1)
    df["lag_ret_2"]  = df["return_1d"].shift(2)
    df["lag_ret_3"]  = df["return_1d"].shift(3)

    roll_high        = h.rolling(20).max()
    roll_low         = l.rolling(20).min()
    df["dist_high"]  = (c - roll_high) / c
    df["dist_low"]   = (c - roll_low)  / c

    open_col         = df["Open"] if "Open" in df.columns \
                       else c
    df["body_size"]  = abs(c - open_col) / c
    df["upper_wick"] = (h - c.clip(lower=open_col)) / c
    df["lower_wick"] = (c.clip(upper=open_col) - l) / c
    df["ROC_5"]      = c.pct_change(5)
    df["ROC_10"]     = c.pct_change(10)

    # Consecutive streak
    daily_dir        = np.sign(df["return_1d"].fillna(0))
    streak           = []
    count            = 0
    for d in daily_dir:
        if d > 0:   count = max(1,  count + 1)
        elif d < 0: count = min(-1, count - 1)
        else:       count = 0
        streak.append(count)
    df["streak"] = streak

    # Drop intermediate columns
    df.drop(
        columns=["SMA_10", "SMA_20", "EMA_10"],
        inplace=True, errors="ignore"
    )
    return df

# ── LOAD AND MERGE data (identical to model_training.py) ──────
def load_merged(ticker):
    fname     = FILE_MAP.get(ticker, ticker)
    base_path = f"data/{fname}.csv"
    ind_path  = f"data/{fname}_indicators.csv"
    pat_path  = f"data/{fname}_patterns.csv"
    tre_path  = f"data/{fname}_trend.csv"

    for path in [base_path, ind_path, pat_path, tre_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

    base = pd.read_csv(base_path)
    ind  = pd.read_csv(ind_path)
    pat  = pd.read_csv(pat_path)
    tre  = pd.read_csv(tre_path)

    # Flatten MultiIndex columns
    for d in [base, ind, pat, tre]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [
                '_'.join([str(x) for x in c
                          if str(x) != '']).strip()
                for c in d.columns
            ]

    # Standardize column names
    base.columns = base.columns.str.strip().str.title()
    for d in [ind, pat, tre]:
        d.columns = d.columns.str.strip()
        dc = [c for c in d.columns if c.lower() == "date"]
        if dc:
            d.rename(columns={dc[0]: "Date"}, inplace=True)

    # Uniform date format
    for d in [base, ind, pat, tre]:
        d["Date"] = pd.to_datetime(
            d["Date"]
        ).dt.strftime("%Y-%m-%d")

    # Merge all 4 files
    df = base.merge(ind, on="Date", how="left")
    df = df.merge(pat, on="Date", how="left")
    df = df.merge(tre, on="Date", how="left")

    # Remove duplicate Close columns
    cc = [c for c in df.columns if c.lower() == "close"]
    if len(cc) > 1:
        df.drop(columns=cc[1:], inplace=True,
                errors="ignore")

    if "Close" not in df.columns:
        raise KeyError(f"No Close column for {ticker}")

    df = df.sort_values("Date").reset_index(drop=True)

    # Add ALL engineered features — same as training
    df = add_extra_features(df)

    # Fill NaN
    df = df.fillna(0)
    df.reset_index(drop=True, inplace=True)
    return df

# ── FEATURE LIST (must match model_training.py exactly) ───────
FEATURES = [
    "RSI", "MACD", "ROC", "indicator_signal",
    "pattern_label", "trend_direction", "trend_slope",
    "price_to_sma10", "price_to_sma20", "sma10_to_sma20",
    "SMA_cross", "ATR_pct", "volume_ratio", "volume_change",
    "return_1d", "return_3d", "return_5d", "return_10d",
    "lag_ret_1", "lag_ret_2", "lag_ret_3",
    "dist_high", "dist_low",
    "body_size", "upper_wick", "lower_wick",
    "ROC_5", "ROC_10", "streak"
]

# ── GET PREDICTION using saved model ──────────────────────────
def get_prediction(ticker):
    model_path = f"models/{ticker}_model.pkl"
    if not os.path.exists(model_path):
        return None, None, None

    try:
        saved    = joblib.load(model_path)
        model    = saved["model"]
        scaler   = saved.get("scaler")
        features = saved["features"]   # exact list from training
        acc      = saved.get("accuracy", 0)

        # Load data with SAME pipeline as training
        df     = load_merged(ticker)

        # Use only features the model was trained on
        avail  = [f for f in features if f in df.columns]

        # KEY FIX: use the saved features list exactly
        # This guarantees same number of columns as training
        latest = df[avail].iloc[[-1]].fillna(0)

        # Scale using the SAME scaler from training
        if scaler is not None:
            latest_scaled = scaler.transform(latest)
            latest = pd.DataFrame(
                latest_scaled, columns=avail
            )

        pred   = model.predict(latest)[0]
        proba  = model.predict_proba(latest)[0]
        conf   = max(proba) * 100
        dirn   = "UP" if pred == 1 else "DOWN"
        return dirn, conf, acc

    except Exception as e:
        return None, None, str(e)

# ── GET RISK LEVEL ────────────────────────────────────────────
def get_risk(ticker):
    try:
        fname = FILE_MAP.get(ticker, ticker)
        path  = f"data/{fname}_indicators.csv"
        if not os.path.exists(path):
            return "UNKNOWN", "Indicator data not found"

        df     = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        latest = df.iloc[-1]
        risks  = []

        rsi = latest.get("RSI", 50)
        if pd.notna(rsi):
            if float(rsi) > 70:
                risks.append(f"RSI overbought ({float(rsi):.0f})")
            elif float(rsi) < 30:
                risks.append(f"RSI oversold ({float(rsi):.0f})")

        macd_sig = str(latest.get("MACD_signal", "Neutral"))
        if "Bearish" in macd_sig:
            risks.append("MACD bearish crossover")

        roc = latest.get("ROC", 0)
        if pd.notna(roc) and float(roc) < -2:
            risks.append(f"ROC negative ({float(roc):.1f})")

        # Also check trend
        tre_path = f"data/{fname}_trend.csv"
        if os.path.exists(tre_path):
            tre    = pd.read_csv(tre_path)
            tre_l  = tre.iloc[-1]
            td     = tre_l.get("trend_direction", 0)
            if pd.notna(td) and int(td) == -1:
                risks.append("Downtrend active")

        level  = ("HIGH RISK"   if len(risks) >= 2 else
                  "MEDIUM RISK" if len(risks) == 1 else
                  "LOW RISK")
        reason = (", ".join(risks)
                  if risks else "All signals normal")
        return level, reason

    except Exception as e:
        return "UNKNOWN", str(e)

# ── CHATBOT RESPONSE LOGIC ────────────────────────────────────
def get_response(user_msg, ticker):
    msg = user_msg.lower().strip()

    # ── PREDICTION ────────────────────────────────────────────
    if any(w in msg for w in
           ["predict", "tomorrow", "forecast", "go up",
            "go down", "next", "direction", "buy", "sell"]):
        dirn, conf, acc = get_prediction(ticker)
        if dirn is None:
            err = acc if isinstance(acc, str) else ""
            return (f"Could not predict {ticker}. "
                    f"Make sure model files exist. {err}")
        emoji = "📈" if dirn == "UP" else "📉"
        acc_str = (f"{float(acc):.1%}"
                   if isinstance(acc, float) else "N/A")
        return (f"{emoji} **{ticker} Prediction:**\n\n"
                f"Tomorrow's direction: **{dirn}**\n"
                f"Confidence: **{conf:.1f}%**\n"
                f"Model accuracy: **{acc_str}**")

    # ── RISK ──────────────────────────────────────────────────
    elif any(w in msg for w in
             ["risk", "safe", "danger", "should i",
              "invest", "risky", "volatile"]):
        level, reason = get_risk(ticker)
        emoji = ("🔴" if "HIGH" in level else
                 "🟡" if "MEDIUM" in level else "🟢")
        return (f"{emoji} **{ticker} Risk Level: {level}**\n\n"
                f"Reason: {reason}")

    # ── INDICATORS ────────────────────────────────────────────
    elif any(w in msg for w in
             ["rsi", "indicator", "macd", "momentum",
              "overbought", "oversold", "roc"]):
        fname = FILE_MAP.get(ticker, ticker)
        path  = f"data/{fname}_indicators.csv"
        if not os.path.exists(path):
            return "Indicator data not found."
        df     = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        latest = df.iloc[-1]

        rsi     = latest.get("RSI", "N/A")
        macd    = latest.get("MACD_signal", "N/A")
        roc     = latest.get("ROC", "N/A")
        sig     = latest.get("indicator_signal", 0)
        overall = ("Bullish" if sig == 1 else
                   "Bearish" if sig == -1 else "Neutral")

        rsi_str = f"{float(rsi):.1f}" \
                  if pd.notna(rsi) else "N/A"
        roc_str = f"{float(roc):.2f}" \
                  if pd.notna(roc) else "N/A"

        rsi_label = ""
        if pd.notna(rsi):
            r = float(rsi)
            rsi_label = (" (Overbought)" if r > 70 else
                         " (Oversold)"   if r < 30 else
                         " (Neutral)")

        return (f"📊 **{ticker} Indicators:**\n\n"
                f"RSI: **{rsi_str}**{rsi_label}\n"
                f"MACD: **{macd}**\n"
                f"ROC: **{roc_str}**\n\n"
                f"Overall Signal: **{overall}**")

    # ── TREND ─────────────────────────────────────────────────
    elif any(w in msg for w in
             ["trend", "support", "resistance",
              "uptrend", "downtrend", "sideways"]):
        fname = FILE_MAP.get(ticker, ticker)
        path  = f"data/{fname}_trend.csv"
        if not os.path.exists(path):
            return "Trend data not found."
        df     = pd.read_csv(path)
        latest = df.iloc[-1]
        td     = latest.get("trend_direction", 0)
        sp     = latest.get("support_price", 0)
        rp     = latest.get("resistance_price", 0)
        trend  = ("Uptrend 📈"  if td == 1  else
                  "Downtrend 📉" if td == -1 else
                  "Sideways ↔️")
        return (f"📉 **{ticker} Trend:**\n\n"
                f"Direction: **{trend}**\n"
                f"Support: **{float(sp):.2f}**\n"
                f"Resistance: **{float(rp):.2f}**")

    # ── PATTERN ───────────────────────────────────────────────
    elif any(w in msg for w in
             ["pattern", "chart", "formation",
              "double", "triangle", "head"]):
        fname = FILE_MAP.get(ticker, ticker)
        path  = f"data/{fname}_patterns.csv"
        if not os.path.exists(path):
            return "Pattern data not found."
        df     = pd.read_csv(path)
        latest = df.iloc[-1]
        plabel = int(latest.get("pattern_label", 0))
        names  = {
            0: "No pattern detected ⚪",
            1: "Double Bottom 📈 (Bullish reversal)",
            2: "Head & Shoulders 📉 (Bearish reversal)",
            3: "Descending Triangle 📉 (Bearish)"
        }
        return (f"🔍 **{ticker} Pattern:**\n\n"
                f"{names.get(plabel, 'Unknown')}")

    # ── PRICE ─────────────────────────────────────────────────
    elif any(w in msg for w in
             ["price", "current", "value",
              "worth", "trading", "close"]):
        fname = FILE_MAP.get(ticker, ticker)
        path  = f"data/{fname}.csv"
        if not os.path.exists(path):
            return "Price data not found."
        df    = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.title()
        close = df["Close"].iloc[-1]
        date  = df["Date"].iloc[-1]
        return (f"💰 **{ticker} Latest Price:**\n\n"
                f"Price: **{close:.2f}**\n"
                f"Date:  {date}\n\n"
                f"_(From our dataset — not live)_")

    # ── HELP ──────────────────────────────────────────────────
    else:
        return (f"I can help with **{ticker}**! Try:\n\n"
                f"- Will {ticker} go up tomorrow?\n"
                f"- Is {ticker} high risk?\n"
                f"- What is the RSI for {ticker}?\n"
                f"- What trend is {ticker} showing?\n"
                f"- What pattern does {ticker} show?\n"
                f"- What is the price of {ticker}?")

# ── STREAMLIT UI ──────────────────────────────────────────────
def run_chatbot():
    st.title("Stock Analysis Chatbot")
    st.markdown(
        "Ask me anything about AAPL, TSLA or RELIANCE!"
    )

    # Stock selector
    ticker = st.selectbox(
        "Select Stock:",
        ["AAPL", "TSLA", "RELIANCE"]
    )

    # Quick buttons
    st.markdown("**Quick questions:**")
    c1, c2, c3 = st.columns(3)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with c1:
        if st.button("Predict tomorrow"):
            q = f"Will {ticker} go up tomorrow?"
            r = get_response(q, ticker)
            st.session_state.messages += [
                {"role": "user",    "content": q},
                {"role": "assistant","content": r}
            ]
            st.rerun()
    with c2:
        if st.button("Check risk"):
            q = f"Is {ticker} risky?"
            r = get_response(q, ticker)
            st.session_state.messages += [
                {"role": "user",    "content": q},
                {"role": "assistant","content": r}
            ]
            st.rerun()
    with c3:
        if st.button("Show indicators"):
            q = f"What are the indicators for {ticker}?"
            r = get_response(q, ticker)
            st.session_state.messages += [
                {"role": "user",    "content": q},
                {"role": "assistant","content": r}
            ]
            st.rerun()

    st.markdown("---")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Text input
    if prompt := st.chat_input(
        f"Ask about {ticker}..."
    ):
        response = get_response(prompt, ticker)
        st.session_state.messages += [
            {"role": "user",     "content": prompt},
            {"role": "assistant","content": response}
        ]
        st.rerun()

if __name__ == "__main__":
    run_chatbot()