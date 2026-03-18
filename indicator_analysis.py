import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import os

os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)

# ── Load all 3 stock files ──────────────────────────────────────
files = {
    "AAPL":       "data/AAPL.csv",
    "TSLA":       "data/TSLA.csv",
    "RELIANCE":   "data/RELIANCE.csv"
}

def compute_indicators(df, ticker):
    print(f"\nComputing indicators for {ticker}...")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # ── RSI (14-period) ────────────────────────────────────────
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    def rsi_label(x):
        if x > 70:   return "Overbought"
        elif x < 30: return "Oversold"
        else:        return "Neutral"

    df["RSI_signal"] = df["RSI"].apply(
        lambda x: rsi_label(x) if pd.notna(x) else "Neutral"
    )

    # ── MACD (12, 26, 9) ───────────────────────────────────────
    macd_obj        = ta.trend.MACD(close, window_slow=26,
                                    window_fast=12, window_sign=9)
    df["MACD"]      = macd_obj.macd()
    df["MACD_signal_line"] = macd_obj.macd_signal()
    df["MACD_hist"] = macd_obj.macd_diff()

    def macd_label(row):
        if pd.isna(row["MACD"]) or pd.isna(row["MACD_signal_line"]):
            return "Neutral"
        if row["MACD"] > row["MACD_signal_line"]:
            return "Bullish"
        elif row["MACD"] < row["MACD_signal_line"]:
            return "Bearish"
        return "Neutral"

    df["MACD_signal"] = df.apply(macd_label, axis=1)

    # ── ROC — Rate of Change (12-period) ──────────────────────
    df["ROC"] = ta.momentum.ROCIndicator(close, window=12).roc()

    def roc_label(x):
        if pd.isna(x):   return "Neutral"
        if x > 0:        return "Bullish"
        elif x < 0:      return "Bearish"
        return "Neutral"

    df["ROC_signal"] = df["ROC"].apply(roc_label)

    # ── Bollinger Bands (20-period) ────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"]  = bb.bollinger_hband()
    df["BB_lower"]  = bb.bollinger_lband()
    df["BB_middle"] = bb.bollinger_mavg()

    def bb_label(row):
        if pd.isna(row["BB_upper"]): return "Neutral"
        if row["Close"] > row["BB_upper"]:  return "Overbought"
        if row["Close"] < row["BB_lower"]:  return "Oversold"
        return "Neutral"

    df["BB_signal"] = df.apply(bb_label, axis=1)

    # ── Overall signal (combine all 4) ─────────────────────────
    def overall_signal(row):
        score = 0
        mapping = {"Bullish": 1, "Oversold": 1,
                   "Bearish": -1, "Overbought": -1,
                   "Neutral": 0}
        score += mapping.get(row["RSI_signal"],  0)
        score += mapping.get(row["MACD_signal"], 0)
        score += mapping.get(row["ROC_signal"],  0)
        score += mapping.get(row["BB_signal"],   0)
        if score > 0:  return 1   # Bullish
        if score < 0:  return -1  # Bearish
        return 0                  # Neutral

    df["indicator_signal"] = df.apply(overall_signal, axis=1)

    return df

# ── Plot and save charts ────────────────────────────────────────
def save_charts(df, ticker):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"{ticker} — Technical Indicators", fontsize=14)

    # Plot 1: Closing price + Bollinger Bands
    axes[0].plot(df["Date"], df["Close"],
                 label="Close", color="blue", linewidth=1)
    axes[0].plot(df["Date"], df["BB_upper"],
                 label="BB Upper", color="red",
                 linestyle="--", linewidth=0.8)
    axes[0].plot(df["Date"], df["BB_lower"],
                 label="BB Lower", color="green",
                 linestyle="--", linewidth=0.8)
    axes[0].fill_between(df["Date"],
                         df["BB_upper"], df["BB_lower"],
                         alpha=0.1, color="gray")
    axes[0].set_title("Price + Bollinger Bands")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)

    # Plot 2: RSI
    axes[1].plot(df["Date"], df["RSI"],
                 color="purple", linewidth=1)
    axes[1].axhline(70, color="red",   linestyle="--",
                    linewidth=0.8, label="Overbought (70)")
    axes[1].axhline(30, color="green", linestyle="--",
                    linewidth=0.8, label="Oversold (30)")
    axes[1].set_title("RSI (14-period)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)

    # Plot 3: MACD
    axes[2].plot(df["Date"], df["MACD"],
                 label="MACD", color="blue", linewidth=1)
    axes[2].plot(df["Date"], df["MACD_signal_line"],
                 label="Signal", color="orange", linewidth=1)
    axes[2].bar(df["Date"], df["MACD_hist"],
                label="Histogram", color="gray", alpha=0.4)
    axes[2].set_title("MACD (12, 26, 9)")
    axes[2].legend(fontsize=8)
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    chart_path = f"charts/{ticker}_indicators.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved: {chart_path}")

# ── Main loop — run for all 3 stocks ───────────────────────────
for ticker, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"File not found: {filepath} — skipping")
        continue

    df = pd.read_csv(filepath)
    df = compute_indicators(df, ticker)
    save_charts(df, ticker)

    # Save indicator CSV for Member 1 to merge
    out_path = f"data/{ticker}_indicators.csv"
    cols_to_save = [
        "Date", "RSI", "RSI_signal",
        "MACD", "MACD_signal_line", "MACD_signal",
        "ROC", "ROC_signal",
        "BB_upper", "BB_lower", "BB_signal",
        "indicator_signal"
    ]
    df[cols_to_save].to_csv(out_path, index=False)
    print(f"Indicators saved: {out_path}")

    # Print summary of latest signals
    latest = df.dropna().iloc[-1]
    print(f"\n{ticker} Latest Signals:")
    print(f"  RSI        : {latest['RSI']:.1f} → {latest['RSI_signal']}")
    print(f"  MACD       : {latest['MACD_signal']}")
    print(f"  ROC        : {latest['ROC']:.2f} → {latest['ROC_signal']}")
    print(f"  BB         : {latest['BB_signal']}")
    print(f"  Overall    : {int(latest['indicator_signal'])} "
          f"(1=Bullish, 0=Neutral, -1=Bearish)")

print("\nAll indicators done!")