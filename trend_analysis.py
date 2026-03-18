import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("data",   exist_ok=True)
os.makedirs("charts", exist_ok=True)

files = {
    "AAPL":     "data/AAPL.csv",
    "TSLA":     "data/TSLA.csv",
    "RELIANCE": "data/RELIANCE.csv"
}

# ── Compute support and resistance using OLS regression ────────
# For each row, look back 'window' bars and fit a line
# through the highs (resistance) and lows (support)
def compute_trend(df, window=30):
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    n     = len(df)

    trend_direction = np.zeros(n, dtype=int)
    support_price   = np.full(n, np.nan)
    resistance_price= np.full(n, np.nan)
    trend_slope     = np.full(n, np.nan)

    for i in range(window, n):
        seg_high = high[i - window:i]
        seg_low  = low[i  - window:i]
        x        = np.arange(window)

        # OLS regression on highs → resistance line slope
        slope_r = np.polyfit(x, seg_high, 1)[0]
        # OLS regression on lows  → support line slope
        slope_s = np.polyfit(x, seg_low,  1)[0]

        # Average slope of both lines
        avg_slope = (slope_r + slope_s) / 2

        # Normalize slope relative to price level
        price_level    = np.mean(seg_low)
        norm_slope     = avg_slope / price_level if price_level != 0 else 0
        trend_slope[i] = norm_slope

        # Classify trend
        if norm_slope > 0.0001:
            trend_direction[i] = 1    # Uptrend
        elif norm_slope < -0.0001:
            trend_direction[i] = -1   # Downtrend
        else:
            trend_direction[i] = 0    # Sideways

        # Current support and resistance price levels
        support_price[i]    = seg_low[-1]
        resistance_price[i] = seg_high[-1]

    df["trend_direction"]  = trend_direction
    df["support_price"]    = support_price
    df["resistance_price"] = resistance_price
    df["trend_slope"]      = trend_slope

    return df

# ── Save chart with support and resistance lines ───────────────
def save_trend_chart(df, ticker):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(f"{ticker} — Trend Analysis", fontsize=14)

    ax.plot(df["Date"], df["Close"],
            color="steelblue", linewidth=1,
            label="Close Price", zorder=3)

    ax.plot(df["Date"], df["support_price"],
            color="green", linewidth=1,
            linestyle="--", label="Support", alpha=0.7)

    ax.plot(df["Date"], df["resistance_price"],
            color="red", linewidth=1,
            linestyle="--", label="Resistance", alpha=0.7)

    # Color background by trend direction
    up   = df["trend_direction"] == 1
    down = df["trend_direction"] == -1

    ax.fill_between(df["Date"], df["Close"].min(),
                    df["Close"].max(),
                    where=up,   alpha=0.06,
                    color="green", label="Uptrend zone")
    ax.fill_between(df["Date"], df["Close"].min(),
                    df["Close"].max(),
                    where=down, alpha=0.06,
                    color="red",   label="Downtrend zone")

    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    plt.tight_layout()
    path = f"charts/{ticker}_trend.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Chart saved: {path}")

# ── Main loop ──────────────────────────────────────────────────
for ticker, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"File not found: {filepath} — skipping")
        continue

    print(f"\nAnalyzing trend for {ticker}...")

    df          = pd.read_csv(filepath)
    df["Date"]  = pd.to_datetime(df["Date"])
    df          = df.sort_values("Date").reset_index(drop=True)

    df = compute_trend(df)

    # Summary
    counts = df["trend_direction"].value_counts()
    names  = {1: "Uptrend", 0: "Sideways", -1: "Downtrend"}
    print(f"\n{ticker} Trend Summary:")
    for tid, count in sorted(counts.items()):
        print(f"  {names.get(tid, tid)}: {count} rows")

    # Latest trend
    latest = df.dropna(subset=["trend_direction"]).iloc[-1]
    print(f"Latest trend     : {names[latest['trend_direction']]}")
    print(f"Support price    : {latest['support_price']:.2f}")
    print(f"Resistance price : {latest['resistance_price']:.2f}")

    # Save chart
    save_trend_chart(df, ticker)

    # Save trend CSV for Member 1
    out_path = f"data/{ticker}_trend.csv"
    df[["Date", "trend_direction",
        "support_price", "resistance_price",
        "trend_slope"]].to_csv(out_path, index=False)
    print(f"Trend CSV saved: {out_path}")

print("\nAll trends done!")