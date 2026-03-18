import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.signal import argrelextrema

os.makedirs("data",   exist_ok=True)
os.makedirs("charts", exist_ok=True)

files = {
    "AAPL":     "data/AAPL.csv",
    "TSLA":     "data/TSLA.csv",
    "RELIANCE": "data/RELIANCE.csv"
}

# ── PATTERN 1: Double Bottom ───────────────────────────────────
# Logic: Find 2 local lows that are close in price with a
#        rebound peak between them — forms a W shape
def detect_double_bottom(df, order=10, tolerance=0.03):
    close  = df["Close"].values
    labels = np.zeros(len(df), dtype=int)

    # Find all local minima (troughs)
    lows = argrelextrema(close, np.less, order=order)[0]

    for i in range(len(lows) - 1):
        l1, l2 = lows[i], lows[i + 1]
        price1, price2 = close[l1], close[l2]

        # Both lows must be within 3% of each other
        if abs(price1 - price2) / price1 > tolerance:
            continue

        # There must be a rebound peak between the two lows
        between = close[l1:l2]
        if len(between) < 3:
            continue

        peak = np.max(between)
        # Peak must be at least 2% above both lows
        if peak < price1 * 1.02:
            continue

        # Label the region as Double Bottom
        labels[l1:l2 + 1] = 1

    return labels

# ── PATTERN 2: Head and Shoulders ─────────────────────────────
# Logic: Find 3 local peaks where the middle one (head)
#        is higher than both sides (shoulders)
def detect_head_and_shoulders(df, order=10, tolerance=0.05):
    close  = df["Close"].values
    labels = np.zeros(len(df), dtype=int)

    # Find all local maxima (peaks)
    highs = argrelextrema(close, np.greater, order=order)[0]

    for i in range(len(highs) - 2):
        l, h, r = highs[i], highs[i + 1], highs[i + 2]
        lp, hp, rp = close[l], close[h], close[r]

        # Head must be higher than both shoulders
        if hp <= lp or hp <= rp:
            continue

        # Both shoulders must be within 5% of each other
        if abs(lp - rp) / lp > tolerance:
            continue

        # Label the region as Head and Shoulders
        labels[l:r + 1] = 2

    return labels

# ── PATTERN 3: Descending Triangle ────────────────────────────
# Logic: Find a window where highs are falling (resistance
#        sloping down) but lows stay flat (flat support)
def detect_descending_triangle(df, window=30, tolerance=0.02):
    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    labels = np.zeros(len(df), dtype=int)

    for i in range(window, len(df)):
        segment_high = high[i - window:i]
        segment_low  = low[i  - window:i]

        # Check if highs are falling (negative slope)
        x       = np.arange(window)
        slope_h = np.polyfit(x, segment_high, 1)[0]
        if slope_h >= 0:
            continue

        # Check if lows are relatively flat
        slope_l = np.polyfit(x, segment_low, 1)[0]
        low_mean = np.mean(segment_low)
        if low_mean == 0:
            continue
        if abs(slope_l / low_mean) > tolerance:
            continue

        # Label the end of this window as Descending Triangle
        labels[i] = 3

    return labels

# ── CHART: Plot patterns on candlestick-style chart ───────────
def save_pattern_chart(df, ticker, labels):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(f"{ticker} — Pattern Analysis", fontsize=14)

    # Plot closing price line
    ax.plot(df["Date"], df["Close"],
            color="steelblue", linewidth=1, label="Close Price")

    # Shade pattern regions with different colors
    colors = {
        1: ("green",  "Double Bottom"),
        2: ("red",    "Head & Shoulders"),
        3: ("orange", "Descending Triangle")
    }

    for pattern_id, (color, name) in colors.items():
        mask = labels == pattern_id
        if mask.any():
            ax.fill_between(
                df["Date"], df["Close"],
                where=mask,
                alpha=0.25,
                color=color,
                label=name
            )

    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    plt.tight_layout()
    path = f"charts/{ticker}_patterns.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Chart saved: {path}")

# ── MAIN: Run for all 3 stocks ─────────────────────────────────
for ticker, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"File not found: {filepath} — skipping")
        continue

    print(f"\nAnalyzing patterns for {ticker}...")

    df             = pd.read_csv(filepath)
    df["Date"]     = pd.to_datetime(df["Date"])
    df             = df.sort_values("Date").reset_index(drop=True)

    # Detect all 3 patterns
    db_labels  = detect_double_bottom(df)
    hs_labels  = detect_head_and_shoulders(df)
    dt_labels  = detect_descending_triangle(df)

    # Combine: priority order → Double Bottom > H&S > Triangle > None
    combined = np.zeros(len(df), dtype=int)
    combined[dt_labels == 3] = 3   # Descending Triangle
    combined[hs_labels == 2] = 2   # Head and Shoulders
    combined[db_labels == 1] = 1   # Double Bottom (highest priority)

    df["pattern_label"] = combined

    # Count how many of each pattern found
    counts = pd.Series(combined).value_counts()
    label_names = {0: "None", 1: "Double Bottom",
                   2: "Head & Shoulders", 3: "Descending Triangle"}
    print(f"\n{ticker} Pattern Summary:")
    for pid, count in sorted(counts.items()):
        print(f"  {label_names.get(pid, pid)}: {count} rows")

    # Save chart
    save_pattern_chart(df, ticker, combined)

    # Save pattern CSV for Member 1 to merge
    out_path = f"data/{ticker}_patterns.csv"
    df[["Date", "pattern_label"]].to_csv(out_path, index=False)
    print(f"Pattern CSV saved: {out_path}")

    # Show latest pattern detected
    latest_pattern = combined[combined != 0]
    if len(latest_pattern) > 0:
        last_id = combined[combined != 0][-1]
        print(f"Most recent pattern: {label_names[last_id]}")
    else:
        print("No strong patterns detected in recent data")

print("\nAll patterns done!")