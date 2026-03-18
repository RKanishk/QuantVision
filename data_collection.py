import yfinance as yf
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

stocks = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "RELIANCE": "Reliance"
}

for ticker, name in stocks.items():
    print(f"Downloading {name} ({ticker})...")
    df = yf.download(ticker, start="2022-01-01", end="2025-01-01")
    df.reset_index(inplace=True)
    filename = f"data/{ticker.replace('.', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename} — {len(df)} rows")

print("\nAll done! Check the data/ folder.")