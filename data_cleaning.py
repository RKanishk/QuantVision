import pandas as pd
import os

files = [
    "data/AAPL.csv",
    "data/TSLA.csv",
    "data/RELIANCE.csv"
]

for filepath in files:
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue

    df = pd.read_csv(filepath)
    print(f"\n--- {filepath} ---")
    print(f"Rows before cleaning: {len(df)}")

    # Drop missing values
    df.dropna(inplace=True)

    # Drop duplicates
    df.drop_duplicates(subset="Date", inplace=True)

    # Make sure required columns exist
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            print(f"WARNING: Missing column: {col}")

    df.to_csv(filepath, index=False)
    print(f"Rows after cleaning: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

print("\nCleaning complete!")