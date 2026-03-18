import pandas as pd

def create_target(df):
    # Target: 1 = next day price goes UP, 0 = goes DOWN
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df


df_aapl = pd.read_csv("data/AAPL.csv")


df_aapl = create_target(df_aapl)

df_aapl.to_csv("data/AAPL_features.csv", index=False)
print("Feature file saved!")
print(df_aapl.head())