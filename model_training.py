import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)

# ── FILE MAP ───────────────────────────────────────────────────
FILE_MAP = {
    "AAPL":     "AAPL",
    "TSLA":     "TSLA",
    "RELIANCE": "RELIANCE"
}

# ── ADD FEATURES (only percentage/ratio based — NO raw prices) ─
# KEY FIX: never use raw Close/lag prices as features directly
# They cause the scaler to break — use returns and ratios instead
def add_extra_features(df):

    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # Moving average ratios (price / SMA — normalized)
    df["SMA_10"]       = c.rolling(10).mean()
    df["SMA_20"]       = c.rolling(20).mean()
    df["EMA_10"]       = c.ewm(span=10).mean()
    df["price_to_sma10"] = c / df["SMA_10"] - 1   # ratio not raw
    df["price_to_sma20"] = c / df["SMA_20"] - 1
    df["sma10_to_sma20"] = df["SMA_10"] / df["SMA_20"] - 1
    df["SMA_cross"]    = (df["SMA_10"] > df["SMA_20"]).astype(int)

    # ATR as percentage of price (normalized volatility)
    tr                 = h - l
    df["ATR_pct"]      = tr.rolling(14).mean() / c

    # Volume features (ratios — not raw volume)
    df["volume_ratio"] = v / v.rolling(20).mean()
    df["volume_change"]= v.pct_change()

    # Price returns (percentage changes — safe to scale)
    df["return_1d"]    = c.pct_change(1)
    df["return_3d"]    = c.pct_change(3)
    df["return_5d"]    = c.pct_change(5)
    df["return_10d"]   = c.pct_change(10)

    # Lag RETURNS not lag prices (KEY FIX)
    df["lag_ret_1"]    = df["return_1d"].shift(1)
    df["lag_ret_2"]    = df["return_1d"].shift(2)
    df["lag_ret_3"]    = df["return_1d"].shift(3)

    # Distance from 20-day high/low as percentage
    roll_high          = h.rolling(20).max()
    roll_low           = l.rolling(20).min()
    df["dist_high"]    = (c - roll_high) / c
    df["dist_low"]     = (c - roll_low)  / c

    # Candle body and wick sizes (normalized)
    df["body_size"]    = abs(c - df["Open"]) / c
    df["upper_wick"]   = (h - c.clip(lower=df["Open"])) / c
    df["lower_wick"]   = (c.clip(upper=df["Open"]) - l) / c

    # Momentum: rate of change
    df["ROC_5"]        = c.pct_change(5)
    df["ROC_10"]       = c.pct_change(10)

    # Consecutive up/down days streak
    daily_dir          = np.sign(df["return_1d"].fillna(0))
    streak             = []
    count              = 0
    for d in daily_dir:
        if d > 0:
            count = max(1, count + 1)
        elif d < 0:
            count = min(-1, count - 1)
        else:
            count = 0
        streak.append(count)
    df["streak"]       = streak

    # Drop intermediate columns not used as features
    df.drop(columns=["SMA_10","SMA_20","EMA_10"],
            inplace=True, errors="ignore")

    return df

# ── LOAD AND MERGE all files for one ticker ───────────────────
def load_merged(ticker):
    fname = FILE_MAP.get(ticker, ticker)

    paths = {
        "base": f"data/{fname}.csv",
        "ind":  f"data/{fname}_indicators.csv",
        "pat":  f"data/{fname}_patterns.csv",
        "tre":  f"data/{fname}_trend.csv"
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path} — run the "
                f"{name} script first!"
            )

    base = pd.read_csv(paths["base"])
    ind  = pd.read_csv(paths["ind"])
    pat  = pd.read_csv(paths["pat"])
    tre  = pd.read_csv(paths["tre"])

    # Flatten MultiIndex columns (yfinance sometimes adds these)
    for d in [base, ind, pat, tre]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = ['_'.join(
                [str(x) for x in c if str(x) != '']
            ).strip() for c in d.columns]

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

    # Merge all on Date
    df = base.merge(ind, on="Date", how="left")
    df = df.merge(pat, on="Date", how="left")
    df = df.merge(tre, on="Date", how="left")

    # Remove duplicate Close columns
    cc = [c for c in df.columns if c.lower() == "close"]
    if len(cc) > 1:
        df.drop(columns=cc[1:], inplace=True, errors="ignore")

    if "Close" not in df.columns:
        raise KeyError(f"No Close column for {ticker}")

    # Sort oldest to newest
    df = df.sort_values("Date").reset_index(drop=True)

    # Add engineered features
    df = add_extra_features(df)

    # Target: 1 = UP next day, 0 = DOWN next day
    df["target"] = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)

    # Fill NaN in features only (not Close/target)
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df.dropna(subset=["Close", "target"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ── FEATURE LIST (all percentage/ratio based — safe to scale) ──
FEATURES = [
    # From indicator_analysis.py
    "RSI",
    "MACD",
    "ROC",
    "indicator_signal",
    # From pattern_analysis.py
    "pattern_label",
    # From trend_analysis.py
    "trend_direction",
    "trend_slope",
    # New normalized features
    "price_to_sma10",
    "price_to_sma20",
    "sma10_to_sma20",
    "SMA_cross",
    "ATR_pct",
    "volume_ratio",
    "volume_change",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "lag_ret_1",
    "lag_ret_2",
    "lag_ret_3",
    "dist_high",
    "dist_low",
    "body_size",
    "upper_wick",
    "lower_wick",
    "ROC_5",
    "ROC_10",
    "streak"
]

# ── TRAIN all models and pick best ───────────────────────────
def train_model(ticker):
    print(f"\n{'='*40}")
    print(f"Training model for {ticker}...")

    df        = load_merged(ticker)
    available = [f for f in FEATURES if f in df.columns]

    if not available:
        print(f"No features found — skipping {ticker}")
        return None, []

    print(f"Features       : {len(available)}")
    print(f"Total samples  : {len(df)}")
    print(f"UP  (1)        : {int(df['target'].sum())}")
    print(f"DOWN(0)        : {int((df['target']==0).sum())}")

    if len(df) < 100:
        print("Not enough data (need 100+ rows)")
        return None, []

    X = df[available].values
    y = df["target"].values

    # Time-based split — last 20% as test
    split     = int(len(X) * 0.8)
    X_train   = X[:split]
    X_test    = X[split:]
    y_train   = y[:split]
    y_test    = y[split:]

    # Scale — fit on train only, apply to test
    scaler         = StandardScaler()
    X_train_s      = scaler.fit_transform(X_train)
    X_test_s       = scaler.transform(X_test)

    results = {}

    # ── Model 1: Random Forest (balanced) ─────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=30,
        min_samples_leaf=15,
        class_weight="balanced",   # KEY FIX for imbalance
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_s))
    results["RandomForest"] = (rf, rf_acc)
    print(f"\nRandom Forest  : {rf_acc:.2%}")

    # ── Model 2: XGBoost (scale_pos_weight fixes imbalance) ───
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb   = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,    # KEY FIX for class imbalance
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(X_train_s, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test_s))
    results["XGBoost"] = (xgb, xgb_acc)
    print(f"XGBoost        : {xgb_acc:.2%}")

    # ── Model 3: Gradient Boosting ────────────────────────────
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train_s, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test_s))
    results["GradientBoosting"] = (gb, gb_acc)
    print(f"GradientBoost  : {gb_acc:.2%}")

    # ── Model 4: Logistic Regression (simple baseline) ────────
    lr = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train_s, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_s))
    results["LogisticRegression"] = (lr, lr_acc)
    print(f"LogisticReg    : {lr_acc:.2%}")

    # ── Pick the best model ───────────────────────────────────
    best_name  = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    best_acc   = results[best_name][1]

    # Print full report for best model
    y_pred = best_model.predict(X_test_s)
    print(f"\nBest model: {best_name} ({best_acc:.2%})")
    print(classification_report(
        y_test, y_pred,
        target_names=["DOWN","UP"],
        zero_division=0
    ))

    # Save model + scaler + feature list
    model_path = f"models/{ticker}_model.pkl"
    joblib.dump({
        "model":    best_model,
        "scaler":   scaler,
        "features": available,
        "ticker":   ticker,
        "accuracy": best_acc
    }, model_path)

    print(f"Saved to: {model_path}")

    # Top 5 important features (if available)
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(
            best_model.feature_importances_,
            index=available
        ).sort_values(ascending=False).head(5)
        print("\nTop 5 features:")
        for feat, score in imp.items():
            print(f"  {feat:20s}: {score:.4f}")

    return best_model, available

# ── PREDICT tomorrow ──────────────────────────────────────────
def predict_tomorrow(ticker):
    path = f"models/{ticker}_model.pkl"
    if not os.path.exists(path):
        return f"No model for {ticker} — train first"

    saved    = joblib.load(path)
    model    = saved["model"]
    scaler   = saved["scaler"]
    features = saved["features"]
    acc      = saved.get("accuracy", 0)

    df     = load_merged(ticker)
    latest = df[features].iloc[[-1]].values
    latest_s = scaler.transform(latest)

    pred  = model.predict(latest_s)[0]
    proba = model.predict_proba(latest_s)[0]
    conf  = max(proba) * 100
    dirn  = "UP" if pred == 1 else "DOWN"

    return (f"{ticker}: {dirn} — {conf:.1f}% confidence "
            f"(model acc: {acc:.1%})")

# ── RISK LEVEL ────────────────────────────────────────────────
def risk_level(ticker):
    path = f"models/{ticker}_model.pkl"
    if not os.path.exists(path):
        return f"No model for {ticker}"

    saved    = joblib.load(path)
    features = saved["features"]
    df       = load_merged(ticker)
    latest   = df.iloc[-1]
    risks    = []

    rsi = latest.get("RSI", 50)
    if "RSI" in features:
        if rsi > 70:
            risks.append(f"RSI overbought ({rsi:.0f})")
        elif rsi < 30:
            risks.append(f"RSI oversold ({rsi:.0f})")

    if "trend_direction" in features:
        if latest.get("trend_direction", 0) == -1:
            risks.append("Downtrend active")

    if "pattern_label" in features:
        p = latest.get("pattern_label", 0)
        if p == 2: risks.append("Head & Shoulders")
        elif p == 3: risks.append("Descending Triangle")

    if "return_5d" in features:
        r = latest.get("return_5d", 0)
        if r < -0.05:
            risks.append(f"5d return: {r:.1%}")

    if "ATR_pct" in features:
        if latest.get("ATR_pct", 0) > 0.025:
            risks.append("High volatility")

    level = ("HIGH RISK"   if len(risks) >= 2 else
             "MEDIUM RISK" if len(risks) == 1 else
             "LOW RISK")

    reason = ", ".join(risks) if risks else "No risk signals"
    return f"{ticker}: {level} — {reason}"

# ── RUN ───────────────────────────────────────────────────────
tickers = ["AAPL", "TSLA", "RELIANCE"]

for t in tickers:
    try:
        train_model(t)
    except FileNotFoundError as e:
        print(f"Skipping {t}: {e}")
    except Exception as e:
        print(f"Error {t}: {e}")

print("\n" + "="*40)
print("FINAL PREDICTIONS:")
print("="*40)
for t in tickers:
    try:
        print(predict_tomorrow(t))
        print(risk_level(t))
        print()
    except Exception as e:
        print(f"{t}: {e}")

print("Done!")