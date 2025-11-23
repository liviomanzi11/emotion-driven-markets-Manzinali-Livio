import pandas as pd
import numpy as np
from pathlib import Path

# Path configuration
BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# Input file
INPUT_MERGED = PROC / "merged_sentiment_market.csv"

# Output files
OUT_FULL = PROC / "nasdaq_features_full.csv"
OUT_TRAIN = PROC / "nasdaq_features_train.csv"
OUT_TEST = PROC / "nasdaq_features_test.csv"



# Compute moving averages for closing price
def compute_moving_average(df, window):
    return df["close"].rolling(window=window).mean()

# Compute rolling volatility based on returns
def compute_volatility(df, window):
    return df["return"].rolling(window=window).std()

# Compute RSI indicator
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Compute MACD (EMA12 - EMA26) and signal line
def compute_macd(df):
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def main():
    # Load merged sentiment + market data
    df = pd.read_csv(INPUT_MERGED)

    # Convert date to datetime and sort
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")

    # Create polarity from FinBERT scores
    df["sentiment_label"] = df[["positive", "neutral", "negative"]].idxmax(axis=1)
    df["polarity"] = df["sentiment_label"].map({"positive": 1, "negative": -1}).fillna(0)

    # Price-based features
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Moving averages
    df["ma5"] = compute_moving_average(df, 5)
    df["ma10"] = compute_moving_average(df, 10)
    df["ma20"] = compute_moving_average(df, 20)

    # Volatility
    df["volatility_5d"] = compute_volatility(df, 5)
    df["volatility_10d"] = compute_volatility(df, 10)
    df["volatility_20d"] = compute_volatility(df, 20)

    # RSI
    df["rsi_14"] = compute_rsi(df, 14)

    # MACD
    df["macd_line"], df["signal_line"] = compute_macd(df)

    # Sentiment moving averages
    df["sentiment_ma3"] = df["polarity"].rolling(3).mean()
    df["sentiment_ma7"] = df["polarity"].rolling(7).mean()

    # Target: NASDAQ up/down next day
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Remove last row without target
    df = df.dropna(subset=["target"])

    # Remove rows where technical indicators cannot be computed (early days of MA/RSI/MACD)
    df = df.dropna().reset_index(drop=True)

    # Save full dataset
    df.to_csv(OUT_FULL, index=False)

    # Train/test split
    df["date"] = pd.to_datetime(df["date"])

    train = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2018-12-31")]
    test = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2019-12-31")]

    # Save train and test datasets
    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST, index=False)

    print("Feature engineering complete.")
    print(f"- Saved: {OUT_FULL.name}")
    print(f"- Saved: {OUT_TRAIN.name}")
    print(f"- Saved: {OUT_TEST.name}")


if __name__ == "__main__":
    main()
