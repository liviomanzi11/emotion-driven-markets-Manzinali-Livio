import pandas as pd
import numpy as np
from pathlib import Path

# Path configuration
BASE = Path(__file__).resolve().parents[2]
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# Folder for company features
FEATURES_COMPANY = PROC / "features_company"
FEATURES_COMPANY.mkdir(parents=True, exist_ok=True)

# Input file
INPUT_MERGED = PROC / "merged_sentiment_stock_company.csv"

# Output files (will be split by company later)
OUT_DIR = FEATURES_COMPANY


def compute_moving_average(series, window):
    """Compute moving average for a series."""
    return series.rolling(window=window).mean()


def compute_volatility(series, window):
    """Compute rolling volatility of returns."""
    return series.rolling(window=window).std()


def compute_rsi(series, period=14):
    """Compute RSI indicator."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series):
    """Compute MACD line and signal line."""
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def engineer_features_for_company(df_company, ticker):
    """
    Engineer features for a single company.
    Returns a DataFrame with technical and enriched sentiment features.
    """
    df = df_company.copy()
    
    # Time-series operations require proper datetime index for rolling windows and shifts
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    
    # Keep original FinBERT polarity (already computed in merge step)
    # But also add moving averages of enriched sentiment
    
    # Moving averages of impact-weighted sentiment
    df["impact_weighted_ma3"] = df["impact_weighted_sentiment"].rolling(3).mean()
    df["impact_weighted_ma7"] = df["impact_weighted_sentiment"].rolling(7).mean()
    
    # Moving averages of influence-weighted sentiment
    df["influence_weighted_ma3"] = df["influence_weighted_sentiment"].rolling(3).mean()
    df["influence_weighted_ma7"] = df["influence_weighted_sentiment"].rolling(7).mean()
    
    # Sentiment volatility (standard deviation of polarity over time)
    df["sentiment_volatility_5d"] = df["polarity"].rolling(5).std()
    df["sentiment_volatility_10d"] = df["polarity"].rolling(10).std()
    
    # Cumulative extreme sentiment over past N days
    df["extreme_count_5d"] = df["is_extreme"].rolling(5).sum()
    df["extreme_count_10d"] = df["is_extreme"].rolling(10).sum()
    
    # Tweet volume momentum (rate of change)
    df["tweet_volume_ma5"] = df["tweet_volume"].rolling(5).mean()
    df["tweet_volume_change_rate"] = df["tweet_volume"].pct_change()
    
    # Engagement momentum
    df["total_engagement"] = df["retweet_num"] + df["like_num"] + df["comment_num"]
    df["engagement_ma5"] = df["total_engagement"].rolling(5).mean()
    # Safe pct_change: avoid division by zero
    df["engagement_change"] = df["total_engagement"].pct_change().replace([np.inf, -np.inf], 0)
    
    
    # Price-based features (use 'adj close' with space, not 'adj_close')
    df["return"] = df["adj close"].pct_change()
    df["log_return"] = np.log(df["adj close"] / df["adj close"].shift(1))
    
    # Moving averages
    df["ma5"] = compute_moving_average(df["adj close"], 5)
    df["ma10"] = compute_moving_average(df["adj close"], 10)
    df["ma20"] = compute_moving_average(df["adj close"], 20)
    
    # Volatility
    df["volatility_5d"] = compute_volatility(df["return"], 5)
    df["volatility_10d"] = compute_volatility(df["return"], 10)
    df["volatility_20d"] = compute_volatility(df["return"], 20)
    
    # RSI
    df["rsi_14"] = compute_rsi(df["adj close"], 14)
    
    # MACD
    df["macd_line"], df["signal_line"] = compute_macd(df["adj close"])
    
    # Target: stock up/down next day
    df["target"] = (df["adj close"].shift(-1) > df["adj close"]).astype(int)
    
    # Last row has no future price to predict - cannot compute target
    df = df.dropna(subset=["target"])
    
    # Rolling indicators create NaN for initial periods (e.g., MA20 needs 20 days)
    # Sort by date BEFORE dropna to ensure deterministic row ordering
    df = df.sort_values("date").dropna().reset_index(drop=True)
    
    return df


def main():
    """
    Main pipeline for feature engineering per company.
    Processes each company separately and saves train/test splits.
    """
    print("\n→ Loading merged company data...")
    df_all = pd.read_csv(INPUT_MERGED)
    
    tickers = df_all["ticker_symbol"].unique()
    print(f"  Found {len(tickers)} companies: {', '.join(tickers)}\n")
    
    for ticker in tickers:
        print(f"→ Processing {ticker}...", end=" ", flush=True)
        
        # Filter data for this company
        df_company = df_all[df_all["ticker_symbol"] == ticker].copy()
        
        # Engineer features
        df_features = engineer_features_for_company(df_company, ticker)
        
        if len(df_features) < 100:
            print(f"❌ Insufficient data ({len(df_features)} rows)")
            continue
        
        # Temporal split prevents lookahead bias - test period (2019) never seen during training (2015-2018)
        train = df_features[(df_features["date"] >= "2015-01-01") & (df_features["date"] <= "2018-12-31")]
        test = df_features[(df_features["date"] >= "2019-01-01") & (df_features["date"] <= "2019-12-31")]
        
        if len(train) < 50 or len(test) < 20:
            print(f"❌ Insufficient train/test data (train: {len(train)}, test: {len(test)})")
            continue
        
        # Save files
        train.to_csv(OUT_DIR / f"{ticker}_features_train.csv", index=False)
        test.to_csv(OUT_DIR / f"{ticker}_features_test.csv", index=False)
        
        print(f"[OK] (train: {len(train)}, test: {len(test)})")
    
    print(f"\n[OK] Feature engineering complete. Files saved in {OUT_DIR.name}/\n")


if __name__ == "__main__":
    main()
