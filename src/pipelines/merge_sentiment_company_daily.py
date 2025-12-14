import pandas as pd
import numpy as np
from pathlib import Path

# Path Configuration
BASE = Path(__file__).resolve().parents[2]
PROC = BASE / "data" / "processed"

# Input files
TWEET_SENT = PROC / "tweet_sentiment.csv"  # Tweet-level sentiment with FinBERT
COMPANY_DATA = PROC / "company_stock_data.csv"

# Output merged file
OUT_MERGED = PROC / "merged_sentiment_stock_company.csv"

# Sample output
SAMPLES = PROC / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)
OUT_MERGED_SAMPLE = SAMPLES / "merged_sentiment_stock_company_SAMPLE.csv"


def compute_tweet_impact(df):
    """
    Compute impact score for each tweet based on engagement metrics.
    
    Uses logarithmic scaling to prevent viral tweets from dominating.
    Comments weighted higher than likes as they indicate deeper engagement.
    """
    engagement = df["retweet_num"] + 2 * df["like_num"] + 3 * df["comment_num"]
    impact = 1 + np.log1p(engagement)
    return impact


def compute_writer_influence(df):
    """
    Calculate historical influence of each writer based on their average engagement.
    
    Writers with consistently high engagement get higher weights,
    preventing one-off viral tweets from skewing sentiment.
    """
    writer_stats = df.groupby("writer").agg({
        "retweet_num": "mean",
        "like_num": "mean",
        "comment_num": "mean"
    }).reset_index()
    
    writer_stats["influence"] = (
        writer_stats["retweet_num"] + 
        2 * writer_stats["like_num"] + 
        3 * writer_stats["comment_num"]
    )
    
    # Prevent division by zero for writers with no historical engagement
    writer_stats["influence"] = 1 + writer_stats["influence"]
    
    return dict(zip(writer_stats["writer"], writer_stats["influence"]))


def aggregate_sentiment_daily(df):
    """
    Aggregate tweet-level sentiment to daily company-level features.
    
    Combines simple averages with weighted metrics to capture both
    overall mood and influence-adjusted sentiment strength.
    """
    df["impact"] = compute_tweet_impact(df)
    
    writer_influence_map = compute_writer_influence(df)
    df["writer_influence"] = df["writer"].map(writer_influence_map)
    
    df["weighted_polarity"] = df["polarity"] * df["impact"]
    df["influence_weighted_polarity"] = df["polarity"] * df["writer_influence"]
    
    df["is_extreme"] = (df["polarity"].abs() == 1.0).astype(int)
    
    daily_agg = df.groupby(["ticker_symbol", "date"]).agg({
        "positive": "mean",
        "neutral": "mean",
        "negative": "mean",
        "polarity": "mean",
        "impact": "sum",
        "weighted_polarity": "sum",
        "writer_influence": "mean",
        "influence_weighted_polarity": "sum",
        "tweet_id": "count",
        "is_extreme": "sum",
        "retweet_num": "sum",
        "like_num": "sum",
        "comment_num": "sum"
    }).reset_index()
    
    daily_agg.rename(columns={"tweet_id": "tweet_volume"}, inplace=True)
    
    # Normalize weighted sentiment by total impact/volume
    daily_agg["impact_weighted_sentiment"] = (
        daily_agg["weighted_polarity"] / daily_agg["impact"]
    )
    daily_agg["influence_weighted_sentiment"] = (
        daily_agg["influence_weighted_polarity"] / daily_agg["tweet_volume"]
    )
    
    daily_agg.drop(columns=["weighted_polarity", "influence_weighted_polarity"], inplace=True)
    
    daily_agg = daily_agg.sort_values(["ticker_symbol", "date"]).reset_index(drop=True)
    
    # Sentiment changes over time may be more predictive than absolute levels
    daily_agg["delta_polarity"] = daily_agg.groupby("ticker_symbol")["polarity"].diff()
    daily_agg["delta_impact_weighted_sentiment"] = daily_agg.groupby("ticker_symbol")["impact_weighted_sentiment"].diff()
    daily_agg["delta_tweet_volume"] = daily_agg.groupby("ticker_symbol")["tweet_volume"].diff()
    
    daily_agg.fillna(0, inplace=True)
    
    return daily_agg


def load_data():
    """Load tweet-level sentiment and stock data."""
    # Load tweet sentiment (output from sentiment_pipeline_company.py)
    df_tweets = pd.read_csv(TWEET_SENT)
    
    # Load company stock OHLCV data
    df_stock = pd.read_csv(COMPANY_DATA)

    # Ensure date columns are parsed correctly
    df_tweets["date"] = pd.to_datetime(df_tweets["date"]).dt.date
    df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.date

    return df_tweets, df_stock


def merge_data(df_sent, df_stock):
    """Merge company sentiment with stock data on ticker and date."""
    # Inner join ensures we only keep days with both sentiment AND price data
    merged = pd.merge(
        df_stock,        # Stock data first
        df_sent,         # Sentiment daily
        on=["ticker_symbol", "date"],
        how="inner"      # Keep only matching rows
    )
    return merged


def main():
    """Main Execution Pipeline."""
    # Load tweet sentiment and stock data
    df_tweets, df_stock = load_data()
    
    # Aggregate tweets to daily company-level features
    df_sent_daily = aggregate_sentiment_daily(df_tweets)
    
    # Perform merge operation
    merged = merge_data(df_sent_daily, df_stock)
    
    # Save merged dataset
    merged.to_csv(OUT_MERGED, index=False)
    
    # Save sample (first 1000 rows for quick testing)
    merged.head(1000).to_csv(OUT_MERGED_SAMPLE, index=False)

    # Print success information
    print("\nMerged dataset created successfully:\n")
    print(f"- Rows: {len(merged):,}")
    print(f"- Companies: {merged['ticker_symbol'].nunique()}")
    print(f"- Saved to: {OUT_MERGED.name}\n")


if __name__ == "__main__":
    main()
