import pandas as pd
from pathlib import Path

# Path Configuration
# Get base project directory (one level up from the script location)
BASE = Path(__file__).resolve().parents[1]
# Define path to processed data
PROC = BASE / "data" / "processed"

# Input files
GLOBAL_SENT = PROC / "global_sentiment_daily.csv"
NASDAQ_DATA = PROC / "nasdaq_market_data.csv"

# Output merged file
OUT_MERGED = PROC / "merged_sentiment_market.csv"

# Load global sentiment and NASDAQ market data
def load_data():
    # Load global daily sentiment file
    df_sent = pd.read_csv(GLOBAL_SENT)
    # Load NASDAQ OHLCV data
    df_mkt = pd.read_csv(NASDAQ_DATA)

    # Ensure date columns are parsed correctly
    df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.date
    df_mkt["date"] = pd.to_datetime(df_mkt["date"]).dt.date

    return df_sent, df_mkt

# Merge global sentiment with NASDAQ market data
def merge_data(df_sent, df_mkt):
    # Merge datasets on the common 'date' column
    merged = pd.merge(
        df_mkt,          # Market data first
        df_sent,         # Sentiment daily
        on="date",
        how="inner"      # Keep only matching rows
    )
    return merged

# Main Execution Pipeline
def main():
    # Load sentiment and market data
    df_sent, df_mkt = load_data()

    # Perform merge operation
    merged = merge_data(df_sent, df_mkt)

    # Save merged dataset
    merged.to_csv(OUT_MERGED, index=False)

    # Print success information
    print("\nMerged dataset created successfully:\n")
    print(f"- Rows: {len(merged)}")
    print(f"- Saved to: {OUT_MERGED}")

if __name__ == "__main__":
    main()
