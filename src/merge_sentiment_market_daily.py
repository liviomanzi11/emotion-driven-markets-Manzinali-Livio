import pandas as pd
from pathlib import Path

# Load sentiment and market data CSVs
def load_data():
    # Build portable path to /data/processed
    base = Path(__file__).resolve().parents[1]
    processed = base / "data" / "processed"

    # Load daily company sentiment
    df_sent = pd.read_csv(processed / "company_sentiment_daily.csv")

    # Load daily market OHLCV data
    df_mkt = pd.read_csv(processed / "market_data.csv")

    # Standardize column names
    df_sent.columns = df_sent.columns.str.lower().str.strip()
    df_mkt.columns = df_mkt.columns.str.lower().str.strip()

    # Convert date to datetime.date
    df_sent["date"] = pd.to_datetime(df_sent["date"], errors="coerce").dt.date
    df_mkt["date"] = pd.to_datetime(df_mkt["date"], errors="coerce").dt.date

    # Remove invalid dates
    df_sent = df_sent.dropna(subset=["date"])
    df_mkt = df_mkt.dropna(subset=["date"])

    return df_sent, df_mkt


# Merge sentiment & market data
def merge_data(df_sent, df_mkt):

    # Verify required merge columns exist
    for col in ["ticker_symbol", "date"]:
        if col not in df_sent.columns:
            raise KeyError(f"Missing column in sentiment file: {col}")
        if col not in df_mkt.columns:
            raise KeyError(f"Missing column in market data: {col}")

    # Merge on ticker + date
    merged = pd.merge(
        df_mkt,     # OHLCV first
        df_sent,    # sentiment next
        on=["ticker_symbol", "date"],
        how="inner" # keep only matching rows
    )

    return merged


# Main execution
def main():
    print("Loading data...")
    df_sent, df_mkt = load_data()

    print("Merging datasets...")
    merged = merge_data(df_sent, df_mkt)

    # Output path
    output_path = (
        Path(__file__).resolve().parents[1]
        / "data" / "processed" / "merged_sentiment_market.csv"
    )

    # Save merged file
    merged.to_csv(output_path, index=False)

    print("\nMerge complete!")
    print(f"Saved to: {output_path}")
    print(f"Final shape: {merged.shape}")


if __name__ == "__main__":
    main()
