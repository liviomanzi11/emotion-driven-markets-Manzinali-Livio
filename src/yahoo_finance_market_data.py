import pandas as pd
import yfinance as yf
from pathlib import Path


# 1. Path configuration (portable)
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]     
RAW = BASE / "data" / "raw"                    
PROC = BASE / "data" / "processed"       
PROC.mkdir(parents=True, exist_ok=True)

# Input file
MAP_CSV = RAW / "Company_Tweet.csv"

# Output file
OUT_MARKET = PROC / "market_data.csv"

# Input file
MAP_CSV = RAW / "Company_Tweet.csv"

# Output file
OUT_MARKET = PROC / "market_data.csv"


# 2. Load unique tickers
def load_tickers():
    df = pd.read_csv(MAP_CSV)
    tickers = (
        df["ticker_symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return tickers


# 3. Download and standardize Yahoo Finance data
def download_market_data(tickers, start="2015-01-01", end="2020-12-31"):
    frames = []

    for ticker in tickers:
        try:
            # Download OHLCV data (don't use group_by for single ticker)
            data = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False  # Fix FutureWarning
            )

            if data.empty:
                print(f"! No data for {ticker}")
                continue

            # Handle single ticker download - yfinance returns MultiIndex columns
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                # Get the first level (Price names like 'Open', 'Close', etc.)
                data.columns = data.columns.get_level_values(0)
            
            # Normalize column names to lowercase
            data.columns = [str(c).lower() for c in data.columns]
            
            # Reset index to make date a column
            data = data.reset_index()
            
            clean_df = pd.DataFrame()
            # yfinance uses 'Date' (capital D) as the index name
            clean_df["date"] = data["Date"] if "Date" in data.columns else (data["date"] if "date" in data.columns else data.index)
            clean_df["open"] = data["open"] if "open" in data.columns else pd.Series()
            clean_df["high"] = data["high"] if "high" in data.columns else pd.Series()
            clean_df["low"] = data["low"] if "low" in data.columns else pd.Series()
            clean_df["close"] = data["close"] if "close" in data.columns else pd.Series()
            clean_df["volume"] = data["volume"] if "volume" in data.columns else pd.Series()
            clean_df["ticker_symbol"] = ticker
            
            # Drop rows where all price values are NaN
            clean_df = clean_df.dropna(how='all', subset=['open', 'high', 'low', 'close', 'volume'])

            frames.append(clean_df)
            print(f"✔ Downloaded {len(clean_df)} rows for {ticker}")

        except Exception as e:
            print(f"X Error downloading {ticker}: {e}")

    if len(frames) == 0:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# 4. Final formatting
def clean_market_data(df):
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# 5. Main pipeline
def main():
    tickers = load_tickers()
    print(f"Tickers found: {tickers}")

    market_df = download_market_data(tickers)

    if market_df.empty:
        print("No market data downloaded.")
        return

    market_df = clean_market_data(market_df)

    market_df.to_csv(OUT_MARKET, index=False)
    print("\nMarket data saved successfully.")
    print("Saving to:", OUT_MARKET)
    print(market_df.head())



# Entry point
if __name__ == "__main__":
    main()
