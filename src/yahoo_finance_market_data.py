import yfinance as yf
import pandas as pd
from pathlib import Path

# Path Configuration
# Get base project directory (one level up from script location)
BASE = Path(__file__).resolve().parents[1]
# Define output path for processed data
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# Define ticker and date range for NASDAQ Composite Index (^IXIC)
TICKER = "^IXIC"
START_DATE = "2015-01-01"
END_DATE = "2019-12-31"

# Output file for NASDAQ market data
OUT_FILE = PROC / "nasdaq_market_data.csv"

# Download market data from Yahoo Finance
def download_nasdaq():
    # Fetch OHLCV data for NASDAQ within the chosen date range
    data = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=False)
    
    # Handle MultiIndex columns (yfinance returns MultiIndex for single ticker)
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten MultiIndex by taking the first level (price names like 'Open', 'Close', etc.)
        data.columns = data.columns.get_level_values(0)
    
    # Reset index to convert date index into a column
    data = data.reset_index()

    # Standardize column names (now safe since columns are not MultiIndex)
    data.columns = data.columns.str.lower().str.strip()

    # Add ticker column for consistency
    data["ticker_symbol"] = TICKER

    return data

# Main Execution Pipeline
def main():
    # Download NASDAQ market data
    df = download_nasdaq()

    # Save NASDAQ OHLCV dataset
    df.to_csv(OUT_FILE, index=False)

    # Print success information
    print("\nNASDAQ market data downloaded successfully:\n")
    print(f"- Ticker: {TICKER}")
    print(f"- Rows: {len(df)}")
    print(f"- Saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()

