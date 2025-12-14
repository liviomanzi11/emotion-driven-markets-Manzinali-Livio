import yfinance as yf
import pandas as pd
from pathlib import Path

# Path Configuration
BASE = Path(__file__).resolve().parents[2]
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# Define tickers and date range (matching NASDAQ global data)
TICKERS = ["AAPL", "GOOG", "GOOGL", "AMZN", "TSLA", "MSFT"]
START_DATE = "2015-01-01"
END_DATE = "2019-12-31"  # Same as NASDAQ data

# Output file for company stock data
OUT_FILE = PROC / "company_stock_data.csv"


def download_company_data():
    """Download historical stock data for multiple companies from Yahoo Finance."""
    all_data = []
    
    for ticker in TICKERS:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data = data.reset_index()
        data.columns = data.columns.str.lower().str.strip()
        data["ticker_symbol"] = ticker
        all_data.append(data)
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def main():
    """Download and save company stock data."""
    df = download_company_data()
    df.to_csv(OUT_FILE, index=False)
    
    print(f"\nCompany stock data downloaded successfully:\n")
    print(f"- Tickers: {', '.join(TICKERS)}")
    print(f"- Rows: {len(df)}")
    print(f"- Saved to: {OUT_FILE.name}\n")


if __name__ == "__main__":
    main()
