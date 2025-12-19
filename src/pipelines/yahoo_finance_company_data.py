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


def download_company_data(apply_rounding=False):
    """Download historical stock data for multiple companies from Yahoo Finance.
    
    Args:
        apply_rounding: If True, round to 2 decimals for consistency.
                       If False, keep full Yahoo Finance precision (recommended for initial download).
    """
    all_data = []
    
    for ticker in TICKERS:
        # threads=False ensures deterministic download order
        # progress=False prevents tqdm from interfering
        data = yf.download(
            ticker, 
            start=START_DATE, 
            end=END_DATE, 
            auto_adjust=False,
            threads=False,
            progress=False
        )
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data = data.reset_index()
        data.columns = data.columns.str.lower().str.strip()
        data["ticker_symbol"] = ticker
        
        # Sort each ticker's data by date immediately
        data = data.sort_values("date").reset_index(drop=True)
        all_data.append(data)
    
    combined = pd.concat(all_data, ignore_index=True)
    # Sort by ticker and date to ensure deterministic ordering
    combined = combined.sort_values(["ticker_symbol", "date"]).reset_index(drop=True)
    
    # Apply rounding ONLY if requested (for fallback downloads)
    if apply_rounding:
        numeric_cols = ["open", "high", "low", "close", "adj close", "volume"]
        for col in numeric_cols:
            if col in combined.columns:
                if col == "volume":
                    combined[col] = combined[col].round(0).astype(int)
                else:
                    combined[col] = combined[col].round(2)  # 2 decimals = cent precision
    else:
        # Just ensure volume is integer
        if "volume" in combined.columns:
            combined["volume"] = combined["volume"].round(0).astype(int)
    
    return combined


def main():
    """
    Download and save company stock data.
    
    REPRODUCIBILITY STRATEGY:
    1. First run: Downloads with FULL PRECISION (no rounding)
       → Save this file to Google Drive
       → Ensures 100% reproducible results across different environments
    
    2. Subsequent runs: 
       - If file exists → Use cached version (100% reproducible)
       - If missing → Download with round(2) for acceptable consistency
    """
    if OUT_FILE.exists():
        print(f"\n[OK] Detected cached company_stock_data.csv (100% reproducible)")
        print(f"     File: data\\processed\\company_stock_data.csv")
        print(f"     To re-download, delete this file first.\n")
        df = pd.read_csv(OUT_FILE)
        print(f"- Tickers: {', '.join(TICKERS)}")
        print(f"- Rows: {len(df)}")
        print(f"- Date range: {df['date'].min()} to {df['date'].max()}\n")
        return
    
    # First-time download: Use full precision
    print("\n[DOWNLOAD] First-time download from Yahoo Finance (full precision)...")
    print("           This file will be used for all future runs.")
    print("           Upload to Google Drive and share with professor!\n")
    
    df = download_company_data(apply_rounding=False)  # NO rounding on first download
    df.to_csv(OUT_FILE, index=False)
    
    print(f"\nCompany stock data downloaded successfully:\n")
    print(f"- Tickers: {', '.join(TICKERS)}")
    print(f"- Rows: {len(df)}")
    print(f"- Saved to: {OUT_FILE.name}")
    print(f"\n[IMPORTANT] This file ensures 100% reproducibility!")
    print(f"            Keep it safe and share with professor.\n")


if __name__ == "__main__":
    main()
