"""
main.py
=======
Complete pipeline for Company-Level Analysis (6 stocks: AAPL, GOOG, GOOGL, AMZN, TSLA, MSFT).
Compares technical-only vs sentiment-only ML models for stock return prediction.

Usage:
    python main.py

Outputs:
    - Trained models: data/processed/models/classical_company/ and lstm_company/
    - Results: results/classical_company/, lstm_company/, equity_curves/
    - Figures: results/figures_company/
"""

from pathlib import Path
import warnings
import os
import logging
import sys
import shutil

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Silence TensorFlow verbose messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

# Suppress ABSL warnings (TensorFlow/Keras)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# IMPORT PIPELINES
from src.pipelines.yahoo_finance_company_data import main as run_company_download
from src.pipelines.sentiment_pipeline_company import main as run_company_sentiment
from src.pipelines.merge_sentiment_company_daily import main as run_company_merge
from src.pipelines.feature_engineering_company import main as run_company_features

# IMPORT MODELS
from src.models.classical_models_company import run_all as run_company_classical
from src.models.lstm_models_company import run_all as run_company_lstm

# IMPORT VISUALIZATION
from src.visualization.figures_company import run_figures_company

# IMPORT STRATEGY BACKTEST
from src.strategies.backtest_company import run_backtest as run_company_backtest


# CREATE ALL NECESSARY FOLDERS SAFELY
def ensure_folder_structure():
    """
    Create all required directories for data processing.
    Prevents FileNotFoundError during pipeline execution.
    """
    BASE = Path(__file__).resolve().parent

    # Processed data requires separate folders for features and models per company
    (BASE / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (BASE / "data" / "processed" / "samples").mkdir(parents=True, exist_ok=True)
    (BASE / "data" / "processed" / "features_company").mkdir(parents=True, exist_ok=True)
    (BASE / "data" / "processed" / "models" / "classical_company").mkdir(parents=True, exist_ok=True)
    (BASE / "data" / "processed" / "models" / "lstm_company").mkdir(parents=True, exist_ok=True)

    # Results organized by model type to separate classification reports from backtests
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    (BASE / "results" / "classical_company").mkdir(parents=True, exist_ok=True)
    (BASE / "results" / "lstm_company").mkdir(parents=True, exist_ok=True)
    (BASE / "results" / "equity_curves").mkdir(parents=True, exist_ok=True)
    (BASE / "results" / "figures_company").mkdir(parents=True, exist_ok=True)
    (BASE / "results" / "figures_company" / "strategy").mkdir(parents=True, exist_ok=True)

    print("✔ Folder structure verified.\n")


def print_summary():
    """
    Print final summary of backtest results and output locations.
    """
    print("\n" + "="*60)
    print("   📊 COMPANY PIPELINE - RESULTS SUMMARY")
    print("="*60 + "\n")
    
    BASE = Path(__file__).resolve().parent
    
    # Check backtest results
    backtest_file = BASE / "results" / "figures_company" / "strategy" / "backtest_company_results.txt"
    
    if backtest_file.exists():
        print("📈 BACKTEST RESULTS (Average across 6 Companies):\n")
        
        with open(backtest_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract overall comparison section
        if "OVERALL COMPARISON" in content:
            lines = content.split("OVERALL COMPARISON")[1].split("\n")
            for line in lines[:6]:  # First 6 lines after comparison header
                if "Average Return" in line or "Average Sharpe" in line or "Average Drawdown" in line or "Winner" in line:
                    print("  " + line.strip())
    
    print("\n" + "="*60)
    print("   ✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")
    
    print("📁 Output locations:")
    print("   • Models: data/processed/models/classical_company/ and lstm_company/")
    print("   • Figures: results/figures_company/")
    print("   • Backtest: results/figures_company/strategy/backtest_company_results.txt\n")


# MAIN PIPELINE EXECUTION
def main():
    """
    Execute complete pipeline to compare technical vs sentiment predictive power.
    
    Pipeline isolates each signal type (technical-only vs sentiment-only models)
    to measure independent predictive contribution rather than combining features.
    Each company gets 8 models: 4 technical + 4 sentiment (LR, RF, XGB, LSTM).
    
    Steps:
        1. Download OHLCV data from Yahoo Finance
        2. Apply FinBERT sentiment analysis to company tweets
        3. Merge sentiment scores with stock data by date
        4. Engineer 15 technical + 15 sentiment features per company
        5. Train classical models separately on technical vs sentiment
        6. Train LSTM models separately on technical vs sentiment
        7. Backtest all strategies with long-only equity curves
        8. Generate comparison visualizations
    """
    print("\n" + "="*60)
    print("   📈 EMOTION-DRIVEN MARKETS")
    print("   COMPANY-LEVEL PIPELINE EXECUTION")
    print("   (6 Stocks: AAPL, GOOG, GOOGL, AMZN, TSLA, MSFT)")
    print("="*60 + "\n")
    
    # Clear Python cache for reproducibility
    print("🧹 Clearing Python cache...")
    BASE = Path(__file__).resolve().parent
    cache_dirs = [
        BASE / "src" / "__pycache__",
        BASE / "src" / "models" / "__pycache__",
        BASE / "src" / "strategies" / "__pycache__",
        BASE / "src" / "features" / "__pycache__"
    ]
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
    print("  ✓ Cache cleared\n")

    ensure_folder_structure()

    try:
        # Step 1
        print("▶ STEP 1/8 — DOWNLOAD COMPANY STOCK DATA")
        print("  Downloading from Yahoo Finance (6 companies)...")
        run_company_download()
        print("  ✓ Step 1 complete\n")

        # Step 2
        print("▶ STEP 2/8 — SENTIMENT ANALYSIS (FinBERT)")
        print("  Processing company-specific tweets...")
        run_company_sentiment()
        print("  ✓ Step 2 complete\n")

        # Step 3
        print("▶ STEP 3/8 — MERGE STOCK DATA + SENTIMENT")
        print("  Merging sentiment scores with stock prices...")
        run_company_merge()
        print("  ✓ Step 3 complete\n")

        # Step 4
        print("▶ STEP 4/8 — FEATURE ENGINEERING")
        print("  Computing technical and sentiment indicators...")
        run_company_features()
        print("  ✓ Step 4 complete\n")

        # Step 5
        print("▶ STEP 5/8 — CLASSICAL ML MODELS (Logistic, RF, XGBoost)")
        print("  Training 36 models (6 companies × 3 models × 2 feature sets)...")
        run_company_classical()
        print("  ✓ Step 5 complete\n")

        # Step 6
        print("▶ STEP 6/8 — LSTM DEEP LEARNING MODELS")
        print("  Training 12 LSTM models (6 companies × 2 feature sets)...")
        run_company_lstm()
        print("  ✓ Step 6 complete\n")

        # Step 7
        print("▶ STEP 7/8 — TRADING STRATEGY BACKTEST")
        print("  Running backtest for all companies (RF + LSTM strategies)...")
        run_company_backtest()
        print("  ✓ Step 7 complete\n")

        # Step 8
        print("▶ STEP 8/8 — GENERATE VISUALIZATION FIGURES")
        print("  Creating comparative charts and analytics...")
        run_figures_company()
        print("  ✓ Step 8 complete\n")

        # Print final summary
        print_summary()

    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user (Ctrl+C)\n")
        sys.exit(0)

    except Exception as e:
        print("\n" + "="*60)
        print("   ❌ PIPELINE ERROR")
        print("="*60 + "\n")
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
