"""
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
import argparse

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

    print("[OK] Folder structure verified.\n")


def print_summary():
    """
    Print final summary of backtest results and output locations.
    """
    print("\n" + "="*60)
    print("   COMPANY PIPELINE - RESULTS SUMMARY")
    print("="*60 + "\n")
    
    BASE = Path(__file__).resolve().parent
    
    # Check backtest results
    backtest_file = BASE / "results" / "figures_company" / "strategy" / "backtest_company_results.txt"
    
    if backtest_file.exists():
        print("BACKTEST RESULTS (Average across 6 Companies):\n")
        
        with open(backtest_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract overall comparison section
        if "OVERALL COMPARISON" in content:
            lines = content.split("OVERALL COMPARISON")[1].split("\n")
            for line in lines[:6]:  # First 6 lines after comparison header
                if "Average Return" in line or "Average Sharpe" in line or "Average Drawdown" in line or "Winner" in line:
                    print("  " + line.strip())
    
    print("\n" + "="*60)
    print("   PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")
    
    print("Output locations:")
    print("   • Models: data/processed/models/classical_company/ and lstm_company/")
    print("   • Figures: results/figures_company/")
    print("   • Backtest: results/figures_company/strategy/backtest_company_results.txt\n")


# INTERACTIVE MENU
def show_menu():
    """Display interactive menu for pipeline execution."""
    print("\n" + "="*60)
    print("   EMOTION-DRIVEN MARKETS")
    print("   COMPANY-LEVEL PIPELINE")
    print("="*60)
    print("\nPipeline Steps:")
    print("  [1] Load stock data (Yahoo Finance)")
    print("  [2] Sentiment analysis (FinBERT)")
    print("  [3] Merge stock + sentiment data")
    print("  [4] Feature engineering (15 technical + 15 sentiment)")
    print("  [5] Train classical ML models (LR, RF, XGBoost)")
    print("  [6] Train LSTM models")
    print("  [7] Backtest strategies")
    print("  [8] Generate visualizations")
    print("="*60)
    print("\nExecution modes:")
    print("  [1] Run full pipeline (all 8 steps)")
    print("  [2] Run step-by-step (interactive)")
    print("  [3] Run specific steps (select multiple)")
    print("  [0] Exit")
    print("="*60)
    
    choice = input("\nEnter your choice [0-3]: ").strip()
    return choice


def run_step(step_num, step_name, step_func):
    """Execute a single pipeline step with error handling."""
    print(f"\n[STEP {step_num}/8] {step_name}")
    try:
        step_func()
        print(f"  [OK] Step {step_num} complete\n")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Step {step_num} failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_interactive():
    """Run pipeline step-by-step with user confirmation."""
    steps = [
        (1, "LOAD COMPANY STOCK DATA", lambda: (print("  Loading historical stock prices..."), run_company_download())),
        (2, "SENTIMENT ANALYSIS (FinBERT)", lambda: (print("  Processing company-specific tweets..."), run_company_sentiment())),
        (3, "MERGE STOCK DATA + SENTIMENT", lambda: (print("  Merging sentiment scores with stock prices..."), run_company_merge())),
        (4, "FEATURE ENGINEERING", lambda: (print("  Computing technical and sentiment indicators..."), run_company_features())),
        (5, "CLASSICAL ML MODELS", lambda: (print("  Training 36 models (LR, RF, XGBoost)..."), run_company_classical())),
        (6, "LSTM DEEP LEARNING MODELS", lambda: (print("  Training 12 LSTM models..."), run_company_lstm())),
        (7, "TRADING STRATEGY BACKTEST", lambda: (print("  Running backtest..."), run_company_backtest())),
        (8, "GENERATE VISUALIZATIONS", lambda: (print("  Creating charts..."), run_figures_company()))
    ]
    
    for step_num, step_name, step_func in steps:
        print(f"\n{'='*60}")
        print(f"Ready to run: [STEP {step_num}/8] {step_name}")
        print(f"{'='*60}")
        response = input("Press ENTER to continue, 's' to skip, 'q' to quit: ").strip().lower()
        
        if response == 'q':
            print("\n[!] Pipeline stopped by user.\n")
            return
        elif response == 's':
            print(f"  [SKIPPED] Step {step_num}\n")
            continue
        
        if not run_step(step_num, step_name, step_func):
            retry = input("\nStep failed. Retry? (y/n): ").strip().lower()
            if retry == 'y':
                run_step(step_num, step_name, step_func)
            else:
                print("\n[!] Pipeline stopped due to error.\n")
                return
    
    print_summary()


def run_selective():
    """Allow user to select specific steps to run."""
    print("\nAvailable steps:")
    print("  [1] Load stock data")
    print("  [2] Sentiment analysis")
    print("  [3] Merge data")
    print("  [4] Feature engineering")
    print("  [5] Classical ML models")
    print("  [6] LSTM models")
    print("  [7] Backtest")
    print("  [8] Visualizations")
    
    selection = input("\nEnter step numbers (comma-separated, e.g., 1,2,3): ").strip()
    
    try:
        steps_to_run = [int(x.strip()) for x in selection.split(',')]
    except ValueError:
        print("[ERROR] Invalid input. Please enter numbers separated by commas.")
        return
    
    step_map = {
        1: (1, "LOAD STOCK DATA", lambda: run_company_download()),
        2: (2, "SENTIMENT ANALYSIS", lambda: run_company_sentiment()),
        3: (3, "MERGE DATA", lambda: run_company_merge()),
        4: (4, "FEATURE ENGINEERING", lambda: run_company_features()),
        5: (5, "CLASSICAL ML", lambda: run_company_classical()),
        6: (6, "LSTM MODELS", lambda: run_company_lstm()),
        7: (7, "BACKTEST", lambda: run_company_backtest()),
        8: (8, "VISUALIZATIONS", lambda: run_figures_company())
    }
    
    for step in steps_to_run:
        if step in step_map:
            num, name, func = step_map[step]
            run_step(num, name, func)
        else:
            print(f"[WARNING] Step {step} not recognized, skipping.")
    
    if 8 in steps_to_run:
        print_summary()


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
    
    # Clear Python cache for reproducibility
    print("\nClearing Python cache...")
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
    print("  [OK] Cache cleared\n")

    ensure_folder_structure()

    # Check for command-line arguments first
    parser = argparse.ArgumentParser(description='Run emotion-driven markets analysis')
    parser.add_argument('--classical-only', action='store_true', 
                        help='Run only classical ML models (steps 1-5)')
    parser.add_argument('--lstm-only', action='store_true',
                        help='Run only LSTM models (steps 1-4,6)')
    parser.add_argument('--auto', action='store_true',
                        help='Run full pipeline without menu (default behavior)')
    args = parser.parse_args()

    # If command-line args provided, use old behavior
    if args.classical_only or args.lstm_only or args.auto:
        run_classical = args.classical_only or not args.lstm_only
        run_lstm = args.lstm_only or not args.classical_only
        run_full_pipeline = not args.classical_only and not args.lstm_only
        
        print("\n" + "="*60)
        print("   EMOTION-DRIVEN MARKETS")
        print("   COMPANY-LEVEL PIPELINE EXECUTION")
        print("   (6 Stocks: AAPL, GOOG, GOOGL, AMZN, TSLA, MSFT)")
        print("="*60 + "\n")

        print("\n" + "="*60)
        print("   EMOTION-DRIVEN MARKETS")
        print("   COMPANY-LEVEL PIPELINE EXECUTION")
        print("   (6 Stocks: AAPL, GOOG, GOOGL, AMZN, TSLA, MSFT)")
        print("="*60 + "\n")

        try:
            # Step 1
            print("[STEP 1/8] LOAD COMPANY STOCK DATA")
            print("  Loading historical stock prices (AAPL, AMZN, GOOG, GOOGL, MSFT, TSLA)...")
            print("  → Using cached file if available (ensures 100% reproducibility)")
            run_company_download()
            print("  [OK] Step 1 complete\n")

            # Step 2
            print("[STEP 2/8] SENTIMENT ANALYSIS (FinBERT)")
            print("  Processing company-specific tweets...")
            run_company_sentiment()
            print("  [OK] Step 2 complete\n")

            # Step 3
            print("[STEP 3/8] MERGE STOCK DATA + SENTIMENT")
            print("  Merging sentiment scores with stock prices...")
            run_company_merge()
            print("  [OK] Step 3 complete\n")

            # Step 4
            print("[STEP 4/8] FEATURE ENGINEERING")
            print("  Computing technical and sentiment indicators...")
            run_company_features()
            print("  [OK] Step 4 complete\n")

            # Step 5 - Classical models
            if run_classical:
                print("[STEP 5/8] CLASSICAL ML MODELS (Logistic, RF, XGBoost)")
                print("  Training 36 models (6 companies × 3 models × 2 feature sets)...")
                run_company_classical()
                print("  [OK] Step 5 complete\n")
            else:
                print("[STEP 5/8] CLASSICAL ML MODELS - SKIPPED (--lstm-only)\n")

            # Step 6 - LSTM models
            if run_lstm:
                print("[STEP 6/8] LSTM DEEP LEARNING MODELS")
                print("  Training 12 LSTM models (6 companies × 2 feature sets)...")
                run_company_lstm()
                print("  [OK] Step 6 complete\n")
            else:
                print("[STEP 6/8] LSTM MODELS - SKIPPED (--classical-only)\n")

            # Step 7 - Backtest (only if full pipeline)
            if run_full_pipeline:
                print("[STEP 7/8] TRADING STRATEGY BACKTEST")
                print("  Running backtest for all companies (RF + LSTM strategies)...")
                run_company_backtest()
                print("  [OK] Step 7 complete\n")
            else:
                print("[STEP 7/8] TRADING STRATEGY BACKTEST - SKIPPED (partial run)\n")

            # Step 8 - Figures (only if full pipeline)
            if run_full_pipeline:
                print("[STEP 8/8] GENERATE VISUALIZATION FIGURES")
                print("  Creating comparative charts and analytics...")
                run_figures_company()
                print("  [OK] Step 8 complete\n")
            else:
                print("[STEP 8/8] GENERATE VISUALIZATION FIGURES - SKIPPED (partial run)\n")

            # Print final summary (only if full pipeline)
            if run_full_pipeline:
                print_summary()

        except KeyboardInterrupt:
            print("\n\n[!] Pipeline interrupted by user (Ctrl+C)\n")
            sys.exit(0)

        except Exception as e:
            print("\n" + "="*60)
            print("   [ERROR] PIPELINE ERROR")
            print("="*60 + "\n")
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        return

    # Interactive menu mode (default when no args)
    try:
        while True:
            choice = show_menu()
            
            if choice == '0':
                print("\n[!] Exiting pipeline.\n")
                sys.exit(0)
            
            elif choice == '1':
                # Run full pipeline automatically
                print("\n" + "="*60)
                print("   RUNNING FULL PIPELINE")
                print("="*60 + "\n")
                
                steps = [
                    (1, "LOAD STOCK DATA", run_company_download),
                    (2, "SENTIMENT ANALYSIS", run_company_sentiment),
                    (3, "MERGE DATA", run_company_merge),
                    (4, "FEATURE ENGINEERING", run_company_features),
                    (5, "CLASSICAL ML", run_company_classical),
                    (6, "LSTM MODELS", run_company_lstm),
                    (7, "BACKTEST", run_company_backtest),
                    (8, "VISUALIZATIONS", run_figures_company)
                ]
                
                for num, name, func in steps:
                    if not run_step(num, name, func):
                        print("\n[!] Pipeline stopped due to error.\n")
                        break
                else:
                    print_summary()
                
                break
            
            elif choice == '2':
                # Step-by-step interactive
                run_interactive()
                break
            
            elif choice == '3':
                # Selective steps
                run_selective()
                break
            
            else:
                print("\n[ERROR] Invalid choice. Please enter 0, 1, 2, or 3.\n")
    
    except KeyboardInterrupt:
        print("\n\n[!] Pipeline interrupted by user (Ctrl+C)\n")
        sys.exit(0)

    except Exception as e:
        print("\n" + "="*60)
        print("   [ERROR] PIPELINE ERROR")
        print("="*60 + "\n")
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
