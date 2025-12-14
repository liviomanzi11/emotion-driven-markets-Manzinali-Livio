"""
Backtesting module for ML and sentiment-based trading strategies.

This module runs backtests on 9 different strategies for 6 NASDAQ companies:
- Logistic Regression (Technical & Sentiment)
- Random Forest (Technical & Sentiment)
- XGBoost (Technical & Sentiment)
- LSTM (Technical & Sentiment)
- Buy & Hold baseline

Each strategy generates equity curves and performance metrics that are used
for comparative analysis in the visualization pipeline.
"""

import warnings
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = BASE_DIR / "data" / "processed" / "features_company"
MODELS_DIR = BASE_DIR / "data" / "processed" / "models"
RESULTS_DIR = BASE_DIR / "results" / "figures_company" / "strategy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EQUITY_DIR = BASE_DIR / "results" / "equity_curves"
EQUITY_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "GOOG", "GOOGL", "AMZN", "TSLA", "MSFT"]
SEQ_LENGTH = 45  # Number of days for LSTM sequence window


def compute_strategy(probas, actual_returns, dates):
    """
    Convert ML probabilities into a simple trading strategy.
    
    Strategy logic: Go long when model predicts probability > 0.5,
    otherwise stay in cash (0% allocation). This is an academic
    approach to test predictive power without leverage or shorting.
    
    Args:
        probas: Model prediction probabilities (0 to 1)
        actual_returns: Actual daily returns from market data
        dates: Trading dates for time-series alignment
        
    Returns:
        Dictionary with equity curve, metrics (return, sharpe, drawdown)
    """
    # Binary allocation: 100% long if confident, 0% otherwise
    exposure = (probas > 0.5).astype(float)
    strat_returns = exposure * actual_returns
    equity = np.cumprod(1 + strat_returns)

    total_return = (equity[-1] - 1) * 100
    # Annualized Sharpe ratio (252 trading days)
    sharpe = (strat_returns.mean() / (strat_returns.std() + 1e-12)) * np.sqrt(252)
    # Maximum drawdown from peak
    dd = equity / np.maximum.accumulate(equity) - 1
    max_dd = dd.min() * 100

    return {
        "equity": equity,
        "returns": strat_returns,
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "dates": dates,
    }


def load_test_df(ticker, features_type):
    """
    Load test dataset and return feature columns in the EXACT order
    used during model training. This is critical to prevent silent bugs
    where sklearn applies coefficients to misaligned feature columns.
    
    """
    df = pd.read_csv(FEATURES_DIR / f"{ticker}_features_test.csv", parse_dates=["date"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Load the exact feature order saved during training (CRITICAL FIX #1)
    # This prevents the 0% return bug caused by feature misalignment
    feature_order = joblib.load(
        MODELS_DIR / "classical_company" / ticker / f"{features_type}_feature_order.pkl"
    )
    
    # Ensure all expected features exist in the dataframe
    feature_order = [c for c in feature_order if c in df.columns]
    
    return df, feature_order


def get_positive_class_proba(model, X):
    """
    Get probability for positive class (class 1) regardless of class order.
    
    CRITICAL BUG FIX: Some models have classes_ = [0, 1], others have [1, 0].
    Hardcoding [:, 1] assumes class 1 is always at index 1, which is WRONG.
    This function finds the correct index using model.classes_.
    
    This fixes the 0% return bug for AAPL LR technical and TSLA LR sentiment.
    
    Args:
        model: Trained sklearn classifier with classes_ attribute
        X: Feature matrix for prediction
        
    Returns:
        Array of probabilities for class 1 (positive/up movement)
    """
    probas = model.predict_proba(X)
    
    # Find which column corresponds to class 1
    try:
        class_1_idx = list(model.classes_).index(1)
    except ValueError:
        # If class 1 doesn't exist, default to column 1
        class_1_idx = 1
    
    return probas[:, class_1_idx]


def backtest_logistic_regression(ticker, features_type):
    """
    Backtest Logistic Regression model on out-of-sample test data.
    
    Uses the exact feature order from training to prevent coefficient
    misalignment, which would cause incorrect predictions.
    
    CRITICAL: LR requires scaled features. Must apply same scaler used in training.
    """
    print(f"  > Logistic Regression ({features_type})...", end=" ", flush=True)

    df, feature_order = load_test_df(ticker, features_type)
    # Skip first SEQ_LENGTH rows to align with LSTM predictions
    df = df.iloc[SEQ_LENGTH:].reset_index(drop=True)
    
    # Use feature_order to ensure correct alignment with trained model
    X = df[feature_order].values
    y_ret = df["return"].values
    dates = df["date"].values

    # LR trained on normalized data - must apply identical transformation for inference
    model = joblib.load(MODELS_DIR / "classical_company" / ticker / f"{features_type}_logistic_regression_model.pkl")
    scaler = joblib.load(MODELS_DIR / "classical_company" / ticker / f"{features_type}_scaler.pkl")
    
    # CRITICAL: Scale features before prediction (must match training)
    X_scaled = scaler.transform(X)
    
    # CRITICAL FIX: Use helper function to get correct probability column
    prob = get_positive_class_proba(model, X_scaled)

    result = compute_strategy(prob, y_ret, dates)
    save_equity_curve(ticker, "lr", features_type, result)

    print("[OK]")
    return result


def backtest_random_forest(ticker, features_type):
    """
    Backtest Random Forest model on out-of-sample test data.
    
    Random Forest is less sensitive to feature order than linear models,
    but we still enforce consistency for reproducibility.
    """
    print(f"  > Random Forest ({features_type})...", end=" ", flush=True)

    df, feature_order = load_test_df(ticker, features_type)
    df = df.iloc[SEQ_LENGTH:].reset_index(drop=True)
    
    X = df[feature_order].values
    y_ret = df["return"].values
    dates = df["date"].values

    model = joblib.load(MODELS_DIR / "classical_company" / ticker / f"{features_type}_random_forest_model.pkl")
    
    # CRITICAL FIX: Use helper function to get correct probability column
    prob = get_positive_class_proba(model, X)

    result = compute_strategy(prob, y_ret, dates)
    save_equity_curve(ticker, "rf", features_type, result)

    print("[OK]")
    return result


def backtest_xgboost(ticker, features_type):
    """
    Backtest XGBoost model on out-of-sample test data.
    
    XGBoost uses feature names internally, so order matters for
    consistency with the training phase.
    """
    print(f"  > XGBoost ({features_type})...", end=" ", flush=True)

    df, feature_order = load_test_df(ticker, features_type)
    df = df.iloc[SEQ_LENGTH:].reset_index(drop=True)
    
    X = df[feature_order].values
    y_ret = df["return"].values
    dates = df["date"].values

    model = joblib.load(MODELS_DIR / "classical_company" / ticker / f"{features_type}_xgboost_model.pkl")
    
    # CRITICAL FIX: Use helper function to get correct probability column
    prob = get_positive_class_proba(model, X)

    result = compute_strategy(prob, y_ret, dates)
    save_equity_curve(ticker, "xgb", features_type, result)

    print("[OK]")
    return result


def backtest_lstm(ticker, features_type):
    """
    Backtest LSTM model on out-of-sample test data.
    
    LSTM requires sequences of length SEQ_LENGTH, so we create rolling
    windows of historical data for each prediction point.
    """
    print(f"  > LSTM ({features_type})...", end=" ", flush=True)

    df, feature_order = load_test_df(ticker, features_type)

    X_all = df[feature_order].values
    y_all = df["return"].values
    dates_all = df["date"].values

    # LSTM models use standardized features (mean=0, std=1)
    scaler = joblib.load(MODELS_DIR / "lstm_company" / ticker / f"lstm_scaler_{features_type}.pkl")
    X_scaled = scaler.transform(X_all)

    # Create sequences for LSTM (each prediction uses past SEQ_LENGTH days)
    X_seq, y_seq, d_seq = [], [], []
    for i in range(len(X_scaled) - SEQ_LENGTH):
        X_seq.append(X_scaled[i:i + SEQ_LENGTH])
        y_seq.append(y_all[i + SEQ_LENGTH])
        d_seq.append(dates_all[i + SEQ_LENGTH])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    d_seq = np.array(d_seq)

    model = load_model(MODELS_DIR / "lstm_company" / ticker / f"lstm_model_{features_type}.h5")
    prob = model.predict(X_seq, verbose=0).flatten()

    result = compute_strategy(prob, y_seq, d_seq)
    save_equity_curve(ticker, "lstm", features_type, result)

    print("[OK]")
    return result


def backtest_buyhold(ticker):
    """
    Backtest buy-and-hold baseline strategy.
    
    This is the benchmark: buy at the start and hold until the end.
    No rebalancing, no timing, just passive exposure to market returns.
    """
    print(f"  > Buy & Hold...", end=" ", flush=True)

    df, _ = load_test_df(ticker, "technical")
    # Align with other strategies (skip SEQ_LENGTH rows)
    df = df.iloc[SEQ_LENGTH:].reset_index(drop=True)

    ret = df["return"].values
    dates = df["date"].values
    eq = np.cumprod(1 + ret)

    dd = eq / np.maximum.accumulate(eq) - 1
    result = {
        "equity": eq,
        "returns": ret,
        "total_return": (eq[-1] - 1) * 100,
        "sharpe_ratio": (ret.mean() / (ret.std() + 1e-12)) * np.sqrt(252),
        "max_drawdown": dd.min() * 100,
        "dates": dates,
    }

    save_equity_curve(ticker, "buyhold", "na", result)

    print("[OK]")
    return result


def save_equity_curve(ticker, model_tag, features, result):
    """
    Save daily equity curve to CSV for visualization pipeline.
    
    These files are read by figures_company.py to generate comparative
    charts showing strategy performance over time.
    """
    df = pd.DataFrame({
        "date": pd.to_datetime(result["dates"]),
        "equity": result["equity"]
    })
    df.to_csv(EQUITY_DIR / f"{ticker}_{model_tag}_{features}.csv", index=False)


def run_backtest():
    """
    Execute backtest pipeline for all companies and all strategies.
    
    Runs 9 strategies per company (6 companies total = 54 backtests):
    - 3 classical models x 2 feature sets = 6
    - 1 LSTM model x 2 feature sets = 2
    - 1 buy-and-hold baseline = 1
    """
    print("\n=== BACKTESTING ML + SENTIMENT STRATEGIES ===\n")

    results = {}

    for ticker in TICKERS:
        print(f"[{ticker}]")
        results[ticker] = {
            "lr_technical": backtest_logistic_regression(ticker, "technical"),
            "lr_sentiment": backtest_logistic_regression(ticker, "sentiment"),
            "rf_technical": backtest_random_forest(ticker, "technical"),
            "rf_sentiment": backtest_random_forest(ticker, "sentiment"),
            "xgb_technical": backtest_xgboost(ticker, "technical"),
            "xgb_sentiment": backtest_xgboost(ticker, "sentiment"),
            "lstm_technical": backtest_lstm(ticker, "technical"),
            "lstm_sentiment": backtest_lstm(ticker, "sentiment"),
            "buyhold": backtest_buyhold(ticker)
        }
        print()

    save_results(results)
    print("[OK] Backtest complete.\n")


def save_results(results):
    """
    Save backtest results to text file with proper formatting.
    
    CRITICAL: Uses double newlines (\n\n) after each strategy block to ensure
    proper parsing by figures_company.py. Missing newlines would break the parser.
    
    Includes Sharpe ratio as required by visualization pipeline.
    """
    out = RESULTS_DIR / "backtest_company_results.txt"
    with open(out, "w", encoding="utf-8") as f:

        f.write("BACKTEST SUMMARY\n")
        f.write("="*80 + "\n\n")

        for ticker in TICKERS:
            f.write(f"{ticker}\n\n")  # Double newline for parser compatibility

            # Strategy name mapping for readable output
            mapping = {
                "lr_technical": "Logistic Regression (Technical)",
                "lr_sentiment": "Logistic Regression (Sentiment)",
                "rf_technical": "Random Forest (Technical)",
                "rf_sentiment": "Random Forest (Sentiment)",
                "xgb_technical": "XGBoost (Technical)",
                "xgb_sentiment": "XGBoost (Sentiment)",
                "lstm_technical": "LSTM (Technical)",
                "lstm_sentiment": "LSTM (Sentiment)",
                "buyhold": "Buy & Hold",
            }

            for key, pretty_name in mapping.items():
                res = results[ticker][key]
                f.write(f"{pretty_name}:\n")
                f.write(f"  Total Return: {res['total_return']:.2f}%\n")
                f.write(f"  Sharpe Ratio: {res['sharpe_ratio']:.4f}\n")  # FIX #3: Added Sharpe
                f.write(f"  Max Drawdown: {res['max_drawdown']:.2f}%\n\n")  # FIX #2: Double newline


if __name__ == "__main__":
    run_backtest()
