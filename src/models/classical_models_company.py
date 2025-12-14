import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shutil

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Fix random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Project root directory
BASE = Path(__file__).resolve().parents[2]
PROC = BASE / "data" / "processed"

# Directory where company features are stored
FEATURES_COMPANY = PROC / "features_company"

# Directory for storing trained models per company
CLASSICAL_MODELS_COMPANY = PROC / "models" / "classical_company"
CLASSICAL_MODELS_COMPANY.mkdir(parents=True, exist_ok=True)

# Directory for saving results
RESULTS_CLASSICAL_COMPANY = BASE / "results" / "classical_company"
RESULTS_CLASSICAL_COMPANY.mkdir(parents=True, exist_ok=True)


def clear_python_cache():
    """Remove all Python cache to ensure fresh imports."""
    cache_dirs = [
        BASE / "src" / "__pycache__",
        BASE / "src" / "models" / "__pycache__",
        BASE / "src" / "strategies" / "__pycache__",
        BASE / "src" / "features" / "__pycache__"
    ]
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

# List of companies
TICKERS = ["AAPL", "GOOG", "GOOGL", "AMZN", "TSLA", "MSFT"]


def load_dataset(path):
    """Load CSV dataset from the specified path."""
    return pd.read_csv(path)


def prepare_features(df, features_type):
    """
    Select features (X) and target (y) based on the specified feature type.
    
    Args:
        df: DataFrame with all features
        features_type: "technical" or "sentiment"
    
    Returns:
        X, y: Feature matrix and target vector
    """
    # Balanced feature sets (15 features each for fair comparison)
    if features_type == "technical":
        features = [
            "return", "volume", "log_return",
            "ma5", "ma10", "ma20",
            "volatility_5d", "volatility_10d", "volatility_20d",
            "rsi_14", "macd_line", "signal_line",
            "adj close", "close", "open"
        ]

    elif features_type == "sentiment":
        features = [
            "polarity",
            "positive",
            "negative",
            "impact_weighted_sentiment",
            "influence_weighted_sentiment",
            "impact_weighted_ma3",
            "impact_weighted_ma7",
            "delta_polarity",
            "delta_impact_weighted_sentiment",
            "sentiment_volatility_5d",
            "extreme_count_5d",
            "tweet_volume",
            "tweet_volume_ma5",
            "engagement_ma5",
            "engagement_change"
        ]

    else:
        raise ValueError(f"Unknown features_type: {features_type}")

    # Select only specified features and target
    X = df[features]
    y = df["target"]

    return X, y


def train_models(ticker, features_type):
    """
    Train classical models for a specific company and feature type.
    """
    # Feature engineering pipeline produces separate train/test files to prevent data leakage
    train_file = FEATURES_COMPANY / f"{ticker}_features_train.csv"
    df_train = load_dataset(train_file)

    # Prepare features
    X_train, y_train = prepare_features(df_train, features_type)
    
    # Financial data can contain extreme values or divisions by zero from technical indicators
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)

    # CRITICAL: Scale features for Logistic Regression
    # LR needs normalized features or regularization will kill coefficients
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Markets have unequal up/down distributions - balanced weights prevent model bias toward majority class
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # Train Logistic Regression (on SCALED data)
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight_dict,
        random_state=42,
        solver="lbfgs"
    )
    log_reg.fit(X_train_scaled, y_train)

    # Train Random Forest (on UNSCALED data - tree models don't need scaling)
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Train XGBoost (on UNSCALED data)
    scale_pos_weight = class_weights[1] / class_weights[0]
    xgb = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)

    # Save models and feature order (critical for reproducibility in backtesting)
    model_dir = CLASSICAL_MODELS_COMPANY / ticker
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(log_reg, model_dir / f"{features_type}_logistic_regression_model.pkl")
    joblib.dump(scaler, model_dir / f"{features_type}_scaler.pkl")  # Save scaler for LR
    joblib.dump(rf, model_dir / f"{features_type}_random_forest_model.pkl")
    joblib.dump(xgb, model_dir / f"{features_type}_xgboost_model.pkl")
    
    # Sklearn applies coefficients positionally - misaligned features cause silent prediction errors
    feature_order = list(X_train.columns)
    joblib.dump(feature_order, model_dir / f"{features_type}_feature_order.pkl")


def test_models(ticker, features_type):
    """
    Evaluate trained models on test data for a specific company.
    """
    # Test set spans different time period (2019) to validate out-of-sample performance
    test_file = FEATURES_COMPANY / f"{ticker}_features_test.csv"
    df_test = load_dataset(test_file)

    # Prepare features
    X_test, y_test = prepare_features(df_test, features_type)
    
    # Test data must be cleaned with same logic as training for fair comparison
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Trained models must be loaded from disk for out-of-sample evaluation
    model_dir = CLASSICAL_MODELS_COMPANY / ticker
    log_reg = joblib.load(model_dir / f"{features_type}_logistic_regression_model.pkl")
    rf = joblib.load(model_dir / f"{features_type}_random_forest_model.pkl")
    xgb = joblib.load(model_dir / f"{features_type}_xgboost_model.pkl")

    # Evaluate models
    results = []
    
    for name, model in [("Logistic Regression", log_reg), ("Random Forest", rf), ("XGBoost", xgb)]:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        
        results.append({
            "model": name,
            "accuracy": acc,
            "f1": f1,
            "confusion_matrix": cm
        })

    # Save results
    results_dir = RESULTS_CLASSICAL_COMPANY / ticker
    results_dir.mkdir(exist_ok=True, parents=True)
    
    out_file = results_dir / f"classical_results_{features_type}.txt"
    
    with open(out_file, "w") as f:
        f.write(f"=== {ticker} - Classical Models ({features_type}) ===\n\n")
        for res in results:
            f.write(f"{res['model']}:\n")
            f.write(f"  Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"  F1-score: {res['f1']:.4f}\n")
            f.write(f"  Confusion Matrix:\n{res['confusion_matrix']}\n\n")


def run_all():
    """
    Run complete classical models pipeline for all companies.
    Train and test on technical and sentiment features only.
    """
    # Clear Python cache for reproducibility
    clear_python_cache()
    
    print("\n=== Running Classical Models Pipeline (Company) ===\n")
    
    for ticker in TICKERS:
        print(f"→ {ticker}")
        
        # Train models
        print(f"  Training (technical)...", end=" ", flush=True)
        train_models(ticker, "technical")
        print("✓")
        
        print(f"  Training (sentiment)...", end=" ", flush=True)
        train_models(ticker, "sentiment")
        print("✓")
        
        # Test models
        print(f"  Testing (technical)...", end=" ", flush=True)
        test_models(ticker, "technical")
        print("✓")
        
        print(f"  Testing (sentiment)...", end=" ", flush=True)
        test_models(ticker, "sentiment")
        print("✓")
    
    print(f"\n✓ Classical models pipeline complete.\n")


if __name__ == "__main__":
    run_all()
