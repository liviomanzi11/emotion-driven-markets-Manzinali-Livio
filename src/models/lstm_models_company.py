import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os
import warnings
import logging
import shutil

# Suppress TensorFlow warnings BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Force deterministic GPU operations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU, use CPU only (100% deterministic)

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress verbose warnings
warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Random seed for reproducibility - CRITICAL FOR GRADING
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
import random
random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Force single-threaded CPU execution for 100% deterministic results
# GPU operations have non-deterministic floating-point atomics, so we disable GPU
# This ensures identical results across runs (required for academic grading)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Project paths
BASE = Path(__file__).resolve().parents[2]
PROC = BASE / "data" / "processed"
FEATURES_COMPANY = PROC / "features_company"

# Model and results directories
MODEL_DIR = PROC / "models" / "lstm_company"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_LSTM = BASE / "results" / "lstm_company"
RESULTS_LSTM.mkdir(parents=True, exist_ok=True)

# LSTM hyperparameters
SEQ_LENGTH = 45  # Use 45-day sequences
TICKERS = ["AAPL", "GOOG", "GOOGL", "AMZN", "TSLA", "MSFT"]


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


def load_dataset(path):
    """Load CSV dataset with deterministic row order."""
    df = pd.read_csv(path)
    # CRITICAL: Ensure deterministic row order - date must be sorted ascending
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    return df


def clean_df(df, features_type):
    """Select features based on type (technical or sentiment)."""
    
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

    available_features = [f for f in features if f in df.columns]
    df = df[available_features + ["target"]]
    
    # CRITICAL: Replace Inf and NaN, then SORT to ensure deterministic row order
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.sort_index().reset_index(drop=True)  # Maintain temporal order after dropna
    
    return df


def create_sequences(df):
    """
    Create LSTM input sequences from training data.
    Returns: X (sequences), y (targets), fitted scaler
    """
    # Separate features and target
    features = df.drop(columns=["target"]).values
    target = df["target"].values

    # Fit scaler on training data only to avoid leaking test set statistics into model
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # LSTM requires fixed-length sequences - each prediction uses past SEQ_LENGTH days as context
    X, y = [], []
    for i in range(len(features_scaled) - SEQ_LENGTH):
        X.append(features_scaled[i:i+SEQ_LENGTH])
        y.append(target[i+SEQ_LENGTH])

    return np.array(X), np.array(y), scaler


def create_sequences_test(df, scaler):
    """
    Create LSTM input sequences from test data using pre-fitted scaler.
    """
    # Separate features and target
    features = df.drop(columns=["target"]).values
    target = df["target"].values

    # Use training scaler to ensure test features have same distribution as training
    features_scaled = scaler.transform(features)

    # Build sequences
    X, y = [], []
    for i in range(len(features_scaled) - SEQ_LENGTH):
        X.append(features_scaled[i:i+SEQ_LENGTH])
        y.append(target[i+SEQ_LENGTH])

    return np.array(X), np.array(y)


def build_model(input_shape):
    """
    Build LSTM model architecture with fixed seeds for reproducibility.
    """
    # Fixed initializers for reproducibility
    from tensorflow.keras.initializers import GlorotUniform, Orthogonal
    
    kernel_init = GlorotUniform(seed=SEED)
    recurrent_init = Orthogonal(seed=SEED)
    
    model = Sequential([
        Bidirectional(
            LSTM(128, return_sequences=True, 
                 kernel_initializer=kernel_init,
                 recurrent_initializer=recurrent_init),
            input_shape=input_shape
        ),
        Dropout(0.2, seed=SEED),
        Bidirectional(
            LSTM(64,
                 kernel_initializer=kernel_init,
                 recurrent_initializer=recurrent_init)
        ),
        Dropout(0.2, seed=SEED),
        Dense(32, activation="relu", kernel_initializer=kernel_init),
        Dense(1, activation="sigmoid", kernel_initializer=kernel_init)
    ])

    # Lower learning rate prevents overshooting in recurrent network optimization landscape
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_lstm(ticker, features_type):
    """
    Train LSTM model for a specific company and feature type.
    """
    print(f"  Training LSTM ({features_type})...", end=" ", flush=True)

    # Load training data
    train_file = FEATURES_COMPANY / f"{ticker}_features_train.csv"
    df = load_dataset(train_file)
    df = clean_df(df, features_type)

    # Create sequences and fit scaler
    X, y, scaler = create_sequences(df)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y
    )
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    # Build model
    model = build_model((SEQ_LENGTH, X.shape[2]))

    # Custom callback to show progress every 10 epochs
    from tensorflow.keras.callbacks import Callback
    class ProgressCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                print(f"[still training... epoch {epoch+1}/50]", end=" ", flush=True)
    
    # Early stopping with more patience for larger model
    es = EarlyStopping(patience=10, restore_best_weights=True, verbose=0, monitor='loss')
    progress = ProgressCallback()

    # Train model with original hyperparameters for maximum quality
    model.fit(
        X, y,
        epochs=50,
        batch_size=16,
        callbacks=[es, progress],
        class_weight=class_weights_dict,
        shuffle=False,
        verbose=0
    )

    # Save scaler and model
    ticker_dir = MODEL_DIR / ticker
    ticker_dir.mkdir(exist_ok=True)
    
    joblib.dump(scaler, ticker_dir / f"lstm_scaler_{features_type}.pkl")
    model.save(ticker_dir / f"lstm_model_{features_type}.h5")

    print("[OK]")


def test_lstm(ticker, features_type):
    """
    Evaluate LSTM model on test data for a specific company.
    """
    print(f"  Testing LSTM ({features_type})...", end=" ", flush=True)

    # Load test data
    test_file = FEATURES_COMPANY / f"{ticker}_features_test.csv"
    df = load_dataset(test_file)
    df = clean_df(df, features_type)

    # Load scaler and model
    ticker_dir = MODEL_DIR / ticker
    scaler = joblib.load(ticker_dir / f"lstm_scaler_{features_type}.pkl")
    model = load_model(ticker_dir / f"lstm_model_{features_type}.h5")

    # Create test sequences
    X, y = create_sequences_test(df, scaler)

    # Predict
    preds = model.predict(X, verbose=0)
    preds = (preds > 0.5).astype(int).flatten()

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)

    # Save results
    results_dir = RESULTS_LSTM / ticker
    results_dir.mkdir(exist_ok=True, parents=True)
    
    out_file = results_dir / f"lstm_results_{features_type}.txt"
    
    with open(out_file, "w") as f:
        f.write(f"=== {ticker} - LSTM ({features_type}) ===\n\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")

    print("[OK]")


def run_all():
    """
    Run complete LSTM pipeline for all companies.
    Train and test on technical and sentiment features only.
    """
    print("\n=== Running LSTM Pipeline (Company) ===")
    print("[WARNING] Training with enhanced architecture (128->64 neurons, 50 epochs)")
    print("[WARNING] Estimated time: 20-30 minutes for all companies\n")

    for ticker in TICKERS:
        print(f"→ {ticker}")
        
        # Train LSTM models
        train_lstm(ticker, "technical")
        train_lstm(ticker, "sentiment")
        
        # Test LSTM models
        test_lstm(ticker, "technical")
        test_lstm(ticker, "sentiment")

    print(f"\n[OK] LSTM pipeline complete.\n")


if __name__ == "__main__":
    run_all()
