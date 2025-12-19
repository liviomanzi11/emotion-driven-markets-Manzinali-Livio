"""
Integration tests that import and execute real project functions.

These tests increase coverage by actually calling src/ modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))


class TestRealFeatureEngineering:
    """Test actual feature engineering functions from src/."""
    
    def test_real_compute_moving_average(self):
        """Test the actual compute_moving_average function."""
        from pipelines.feature_engineering_company import compute_moving_average
        
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ma5 = compute_moving_average(series, 5)
        
        # First 4 should be NaN
        assert ma5.iloc[:4].isna().all()
        # 5th value: mean(1,2,3,4,5) = 3
        assert ma5.iloc[4] == 3.0
        # 10th value: mean(6,7,8,9,10) = 8
        assert ma5.iloc[9] == 8.0
    
    def test_real_compute_volatility(self):
        """Test the actual compute_volatility function."""
        from pipelines.feature_engineering_company import compute_volatility
        
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01, 0.0, -0.01, 0.02])
        vol = compute_volatility(returns, 5)
        
        # First 4 should be NaN
        assert vol.iloc[:4].isna().all()
        # Remaining should be positive
        assert (vol.iloc[4:] > 0).all()
    
    def test_real_compute_rsi(self):
        """Test the actual compute_rsi function."""
        from pipelines.feature_engineering_company import compute_rsi
        
        # Trending up series
        prices = pd.Series([100 + i*2 for i in range(50)])
        rsi = compute_rsi(prices, 14)
        
        # RSI should be between 0 and 100
        rsi_valid = rsi.dropna()
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()
        
        # Trending up should have RSI > 50
        assert rsi_valid.iloc[-1] > 50
    
    def test_real_compute_macd(self):
        """Test the actual compute_macd function."""
        from pipelines.feature_engineering_company import compute_macd
        
        prices = pd.Series(range(100, 200))
        macd_line, signal_line = compute_macd(prices)
        
        # Both should be Series
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        
        # Same length as input
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        
        # MACD for trending up should be positive
        assert macd_line.iloc[-1] > 0


class TestRealMergeFunctions:
    """Test actual merge and aggregation functions."""
    
    def test_compute_tweet_impact(self):
        """Test impact score calculation."""
        from pipelines.merge_sentiment_company_daily import compute_tweet_impact
        
        df = pd.DataFrame({
            'retweet_num': [0, 10, 100],
            'like_num': [0, 5, 50],
            'comment_num': [0, 2, 20]
        })
        
        impact = compute_tweet_impact(df)
        
        # All should be >= 1 (base value)
        assert (impact >= 1).all()
        
        # Higher engagement → higher impact
        assert impact.iloc[2] > impact.iloc[1] > impact.iloc[0]
    
    def test_compute_writer_influence(self):
        """Test writer influence calculation."""
        from pipelines.merge_sentiment_company_daily import compute_writer_influence
        
        df = pd.DataFrame({
            'writer': ['alice', 'alice', 'bob', 'bob', 'charlie'],
            'retweet_num': [100, 100, 10, 10, 1000],
            'like_num': [50, 50, 5, 5, 500],
            'comment_num': [20, 20, 2, 2, 200]
        })
        
        influence_map = compute_writer_influence(df)
        
        # All writers should have positive influence
        assert all(v > 0 for v in influence_map.values())
        
        # Charlie has highest engagement → highest influence
        assert influence_map['charlie'] > influence_map['alice']
        assert influence_map['alice'] > influence_map['bob']


class TestModelPrepareFunctions:
    """Test model preparation logic."""
    
    def test_prepare_features_technical(self):
        """Test prepare_features for technical features."""
        from models.classical_models_company import prepare_features
        
        # Create mock dataframe with all technical features
        df = pd.DataFrame({
            'return': np.random.randn(50),
            'volume': np.random.randint(1000, 10000, 50),
            'log_return': np.random.randn(50),
            'ma5': np.random.randn(50),
            'ma10': np.random.randn(50),
            'ma20': np.random.randn(50),
            'volatility_5d': np.random.rand(50),
            'volatility_10d': np.random.rand(50),
            'volatility_20d': np.random.rand(50),
            'rsi_14': np.random.uniform(0, 100, 50),
            'macd_line': np.random.randn(50),
            'signal_line': np.random.randn(50),
            'adj close': np.random.uniform(100, 200, 50),
            'close': np.random.uniform(100, 200, 50),
            'open': np.random.uniform(100, 200, 50),
            'target': np.random.randint(0, 2, 50)
        })
        
        X, y = prepare_features(df, "technical")
        
        # Should have 15 technical features
        assert X.shape[1] == 15
        # Target should match input
        assert len(y) == 50
        # X should be DataFrame
        assert isinstance(X, pd.DataFrame)
    
    def test_prepare_features_sentiment(self):
        """Test prepare_features for sentiment features."""
        from models.classical_models_company import prepare_features
        
        # Create mock dataframe with all sentiment features
        df = pd.DataFrame({
            'polarity': np.random.uniform(-1, 1, 50),
            'positive': np.random.uniform(0, 1, 50),
            'negative': np.random.uniform(0, 1, 50),
            'impact_weighted_sentiment': np.random.randn(50),
            'influence_weighted_sentiment': np.random.randn(50),
            'impact_weighted_ma3': np.random.randn(50),
            'impact_weighted_ma7': np.random.randn(50),
            'delta_polarity': np.random.randn(50),
            'delta_impact_weighted_sentiment': np.random.randn(50),
            'sentiment_volatility_5d': np.random.rand(50),
            'extreme_count_5d': np.random.randint(0, 5, 50),
            'tweet_volume': np.random.randint(10, 100, 50),
            'tweet_volume_ma5': np.random.randint(10, 100, 50),
            'engagement_ma5': np.random.randn(50),
            'engagement_change': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })
        
        X, y = prepare_features(df, "sentiment")
        
        # Should have 15 sentiment features
        assert X.shape[1] == 15
        # All features should be selected
        assert 'polarity' in X.columns
        assert 'impact_weighted_sentiment' in X.columns


class TestLSTMFunctions:
    """Test LSTM-specific functions."""
    
    def test_create_sequences_shape(self):
        """Test LSTM sequence creation."""
        from models.lstm_models_company import create_sequences
        
        # Create mock dataframe
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        X, y, scaler = create_sequences(df)
        
        # X should be 3D: (samples, sequence_length, features)
        assert len(X.shape) == 3
        # Sequence length should be 45
        assert X.shape[1] == 45
        # 3 features
        assert X.shape[2] == 3
        # y should match number of sequences
        assert len(y) == len(X)
        # Scaler should be StandardScaler
        from sklearn.preprocessing import StandardScaler
        assert isinstance(scaler, StandardScaler)


class TestBacktestFunctions:
    """Test backtesting computation functions."""
    
    def test_compute_strategy_function(self):
        """Test the actual compute_strategy function."""
        from strategies.backtest_company import compute_strategy
        
        probas = np.array([0.6, 0.7, 0.4, 0.8, 0.3])
        actual_returns = np.array([0.02, -0.01, 0.01, 0.03, -0.02])
        dates = pd.date_range('2019-01-01', periods=5)
        
        result = compute_strategy(probas, actual_returns, dates)
        
        # Should return dict with required keys
        assert 'equity' in result
        assert 'returns' in result
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'dates' in result
        
        # Equity should be numpy array
        assert isinstance(result['equity'], np.ndarray)
        # Total return should be float
        assert isinstance(result['total_return'], (int, float))
        # Sharpe should be finite
        assert np.isfinite(result['sharpe_ratio'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
