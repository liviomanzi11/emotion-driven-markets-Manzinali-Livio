"""
Unit tests for data loading and preprocessing modules.

Tests ensure data pipelines work correctly and handle edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from pipelines.feature_engineering_company import (
    compute_moving_average,
    compute_volatility,
    compute_rsi,
    compute_macd
)


class TestTechnicalIndicators:
    """Test suite for technical indicator calculations."""
    
    def test_moving_average_basic(self):
        """Moving average should compute correctly for simple series."""
        series = pd.Series([1, 2, 3, 4, 5])
        ma = compute_moving_average(series, window=3)
        
        # First two values should be NaN
        assert pd.isna(ma.iloc[0])
        assert pd.isna(ma.iloc[1])
        # Third value should be (1+2+3)/3 = 2.0
        assert ma.iloc[2] == 2.0
        # Fourth value should be (2+3+4)/3 = 3.0
        assert ma.iloc[3] == 3.0
    
    def test_moving_average_window_larger_than_series(self):
        """Moving average with window > series length should return all NaN."""
        series = pd.Series([1, 2, 3])
        ma = compute_moving_average(series, window=5)
        assert ma.isna().all()
    
    def test_volatility_calculation(self):
        """Volatility should measure dispersion of returns."""
        # Create series with known volatility
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        vol = compute_volatility(returns, window=3)
        
        # First two should be NaN
        assert pd.isna(vol.iloc[0])
        assert pd.isna(vol.iloc[1])
        # Remaining should be positive
        assert vol.iloc[2] > 0
        assert vol.iloc[3] > 0
    
    def test_rsi_calculation(self):
        """RSI should be between 0 and 100."""
        # Create price series
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105])
        rsi = compute_rsi(prices, period=5)
        
        # Remove NaN values for testing
        rsi_valid = rsi.dropna()
        
        # All RSI values should be between 0 and 100
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()
    
    def test_macd_calculation(self):
        """MACD should return two series."""
        prices = pd.Series(range(50, 150))  # Trending series
        macd_line, signal_line = compute_macd(prices)
        
        # Both should be Series
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        
        # Same length as input
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)


class TestDataValidation:
    """Test data quality and validation."""
    
    def test_dataframe_columns(self):
        """Test that required columns exist in sample data."""
        # Create mock merged dataframe structure
        df = pd.DataFrame({
            'ticker_symbol': ['AAPL'] * 10,
            'date': pd.date_range('2019-01-01', periods=10),
            'adj close': np.random.uniform(100, 200, 10),
            'volume': np.random.randint(1000000, 10000000, 10),
            'polarity': np.random.uniform(-1, 1, 10),
            'tweet_volume': np.random.randint(10, 100, 10)
        })
        
        # Check required columns exist
        required_cols = ['ticker_symbol', 'date', 'adj close', 'volume', 'polarity']
        for col in required_cols:
            assert col in df.columns
    
    def test_no_inf_values(self):
        """Data should not contain infinite values after cleaning."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, np.inf, 4.0, -np.inf]
        })
        
        # Replace inf with NaN (as done in pipeline)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Should not contain inf
        assert not np.isinf(df['value']).any()
    
    def test_date_parsing(self):
        """Dates should be parsed correctly."""
        dates = pd.Series(['2019-01-01', '2019-01-02', '2019-01-03'])
        parsed = pd.to_datetime(dates)
        
        assert parsed.dtype == 'datetime64[ns]'
        assert len(parsed) == 3


class TestFeatureEngineering:
    """Test feature engineering logic."""
    
    def test_target_creation(self):
        """Target should be 1 if next day price is higher."""
        prices = pd.Series([100, 102, 101, 103])
        target = (prices.shift(-1) > prices).astype(int)
        
        # Day 0: 102 > 100 → 1
        assert target.iloc[0] == 1
        # Day 1: 101 < 102 → 0
        assert target.iloc[1] == 0
        # Day 2: 103 > 101 → 1
        assert target.iloc[2] == 1
        # Last day: shift(-1) gives NaN, but astype(int) converts to 0
        # This is expected behavior in pandas
    
    def test_pct_change_returns(self):
        """Returns should be calculated correctly."""
        prices = pd.Series([100, 110, 121])
        returns = prices.pct_change()
        
        # First return is NaN
        assert pd.isna(returns.iloc[0])
        # Second return: (110-100)/100 = 0.10
        assert abs(returns.iloc[1] - 0.10) < 1e-6
        # Third return: (121-110)/110 = 0.10
        assert abs(returns.iloc[2] - 0.10) < 1e-6
    
    def test_log_returns(self):
        """Log returns should be calculated correctly."""
        prices = pd.Series([100, 110, 121])
        log_returns = np.log(prices / prices.shift(1))
        
        # First is NaN
        assert pd.isna(log_returns.iloc[0])
        # Second: log(110/100) ≈ 0.0953
        assert abs(log_returns.iloc[1] - 0.0953) < 0.001
    
    def test_multiple_moving_averages(self):
        """Test MA5, MA10, MA20 calculations."""
        prices = pd.Series(range(1, 31))  # 1 to 30
        
        ma5 = compute_moving_average(prices, 5)
        ma10 = compute_moving_average(prices, 10)
        ma20 = compute_moving_average(prices, 20)
        
        # MA5 at index 4: mean(1,2,3,4,5) = 3
        assert ma5.iloc[4] == 3.0
        # MA10 at index 9: mean(1..10) = 5.5
        assert ma10.iloc[9] == 5.5
        # MA20 at index 19: mean(1..20) = 10.5
        assert ma20.iloc[19] == 10.5
    
    def test_multiple_volatility_windows(self):
        """Test volatility with different windows."""
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01] * 10)
        
        vol5 = compute_volatility(returns, 5)
        vol10 = compute_volatility(returns, 10)
        vol20 = compute_volatility(returns, 20)
        
        # All should be positive (after initial NaN)
        assert (vol5.dropna() > 0).all()
        assert (vol10.dropna() > 0).all()
        assert (vol20.dropna() > 0).all()
    
    def test_sentiment_polarity_range(self):
        """Polarity should be in range [-1, 1]."""
        # Simulate FinBERT probabilities
        sentiment_df = pd.DataFrame({
            'positive': [0.8, 0.1, 0.3],
            'neutral': [0.1, 0.2, 0.4],
            'negative': [0.1, 0.7, 0.3]
        })
        
        # Determine label
        sentiment = sentiment_df.idxmax(axis=1)
        polarity = sentiment.map({'positive': 1.0, 'negative': -1.0, 'neutral': 0.0})
        
        assert (polarity >= -1).all()
        assert (polarity <= 1).all()
        assert polarity.iloc[0] == 1.0  # positive
        assert polarity.iloc[1] == -1.0  # negative


class TestAdvancedFeatures:
    """Test advanced sentiment and engagement features."""
    
    def test_impact_weighting(self):
        """Impact weighting should scale with engagement."""
        df = pd.DataFrame({
            'retweet_num': [10, 100, 1000],
            'like_num': [5, 50, 500],
            'comment_num': [2, 20, 200]
        })
        
        # Impact formula: 1 + log1p(retweet + 2*like + 3*comment)
        engagement = df['retweet_num'] + 2*df['like_num'] + 3*df['comment_num']
        impact = 1 + np.log1p(engagement)
        
        # Higher engagement → higher impact
        assert impact.iloc[2] > impact.iloc[1] > impact.iloc[0]
        # All positive
        assert (impact > 0).all()
    
    def test_delta_features(self):
        """Delta features should measure change over time."""
        sentiment = pd.Series([0.5, 0.6, 0.4, 0.7, 0.3])
        delta = sentiment.diff()
        
        # First is NaN
        assert pd.isna(delta.iloc[0])
        # Changes: +0.1, -0.2, +0.3, -0.4
        assert abs(delta.iloc[1] - 0.1) < 1e-6
        assert abs(delta.iloc[2] - (-0.2)) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
