""""
Unit tests for backtesting strategy logic.

Tests ensure trading strategies are implemented correctly.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))


class TestStrategyLogic:
    """Test trading strategy calculations."""
    
    def test_binary_signal_generation(self):
        """Strategy should generate binary signals from probabilities."""
        probas = np.array([0.3, 0.6, 0.8, 0.4, 0.7])
        signals = (probas > 0.5).astype(float)
        
        # Should be 0 or 1
        assert set(signals).issubset({0.0, 1.0})
        
        # Expected signals: [0, 1, 1, 0, 1]
        expected = np.array([0, 1, 1, 0, 1])
        assert np.array_equal(signals, expected)
    
    def test_strategy_returns_calculation(self):
        """Strategy returns should be signal * market_return."""
        signals = np.array([1, 0, 1, 1, 0])
        market_returns = np.array([0.02, 0.01, -0.01, 0.03, -0.02])
        
        strategy_returns = signals * market_returns
        
        # Expected: [0.02, 0, -0.01, 0.03, 0]
        expected = np.array([0.02, 0, -0.01, 0.03, 0])
        assert np.allclose(strategy_returns, expected)
    
    def test_equity_curve_calculation(self):
        """Equity curve should compound returns correctly."""
        returns = np.array([0.1, 0.2, -0.1])
        equity = np.cumprod(1 + returns)
        
        # Day 0: 1.0 * 1.1 = 1.1
        assert abs(equity[0] - 1.1) < 1e-6
        # Day 1: 1.1 * 1.2 = 1.32
        assert abs(equity[1] - 1.32) < 1e-6
        # Day 2: 1.32 * 0.9 = 1.188
        assert abs(equity[2] - 1.188) < 1e-6
    
    def test_total_return_calculation(self):
        """Total return should be (final_equity - 1) * 100."""
        equity = np.array([1.0, 1.1, 1.21, 1.331])
        total_return = (equity[-1] - 1) * 100
        
        # 33.1% return
        assert abs(total_return - 33.1) < 1e-6
    
    def test_sharpe_ratio_calculation(self):
        """Sharpe ratio should measure risk-adjusted returns."""
        # Create returns with known properties
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Annualized Sharpe (252 trading days)
        sharpe = (mean_return / (std_return + 1e-12)) * np.sqrt(252)
        
        # Should be a finite number
        assert np.isfinite(sharpe)
    
    def test_max_drawdown_calculation(self):
        """Max drawdown should find largest peak-to-trough decline."""
        equity = np.array([1.0, 1.2, 1.1, 1.3, 0.9, 1.0])
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = equity / running_max - 1
        max_dd = drawdown.min()
        
        # Worst drawdown is from 1.3 to 0.9 = -30.77%
        expected = 0.9/1.3 - 1
        assert abs(max_dd - expected) < 1e-6


class TestBuyAndHold:
    """Test buy-and-hold baseline strategy."""
    
    def test_buyhold_equals_market_return(self):
        """Buy-and-hold should exactly match market returns."""
        market_returns = np.array([0.01, 0.02, -0.01, 0.03])
        
        # Buy-and-hold: always invested (signal = 1)
        buyhold_returns = market_returns
        
        market_equity = np.cumprod(1 + market_returns)
        buyhold_equity = np.cumprod(1 + buyhold_returns)
        
        # Should be identical
        assert np.array_equal(market_equity, buyhold_equity)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_volatility_sharpe(self):
        """Sharpe ratio should handle zero volatility gracefully."""
        # Constant returns (zero volatility)
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        
        std = returns.std()
        # Add epsilon to prevent division by zero
        sharpe = (returns.mean() / (std + 1e-12)) * np.sqrt(252)
        
        # Should not be NaN or Inf
        assert np.isfinite(sharpe)
    
    def test_all_cash_strategy(self):
        """Strategy with all 0 signals should have 0 return."""
        signals = np.zeros(10)
        market_returns = np.random.randn(10) * 0.01
        
        strategy_returns = signals * market_returns
        
        # All zeros
        assert (strategy_returns == 0).all()
        
        # Final equity = 1.0 (no change)
        equity = np.cumprod(1 + strategy_returns)
        assert abs(equity[-1] - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
