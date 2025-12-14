"""
Unit tests for model training and evaluation.

Tests ensure models train correctly and produce valid predictions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))


class TestModelTraining:
    """Test model training components."""
    
    def test_random_state_reproducibility(self):
        """Same random state should produce same results."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train two models with same seed
        model1 = RandomForestClassifier(random_state=42, n_estimators=10)
        model1.fit(X, y)
        pred1 = model1.predict(X)
        
        model2 = RandomForestClassifier(random_state=42, n_estimators=10)
        model2.fit(X, y)
        pred2 = model2.predict(X)
        
        # Predictions should be identical
        assert np.array_equal(pred1, pred2)
    
    def test_class_weights_balance(self):
        """Class weights should handle imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Imbalanced dataset: 90 class 0, 10 class 1
        y = np.array([0]*90 + [1]*10)
        
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=y
        )
        
        # Weight for minority class should be higher
        assert weights[1] > weights[0]
    
    def test_train_test_split_temporal(self):
        """Train/test split should respect temporal ordering."""
        df = pd.DataFrame({
            'date': pd.date_range('2015-01-01', '2019-12-31', freq='D'),
            'value': np.random.randn(1826)
        })
        
        # Split like in project: 2015-2018 train, 2019 test
        train = df[(df['date'] >= '2015-01-01') & (df['date'] <= '2018-12-31')]
        test = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2019-12-31')]
        
        # Train should end before test begins
        assert train['date'].max() < test['date'].min()
        # No overlap
        assert len(set(train['date']) & set(test['date'])) == 0


class TestPredictions:
    """Test model prediction behavior."""
    
    def test_probability_range(self):
        """Predictions should be valid probabilities."""
        from sklearn.linear_model import LogisticRegression
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Get probabilities
        probs = model.predict_proba(X)
        
        # All probabilities should be between 0 and 1
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        
        # Each row should sum to 1
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_binary_classification_output(self):
        """Binary classifier should output 2 classes."""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(X, y)
        
        # Predictions should be 0 or 1
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})


class TestScaling:
    """Test feature scaling for Logistic Regression."""
    
    def test_standard_scaler_properties(self):
        """StandardScaler should normalize to mean=0, std=1."""
        from sklearn.preprocessing import StandardScaler
        
        # Create data with known mean and std
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Mean should be close to 0
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        
        # Std should be close to 1
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_scaler_consistency(self):
        """Same scaler should produce same transformation."""
        from sklearn.preprocessing import StandardScaler
        
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(20, 5)
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Transform test data twice
        X_test_1 = scaler.transform(X_test)
        X_test_2 = scaler.transform(X_test)
        
        # Should be identical
        assert np.array_equal(X_test_1, X_test_2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
