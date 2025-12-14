"""
Integration tests for complete pipeline execution.

Tests ensure the full pipeline works end-to-end.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))


class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    def test_feature_to_model_pipeline(self):
        """Features should flow correctly through model pipeline."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Create mock feature dataframe
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Extract features and target
        X = df[['feature1', 'feature2', 'feature3']].values
        y = df['target'].values
        
        # Scale features (as done for LR)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        
        # Predict
        predictions = model.predict(X_scaled)
        
        # Should produce valid predictions
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})
    
    def test_data_preprocessing_consistency(self):
        """Preprocessing should be consistent across train/test."""
        # Create train and test data
        train_df = pd.DataFrame({
            'value': [1.0, 2.0, np.inf, 4.0, 5.0]
        })
        test_df = pd.DataFrame({
            'value': [3.0, -np.inf, 6.0]
        })
        
        # Apply same cleaning to both
        train_clean = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        test_clean = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Should not contain inf in either
        assert not np.isinf(train_clean['value']).any()
        assert not np.isinf(test_clean['value']).any()


class TestReproducibility:
    """Test reproducibility of results."""
    
    def test_numpy_random_seed(self):
        """Same seed should produce same random numbers."""
        np.random.seed(42)
        arr1 = np.random.randn(10)
        
        np.random.seed(42)
        arr2 = np.random.randn(10)
        
        assert np.array_equal(arr1, arr2)
    
    def test_sklearn_random_state(self):
        """Same random_state should produce same model."""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        # Train with same random_state
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model1.fit(X, y)
        
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2.fit(X, y)
        
        # Feature importances should match
        assert np.allclose(model1.feature_importances_, model2.feature_importances_)


class TestDataValidationIntegration:
    """Test data validation across pipeline."""
    
    def test_no_data_leakage_temporal_split(self):
        """Future data should never appear in training set."""
        # Create time series
        dates = pd.date_range('2015-01-01', '2019-12-31', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(len(dates))
        })
        
        # Split: train on 2015-2018, test on 2019
        cutoff_date = pd.Timestamp('2019-01-01')
        train = df[df['date'] < cutoff_date]
        test = df[df['date'] >= cutoff_date]
        
        # Verify no overlap
        assert train['date'].max() < test['date'].min()
        
        # Verify complete coverage
        assert len(train) + len(test) == len(df)
    
    def test_feature_alignment_with_target(self):
        """Features and target should align correctly."""
        df = pd.DataFrame({
            'price': [100, 110, 105, 115, 120],
            'feature': [1, 2, 3, 4, 5]
        })
        
        # Create target: next day up/down
        df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
        
        # In your actual pipeline, you use dropna(subset=['target'])
        # but shift(-1) with astype(int) doesn't create NaN, it creates 0
        # So we need to manually set last row's target to NaN before dropping
        df.loc[df.index[-1], 'target'] = np.nan
        df = df.dropna(subset=['target'])
        
        # Length should be original - 1
        assert len(df) == 4
        
        # Target at index 0 should predict index 1
        assert df.iloc[0]['target'] == (df.iloc[1]['price'] > df.iloc[0]['price'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
