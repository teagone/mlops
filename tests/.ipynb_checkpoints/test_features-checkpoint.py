"""
Unit tests for feature engineering functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.utils import load_data, prepare_data_for_prediction, get_feature_ranges


def test_load_data():
    """Test data loading function."""
    data_path = Path(__file__).parent.parent / "data" / "raw" / "lung_cancer.csv"
    
    if data_path.exists():
        df = load_data(str(data_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'index' not in df.columns or 'Patient Id' not in df.columns
    else:
        pytest.skip("Data file not found")


def test_prepare_data_for_prediction():
    """Test data preparation for prediction."""
    input_dict = {
        'Age': 45,
        'Gender': 1,
        'Air Pollution': 5,
        'Smoking': 3
    }
    
    feature_columns = ['Age', 'Gender', 'Air Pollution', 'Smoking', 'Alcohol use']
    df = prepare_data_for_prediction(input_dict, feature_columns)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert all(col in df.columns for col in feature_columns)
    assert df['Age'].iloc[0] == 45


def test_get_feature_ranges():
    """Test feature ranges extraction."""
    data_path = Path(__file__).parent.parent / "data" / "raw" / "lung_cancer.csv"
    
    if data_path.exists():
        ranges = get_feature_ranges(str(data_path))
        assert isinstance(ranges, dict)
        if ranges:
            assert 'min' in list(ranges.values())[0]
            assert 'max' in list(ranges.values())[0]
    else:
        pytest.skip("Data file not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
