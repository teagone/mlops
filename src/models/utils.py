"""
Utility functions for model training and prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import mlflow


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_csv(data_path)
    
    # Drop index and Patient Id columns if present
    cols_to_drop = ['index', 'Patient Id']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    return df


def prepare_data_for_prediction(
    input_dict: Dict[str, Any],
    feature_columns: list
) -> pd.DataFrame:
    """
    Convert input dictionary to DataFrame for prediction.
    
    Args:
        input_dict: Dictionary with feature values
        feature_columns: List of expected feature column names
        
    Returns:
        DataFrame ready for prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_dict])
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Default value
    
    # Reorder columns to match training data
    df = df[feature_columns]
    
    return df


def get_feature_ranges(data_path: str) -> Dict[str, Dict[str, float]]:
    """
    Get min/max ranges for numeric features from the dataset.
    
    Args:
        data_path: Path to the training data
        
    Returns:
        Dictionary with feature ranges
    """
    df = pd.read_csv(data_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ranges = {}
    
    for col in numeric_cols:
        if col not in ['index']:
            ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean())
            }
    
    return ranges
