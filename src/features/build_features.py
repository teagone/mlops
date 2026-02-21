"""
Feature engineering script for lung cancer dataset.
This script performs data cleaning and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../config", config_name="config", output_subdir=None)
def build_features(cfg: DictConfig) -> None:
    """
    Build features from raw data.
    
    Args:
        cfg: Hydra configuration object
    """
    # Load raw data
    raw_path = Path(cfg.data.raw_path)
    df = pd.read_csv(raw_path)
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Drop index column if present
    if 'index' in df.columns:
        df = df.drop('index', axis=1)
    
    # Drop Patient Id (not useful for modeling)
    if 'Patient Id' in df.columns:
        df = df.drop('Patient Id', axis=1)
    
    # Handle missing values (if any)
    if df.isnull().sum().sum() > 0:
        print("Missing values found:")
        print(df.isnull().sum())
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Feature engineering: Create age groups
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=[0, 30, 50, 70, 100],
            labels=['Young', 'Adult', 'Middle-Aged', 'Senior']
        )
    
    # Create risk score (sum of key risk factors)
    risk_factors = [
        'Air Pollution', 'Smoking', 'Alcohol use', 
        'Genetic Risk', 'Obesity', 'Occupational Hazards'
    ]
    
    available_risk_factors = [col for col in risk_factors if col in df.columns]
    if available_risk_factors:
        df['Risk_Score'] = df[available_risk_factors].sum(axis=1)
    
    # Save processed data
    processed_path = Path(cfg.data.processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    
    print(f"Processed data saved to: {processed_path}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    build_features()
