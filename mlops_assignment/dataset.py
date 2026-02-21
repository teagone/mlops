"""Dataset processing module for mlops_assignment package.

This module provides data cleaning and preprocessing functionality.
"""

from pathlib import Path
import sys

from loguru import logger
import typer

from mlops_assignment.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "lung_cancer.csv",
    output_path: Path = PROCESSED_DATA_DIR / "lung_cancer_processed.csv",
) -> None:
    """Process raw dataset: clean, preprocess, and engineer features.
    
    This function performs:
    - Data cleaning (removes index columns, handles missing values)
    - Feature engineering (age groups, risk scores)
    - Saves processed data to output path
    
    Args:
        input_path: Path to raw CSV data file
        output_path: Path where processed data will be saved
    """
    logger.info(f"Processing dataset from {input_path} to {output_path}")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Load raw data
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Drop index column if present
        if 'index' in df.columns:
            df = df.drop('index', axis=1)
            logger.debug("Dropped 'index' column")
        
        # Drop Patient Id (not useful for modeling)
        if 'Patient Id' in df.columns:
            df = df.drop('Patient Id', axis=1)
            logger.debug("Dropped 'Patient Id' column")
        
        # Handle missing values (if any)
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values, filling with median")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Feature engineering: Create age groups
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(
                df['Age'],
                bins=[0, 30, 50, 70, 100],
                labels=['Young', 'Adult', 'Middle-Aged', 'Senior']
            )
            logger.debug("Created 'Age_Group' feature")
        
        # Create risk score (sum of key risk factors)
        risk_factors = [
            'Air Pollution', 'Smoking', 'Alcohol use', 
            'Genetic Risk', 'Obesity', 'Occupational Hazards'
        ]
        
        available_risk_factors = [col for col in risk_factors if col in df.columns]
        if available_risk_factors:
            df['Risk_Score'] = df[available_risk_factors].sum(axis=1)
            logger.debug(f"Created 'Risk_Score' feature from {len(available_risk_factors)} factors")
        
        # Save processed data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.success(f"Processed data saved to {output_path}")
        logger.info(f"Final shape: {df.shape}")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure pandas and numpy are installed: pip install pandas numpy")
        raise
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise


if __name__ == "__main__":
    app()
