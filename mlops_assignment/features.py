"""Feature generation module for mlops_assignment package.

This module provides additional feature engineering on processed datasets.
"""

from pathlib import Path
import sys

from loguru import logger
import typer

from mlops_assignment.config import PROCESSED_DATA_DIR

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "lung_cancer_processed.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
) -> None:
    """Generate features from processed dataset.
    
    This function can be extended with additional feature engineering steps.
    Currently, it copies processed data as features, but can be enhanced with:
    - Feature scaling/normalization
    - Polynomial features
    - Interaction terms
    - Domain-specific feature engineering
    
    Args:
        input_path: Path to processed CSV data file
        output_path: Path where features will be saved
    """
    logger.info(f"Generating features from {input_path} to {output_path}")
    
    try:
        import pandas as pd
        
        # Load processed data
        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            logger.info("Hint: Run dataset processing first using dataset.py")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # For now, just copy processed data as features
        # This can be extended with additional feature engineering
        # Example extensions:
        # - Feature scaling: from sklearn.preprocessing import StandardScaler
        # - Polynomial features: from sklearn.preprocessing import PolynomialFeatures
        # - Feature selection: from sklearn.feature_selection import SelectKBest
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.success(f"Features saved to {output_path}")
        logger.info(f"Final shape: {df.shape}")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure pandas is installed: pip install pandas")
        raise
    except Exception as e:
        logger.error(f"Error generating features: {e}")
        raise


if __name__ == "__main__":
    app()
