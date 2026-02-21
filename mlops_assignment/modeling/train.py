"""Model training module for mlops_assignment package.

This module provides a CLI interface for training machine learning models.
It wraps the existing training functionality from src/models/train.py.
"""

from pathlib import Path
import sys

from loguru import logger
import typer

from mlops_assignment.config import MODELS_DIR, PROCESSED_DATA_DIR

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "lung_cancer_processed.csv",
    labels_path: Path = typer.Option(None, help="Path to labels file (if separate)"),
    model_path: Path = MODELS_DIR / "lung_cancer_model.pkl",
) -> None:
    """Train machine learning model using existing training pipeline.
    
    This function wraps the existing train() function from src/models/train.py,
    which uses Hydra for configuration management and PyCaret for model training.
    
    Args:
        features_path: Path to processed features CSV file
        labels_path: Path to labels file (optional, labels may be in features file)
        model_path: Path where trained model will be saved
        
    Note:
        Requires Python 3.10 or 3.11 due to PyCaret compatibility.
        Uses Hydra configuration from config/config.yaml.
    """
    logger.info("Starting model training...")
    logger.info(f"Features path: {features_path}")
    logger.info(f"Model will be saved to: {model_path}")
    
    try:
        # Import and use existing training code
        from src.models.train import train
        
        # Call the existing train function
        # It uses Hydra config, so we'll let it use the default config
        logger.info("Calling existing train module...")
        train()
        logger.success("Model training complete.")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


if __name__ == "__main__":
    app()
