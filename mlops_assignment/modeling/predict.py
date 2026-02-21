"""Model prediction module for mlops_assignment package.

This module provides a CLI interface for making predictions with trained models.
It wraps the existing prediction functionality from src/models/predict.py.
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
    model_path: Path = MODELS_DIR / "lung_cancer_model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
) -> None:
    """Make predictions using trained model.
    
    This function loads a trained model and makes predictions on input features.
    Supports both single and batch predictions.
    
    Args:
        features_path: Path to CSV file with features for prediction
        model_path: Path to trained model file
        predictions_path: Path where predictions will be saved
        
    Note:
        Requires Python 3.10 or 3.11 due to PyCaret compatibility.
        Model should be trained first using train.py.
    """
    logger.info(f"Performing inference with model from {model_path}")
    logger.info(f"Features path: {features_path}")
    
    try:
        # Import and use existing prediction code
        from src.models.predict import predict, predict_batch
        import pandas as pd
        
        # Load features
        if not features_path.exists():
            logger.error(f"Features file not found: {features_path}")
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        df = pd.read_csv(features_path)
        logger.info(f"Loaded features shape: {df.shape}")
        
        # Make predictions
        if len(df) > 1:
            # Batch prediction
            logger.info("Performing batch prediction...")
            predictions = predict_batch(df)
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(predictions_path, index=False)
            logger.success(f"Predictions saved to {predictions_path}")
            logger.info(f"Predicted {len(predictions)} samples")
        else:
            # Single prediction
            logger.info("Performing single prediction...")
            result = predict(df.iloc[0].to_dict(), return_proba=True)
            logger.info(f"Prediction result: {result}")
            logger.success("Inference complete.")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    app()
