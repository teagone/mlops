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
):
    """
    Make predictions - wrapper around existing predict functionality.
    """
    logger.info(f"Performing inference with model from {model_path}")
    
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
        # For batch prediction
        if len(df) > 1:
            predictions = predict_batch(df)
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(predictions_path, index=False)
            logger.success(f"Predictions saved to {predictions_path}")
        else:
            # Single prediction
            result = predict(df.iloc[0].to_dict(), return_proba=True)
            logger.info(f"Prediction result: {result}")
            logger.success("Inference complete.")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    app()
