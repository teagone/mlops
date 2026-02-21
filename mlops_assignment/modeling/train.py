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
    labels_path: Path = None,  # Labels are in the same file
    model_path: Path = MODELS_DIR / "lung_cancer_model.pkl",
):
    """
    Train model - wrapper around existing train functionality.
    """
    logger.info("Training model...")
    
    try:
        # Import and use existing training code
        from src.models.train import train
        import hydra
        from omegaconf import DictConfig
        
        # Call the existing train function
        # It uses Hydra config, so we'll let it use the default config
        logger.info("Calling existing train module...")
        train()
        logger.success("Model training complete.")
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


if __name__ == "__main__":
    app()
