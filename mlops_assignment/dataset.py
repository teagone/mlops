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
):
    """
    Process raw dataset - wrapper around existing build_features functionality.
    """
    logger.info(f"Processing dataset from {input_path} to {output_path}")
    
    try:
        # Import and use existing feature building code
        from src.features.build_features import build_features
        import hydra
        from omegaconf import DictConfig
        
        # Create a minimal config for compatibility
        class Config:
            data = type('obj', (object,), {
                'raw_path': str(input_path),
                'processed_path': str(output_path),
                'test_size': 0.2,
                'random_state': 42
            })()
        
        cfg = Config()
        
        # Call the existing build_features function
        logger.info("Calling existing build_features module...")
        build_features(cfg)
        logger.success("Processing dataset complete.")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise


if __name__ == "__main__":
    app()
