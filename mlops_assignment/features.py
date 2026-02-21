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
):
    """
    Generate features from processed dataset.
    This is a wrapper that can be extended with additional feature engineering.
    """
    logger.info(f"Generating features from {input_path} to {output_path}")
    
    try:
        import pandas as pd
        
        # Load processed data
        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            logger.info("Running dataset processing first...")
            # Could call dataset.py here if needed
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # For now, just copy processed data as features
        # This can be extended with additional feature engineering
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.success(f"Features saved to {output_path}")
        logger.info(f"Final shape: {df.shape}")
        
    except Exception as e:
        logger.error(f"Error generating features: {e}")
        raise


if __name__ == "__main__":
    app()
