"""
Training script for lung cancer risk prediction using PyCaret.
This script trains multiple models, selects the best one, and logs to MLflow.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pycaret
from pycaret.classification import (
    setup, compare_models, create_model, tune_model,
    finalize_model, evaluate_model, predict_model, save_model,
    plot_model, pull
)
import warnings
warnings.filterwarnings('ignore')


@hydra.main(version_base=None, config_path="../../config", config_name="config", output_subdir=None)
def train(cfg: DictConfig) -> None:
    """
    Train the machine learning model.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set up MLflow tracking
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.model.experiment_name)
    
    # Load data
    data_path = Path(cfg.data.raw_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    cols_to_drop = ['index', 'Patient Id']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    print(f"Data shape: {df.shape}")
    print(f"Target column: {cfg.model.target_column}")
    print(f"Target distribution:\n{df[cfg.model.target_column].value_counts()}")
    
    # Initialize PyCaret
    print("\nSetting up PyCaret environment...")
    setup(
        data=df,
        target=cfg.model.target_column,
        train_size=1 - cfg.data.test_size,
        test_data=None,
        preprocess=True,
        normalize=cfg.preprocessing.normalize,
        feature_selection=cfg.preprocessing.feature_selection,
        remove_multicollinearity=cfg.preprocessing.remove_multicollinearity,
        multicollinearity_threshold=cfg.preprocessing.multicollinearity_threshold,
        bin_numeric_features=cfg.preprocessing.bin_numeric_features if cfg.preprocessing.bin_numeric_features else None,
        session_id=cfg.data.random_state,
        silent=True,
        log_experiment=True,
        experiment_name=cfg.model.experiment_name,
        log_plots=True
    )
    
    # Compare models
    print("\nComparing models...")
    best_models = compare_models(
        include=['lightgbm', 'lr'],
        sort=cfg.model.metric,
        n_select=cfg.model.n_select,
        fold=cfg.model.fold,
        cross_validation=True
    )
    
    # Handle single model or list
    if isinstance(best_models, list):
        best_model = best_models[0]
        print(f"\nTop {len(best_models)} models selected")
    else:
        best_model = best_models
        print(f"\nBest model: {type(best_model).__name__}")
    
    # Hyperparameter tuning
    print("\nTuning hyperparameters...")
    tuned_model = tune_model(
        best_model,
        optimize=cfg.tuning.optimize,
        n_iter=cfg.tuning.n_iter,
        fold=cfg.model.fold
    )
    
    # Finalize model (train on entire dataset)
    print("\nFinalizing model...")
    final_model = finalize_model(tuned_model)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(final_model)
    
    # Generate and save plots
    print("\nGenerating plots...")
    plots_dir = Path(cfg.paths.reports_path)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        plot_model(final_model, plot='confusion_matrix', save=True)
        plot_model(final_model, plot='class_report', save=True)
        plot_model(final_model, plot='feature', save=True)
        print(f"Plots saved to: {plots_dir}")
    except Exception as e:
        print(f"Warning: Could not generate some plots: {e}")
    
    # Save model locally
    model_save_path = Path(cfg.paths.model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(final_model, str(model_save_path))
    print(f"\nModel saved to: {model_save_path}.pkl")
    
    # Log model to MLflow and register
    print("\nLogging model to MLflow...")
    with mlflow.start_run():
        # Get metrics from PyCaret
        results = pull()
        if not results.empty:
            for metric in results.columns:
                if metric != 'Model':
                    try:
                        value = results[metric].iloc[0]
                        if pd.notna(value):
                            mlflow.log_metric(metric.lower().replace(' ', '_'), float(value))
                    except Exception as e:
                        print(f"Warning: Could not log metric {metric}: {e}")
        
        # Log model
        mlflow.pycaret.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=cfg.mlflow.model_name
        )
        
        # Transition model to Staging
        client = mlflow.tracking.MlflowClient()
        try:
            latest_version = client.get_latest_versions(
                cfg.mlflow.model_name,
                stages=["None"]
            )
            if latest_version:
                version = latest_version[0].version
                client.transition_model_version_stage(
                    name=cfg.mlflow.model_name,
                    version=version,
                    stage=cfg.mlflow.model_stage
                )
                print(f"Model registered and moved to {cfg.mlflow.model_stage} stage")
        except Exception as e:
            print(f"Warning: Could not transition model stage: {e}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train()
