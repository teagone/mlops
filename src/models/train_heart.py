"""
Training script for Heart Disease Prediction using PyCaret.

This script implements a complete MLOps pipeline following Task 2 requirements:
1. Initialize training environment with preprocessing (scaling, normalization, feature engineering, binning)
2. Train and evaluate models with k-fold cross-validation
3. Analyze performance using plot_model and evaluate_model
4. Generate predictions on unseen data with predict_model
5. Save the entire pipeline
6. Enable experiment logging using MLflow
7. Register model with MLflow
8. Proper documentation for each step

Author: MLOps Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.sklearn
from pycaret.classification import (
    setup, compare_models, create_model, tune_model,
    finalize_model, evaluate_model, predict_model, save_model,
    plot_model, pull, get_config
)
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="config_heart")
def train(cfg: DictConfig) -> None:
    """
    Train the machine learning model for heart disease prediction.
    
    This function implements the complete ML pipeline as per Task 2 requirements:
    - Data loading and preprocessing
    - Model training with PyCaret
    - Hyperparameter tuning
    - Model evaluation and visualization
    - Model saving and MLflow registration
    
    Args:
        cfg: Hydra configuration object containing all pipeline parameters
    """
    logger.info("=" * 80)
    logger.info("HEART DISEASE PREDICTION - MLOPS PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Set up MLflow tracking
    logger.info("\n[STEP 1] Setting up MLflow tracking...")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.model.experiment_name)
    logger.info(f"MLflow tracking URI: {cfg.mlflow.tracking_uri}")
    logger.info(f"Experiment name: {cfg.model.experiment_name}")
    
    # Step 2: Load and inspect data
    logger.info("\n[STEP 2] Loading data...")
    data_path = Path(cfg.data.raw_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Drop unnecessary columns if present
    cols_to_drop = ['index', 'Patient Id']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Fix feature names: replace spaces with underscores
    df.columns = df.columns.str.replace(' ', '_')
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Target column: {cfg.model.target_column}")
    logger.info(f"\nTarget distribution:\n{df[cfg.model.target_column].value_counts()}")
    logger.info(f"\nData types:\n{df.dtypes}")
    logger.info(f"\nMissing values:\n{df.isnull().sum()}")
    logger.info(f"\nFirst few rows:\n{df.head()}")
    
    # Step 3: Initialize PyCaret with preprocessing pipeline
    logger.info("\n[STEP 3] Initializing PyCaret environment with preprocessing...")
    logger.info("Preprocessing includes:")
    logger.info(f"  - Normalization: {cfg.preprocessing.normalize}")
    logger.info(f"  - Feature Selection: {cfg.preprocessing.feature_selection}")
    logger.info(f"  - Remove Multicollinearity: {cfg.preprocessing.remove_multicollinearity}")
    logger.info(f"  - Multicollinearity Threshold: {cfg.preprocessing.multicollinearity_threshold}")
    logger.info(f"  - Binning Numeric Features: {cfg.preprocessing.bin_numeric_features}")
    logger.info(f"  - Create Interaction Features: {cfg.preprocessing.create_interaction_features}")
    
    # Initialize PyCaret setup with comprehensive preprocessing
    setup(
        data=df,
        target=cfg.model.target_column,
        train_size=1 - cfg.data.test_size,
        test_data=None,
        preprocess=True,
        # Scaling and normalization
        normalize=cfg.preprocessing.normalize,
        normalize_method='zscore',  # Z-score normalization
        
        # Feature engineering
        feature_selection=cfg.preprocessing.feature_selection,
        feature_selection_method='classic',  # Classic feature selection
        feature_interaction=cfg.preprocessing.create_interaction_features,
        
        # Multicollinearity removal
        remove_multicollinearity=cfg.preprocessing.remove_multicollinearity,
        multicollinearity_threshold=cfg.preprocessing.multicollinearity_threshold,
        
        # Binning continuous data into intervals
        bin_numeric_features=cfg.preprocessing.bin_numeric_features if cfg.preprocessing.bin_numeric_features else None,
        
        # Other preprocessing
        remove_outliers=False,  # Can be enabled if needed
        outliers_threshold=0.05,
        
        # Reproducibility
        session_id=cfg.data.random_state,
        
        # MLflow logging (disabled to avoid conflicts, we'll log manually)
        log_experiment=False,
        log_plots=False,
        
        # Silent mode for cleaner output
        silent=False,
        verbose=True
    )
    
    logger.info("PyCaret environment initialized successfully!")
    logger.info("Transformation pipeline created with all preprocessing steps.")
    
    # Step 4: Compare models with k-fold cross-validation
    logger.info("\n[STEP 4] Comparing models with k-fold cross-validation...")
    logger.info(f"Cross-validation folds: {cfg.model.fold}")
    logger.info(f"Optimization metric: {cfg.model.metric}")
    logger.info(f"Number of top models to select: {cfg.model.n_select}")
    
    # Compare multiple models using k-fold cross-validation
    best_models = compare_models(
        include=['lightgbm', 'rf', 'xgboost', 'lr', 'nb', 'dt', 'svm'],
        sort=cfg.model.metric,
        n_select=cfg.model.n_select,
        fold=cfg.model.fold,
        cross_validation=True,
        verbose=True
    )
    
    # Handle single model or list
    if isinstance(best_models, list):
        best_model = best_models[0]
        logger.info(f"\nTop {len(best_models)} models selected:")
        for i, model in enumerate(best_models, 1):
            logger.info(f"  {i}. {type(model).__name__}")
    else:
        best_model = best_models
        logger.info(f"\nBest model selected: {type(best_model).__name__}")
    
    # Get comparison results
    comparison_results = pull()
    logger.info("\nModel Comparison Results:")
    logger.info(f"\n{comparison_results.to_string()}")
    
    # Step 5: Hyperparameter tuning
    logger.info("\n[STEP 5] Performing hyperparameter tuning...")
    logger.info(f"Optimization metric: {cfg.tuning.optimize}")
    logger.info(f"Number of iterations: {cfg.tuning.n_iter}")
    
    tuned_model = tune_model(
        best_model,
        optimize=cfg.tuning.optimize,
        n_iter=cfg.tuning.n_iter,
        fold=cfg.model.fold,
        verbose=True
    )
    
    logger.info("Hyperparameter tuning completed!")
    
    # Step 6: Finalize model (train on entire dataset)
    logger.info("\n[STEP 6] Finalizing model (training on entire dataset)...")
    final_model = finalize_model(tuned_model)
    logger.info("Model finalized successfully!")
    
    # Step 7: Evaluate model performance
    logger.info("\n[STEP 7] Evaluating model performance...")
    logger.info("Generating comprehensive evaluation metrics...")
    
    # Use evaluate_model to get detailed metrics
    evaluate_model(final_model)
    
    # Get detailed metrics
    results = pull()
    logger.info("\nFinal Model Performance Metrics:")
    logger.info(f"\n{results.to_string()}")
    
    # Step 8: Generate and save visualization plots
    logger.info("\n[STEP 8] Generating visualization plots...")
    plots_dir = Path(cfg.paths.reports_path)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to plots directory temporarily to save plots
    import os
    original_dir = os.getcwd()
    try:
        os.chdir(str(plots_dir))
        
        # Generate various plots for model analysis
        logger.info("  - Generating confusion matrix...")
        plot_model(final_model, plot='confusion_matrix', save=True)
        
        logger.info("  - Generating classification report...")
        plot_model(final_model, plot='class_report', save=True)
        
        logger.info("  - Generating feature importance plot...")
        plot_model(final_model, plot='feature', save=True)
        
        logger.info("  - Generating AUC curve...")
        plot_model(final_model, plot='auc', save=True)
        
        logger.info("  - Generating precision-recall curve...")
        plot_model(final_model, plot='pr', save=True)
        
        logger.info("  - Generating learning curve...")
        plot_model(final_model, plot='learning', save=True)
        
        logger.info(f"All plots saved to: {plots_dir}")
    except Exception as e:
        logger.warning(f"Could not generate some plots: {e}")
    finally:
        os.chdir(original_dir)
    
    # Step 9: Generate predictions on test set
    logger.info("\n[STEP 9] Generating predictions on test set...")
    try:
        # Get test data from PyCaret
        X_test = get_config('X_test')
        y_test = get_config('y_test')
        
        # Generate predictions
        predictions = predict_model(final_model, data=X_test)
        
        # Calculate accuracy on test set
        if 'prediction_label' in predictions.columns:
            test_accuracy = (predictions['prediction_label'] == y_test).mean()
            logger.info(f"Test set accuracy: {test_accuracy:.4f}")
        
        # Save predictions
        predictions_path = Path(cfg.data.processed_path.replace('_processed.csv', '_predictions.csv'))
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate test predictions: {e}")
    
    # Step 10: Save the entire pipeline
    logger.info("\n[STEP 10] Saving the entire pipeline...")
    model_save_path = Path(cfg.paths.model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model (this includes the entire preprocessing pipeline)
    save_model(final_model, str(model_save_path))
    logger.info(f"Complete pipeline saved to: {model_save_path}.pkl")
    logger.info("Note: The saved model includes all preprocessing transformations.")
    
    # Step 11: Log experiment to MLflow
    logger.info("\n[STEP 11] Logging experiment to MLflow...")
    with mlflow.start_run(run_name=f"heart_disease_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            'target_column': cfg.model.target_column,
            'test_size': cfg.data.test_size,
            'random_state': cfg.data.random_state,
            'cv_folds': cfg.model.fold,
            'metric': cfg.model.metric,
            'normalize': cfg.preprocessing.normalize,
            'feature_selection': cfg.preprocessing.feature_selection,
            'remove_multicollinearity': cfg.preprocessing.remove_multicollinearity,
            'tuning_iterations': cfg.tuning.n_iter,
            'model_type': type(final_model).__name__
        })
        
        # Get metrics from PyCaret results
        results = pull()
        if not results.empty:
            for metric in results.columns:
                if metric != 'Model':
                    try:
                        value = results[metric].iloc[0]
                        if pd.notna(value):
                            metric_name = metric.lower().replace(' ', '_').replace('-', '_')
                            mlflow.log_metric(metric_name, float(value))
                            logger.info(f"  - Logged metric: {metric_name} = {value:.4f}")
                    except Exception as e:
                        logger.warning(f"Could not log metric {metric}: {e}")
        
        # Log additional metrics if available
        try:
            X_test = get_config('X_test')
            y_test = get_config('y_test')
            predictions = predict_model(final_model, data=X_test)
            
            if 'prediction_label' in predictions.columns:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                y_pred = predictions['prediction_label']
                mlflow.log_metric('test_accuracy', accuracy_score(y_test, y_pred))
                mlflow.log_metric('test_precision', precision_score(y_test, y_pred, average='weighted'))
                mlflow.log_metric('test_recall', recall_score(y_test, y_pred, average='weighted'))
                mlflow.log_metric('test_f1', f1_score(y_test, y_pred, average='weighted'))
        except Exception as e:
            logger.warning(f"Could not log additional test metrics: {e}")
        
        # Log model using sklearn flavor (PyCaret models are sklearn-compatible)
        logger.info("  - Logging model artifact...")
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=cfg.mlflow.model_name
        )
        
        # Log plots as artifacts
        try:
            for plot_file in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(plot_file), "plots")
                logger.info(f"  - Logged plot: {plot_file.name}")
        except Exception as e:
            logger.warning(f"Could not log plots: {e}")
        
        # Step 12: Register model with MLflow
        logger.info("\n[STEP 12] Registering model with MLflow...")
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
                logger.info(f"Model registered and moved to {cfg.mlflow.model_stage} stage")
                logger.info(f"Model name: {cfg.mlflow.model_name}")
                logger.info(f"Model version: {version}")
        except Exception as e:
            logger.warning(f"Could not transition model stage: {e}")
        
        # Log run ID
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        logger.info(f"View results at: {cfg.mlflow.tracking_uri}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nSummary:")
    logger.info(f"  - Model saved: {model_save_path}.pkl")
    logger.info(f"  - Plots saved: {plots_dir}")
    logger.info(f"  - MLflow experiment: {cfg.model.experiment_name}")
    logger.info(f"  - Registered model: {cfg.mlflow.model_name}")
    logger.info(f"\nTo view MLflow UI, run: mlflow ui --backend-store-uri {cfg.mlflow.tracking_uri}")


if __name__ == "__main__":
    train()
