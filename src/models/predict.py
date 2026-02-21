"""
Prediction script for lung cancer risk model.
Loads the trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import mlflow
import mlflow.sklearn
from pycaret.classification import load_model, predict_model
import warnings
warnings.filterwarnings('ignore')


def load_mlflow_model(
    model_name: str = "lung_cancer_risk_model",
    stage: str = "Staging",
    tracking_uri: str = "./mlruns"
) -> Any:
    """
    Load model from MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage (Staging, Production, etc.)
        tracking_uri: MLflow tracking URI
        
    Returns:
        Loaded PyCaret model
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        # Load model from registry using sklearn flavor
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model from MLflow: {model_uri}")
        return model
    except Exception as e:
        print(f"Warning: Could not load from MLflow registry: {e}")
        print("Attempting to load local model...")
        
        # Fallback to local model
        local_path = Path("models/lung_cancer_model.pkl")
        if local_path.exists():
            model = load_model(str(local_path))
            print(f"Loaded local model from: {local_path}")
            return model
        else:
            raise FileNotFoundError(
                f"Model not found in MLflow registry or at {local_path}. "
                "Please train the model first."
            )


def predict(
    input_data: Union[pd.DataFrame, Dict[str, Any]],
    model_name: str = "lung_cancer_risk_model",
    stage: str = "Staging",
    return_proba: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Make prediction on input data.
    
    Args:
        input_data: DataFrame or dictionary with feature values
        model_name: Name of the registered model
        stage: Model stage
        return_proba: Whether to return probability scores
        
    Returns:
        Prediction result (class label or dictionary with probabilities)
    """
    # Load model
    model = load_mlflow_model(model_name, stage)
    
    # Convert dict to DataFrame if needed
    if isinstance(input_data, dict):
        # Normalize feature names in dict (spaces to underscores)
        normalized_dict = {k.replace(' ', '_'): v for k, v in input_data.items()}
        
        # Load training data to get all required columns
        try:
            data_path = Path("data/raw/lung_cancer.csv")
            if data_path.exists():
                train_df = pd.read_csv(data_path)
                train_df.columns = train_df.columns.str.replace(' ', '_')
                # Get feature columns (exclude target and metadata)
                feature_cols = [c for c in train_df.columns if c not in ['index', 'Patient_Id', 'Level']]
                
                # Create DataFrame with all required columns
                input_df = pd.DataFrame([normalized_dict])
                # Add missing columns with default value 0
                for col in feature_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                # Reorder columns to match training data
                input_df = input_df[feature_cols]
            else:
                input_df = pd.DataFrame([normalized_dict])
        except Exception:
            input_df = pd.DataFrame([normalized_dict])
    else:
        input_df = input_data.copy()
        # Fix feature names: replace spaces with underscores to match training data
        input_df.columns = input_df.columns.str.replace(' ', '_')
    
    # Make prediction
    predictions = predict_model(model, data=input_df)
    
    # Extract prediction
    if 'prediction_label' in predictions.columns:
        pred_label = predictions['prediction_label'].iloc[0]
    elif 'Label' in predictions.columns:
        pred_label = predictions['Label'].iloc[0]
    else:
        # Get the last column (usually the prediction)
        pred_label = predictions.iloc[0, -1]
    
    if return_proba:
        # Get probability columns
        proba_cols = [col for col in predictions.columns if 'Score' in col or 'Probability' in col]
        result = {
            'prediction': str(pred_label),
            'probabilities': {}
        }
        for col in proba_cols:
            result['probabilities'][col] = float(predictions[col].iloc[0])
        return result
    else:
        return str(pred_label)


def predict_batch(
    input_df: pd.DataFrame,
    model_name: str = "lung_cancer_risk_model",
    stage: str = "Staging"
) -> pd.DataFrame:
    """
    Make batch predictions on a DataFrame.
    
    Args:
        input_df: DataFrame with feature columns
        model_name: Name of the registered model
        stage: Model stage
        
    Returns:
        DataFrame with predictions added
    """
    model = load_mlflow_model(model_name, stage)
    
    # Fix feature names: replace spaces with underscores to match training data
    input_df.columns = input_df.columns.str.replace(' ', '_')
    
    predictions = predict_model(model, data=input_df)
    return predictions


# Example usage
if __name__ == "__main__":
    # Example input
    example_input = {
        'Age': 45,
        'Gender': 1,
        'Air Pollution': 5,
        'Alcohol use': 3,
        'Dust Allergy': 4,
        'OccuPational Hazards': 4,
        'Genetic Risk': 3,
        'chronic Lung Disease': 2,
        'Balanced Diet': 3,
        'Obesity': 4,
        'Smoking': 5,
        'Passive Smoker': 3,
        'Chest Pain': 4,
        'Coughing of Blood': 2,
        'Fatigue': 3,
        'Weight Loss': 2,
        'Shortness of Breath': 3,
        'Wheezing': 2,
        'Swallowing Difficulty': 2,
        'Clubbing of Finger Nails': 1,
        'Frequent Cold': 2,
        'Dry Cough': 3,
        'Snoring': 4
    }
    
    try:
        result = predict(example_input, return_proba=True)
        print("Prediction result:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        print("Please train the model first using: python src/models/train.py")
