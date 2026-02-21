"""
Prediction script for lung cancer risk model.
Loads the trained model and makes predictions on new data.
"""

"""
Prediction script for lung cancer risk model.
Loads the trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import warnings
from pycaret.classification import load_model, predict_model

warnings.filterwarnings('ignore')


def load_mlflow_model(
    model_name: str = "lung_cancer_risk_model",
    stage: str = "Staging",
    tracking_uri: str = "./mlruns"
) -> Any:
    """
    Load model from local file (PyCaret saved model).
    PyCaret models need to be loaded using load_model() which handles the pipeline.
    
    Args:
        model_name: Name of the registered model (for reference)
        stage: Model stage (for reference)
        tracking_uri: MLflow tracking URI (for reference)
        
    Returns:
        Loaded PyCaret model
    """
    # PyCaret models saved with save_model() need to be loaded with load_model()
    # This handles the full pipeline including preprocessing
    # PyCaret's save_model automatically adds .pkl extension
    local_path = Path("models/lung_cancer_model.pkl")
    if not local_path.exists():
        # Try without .pkl extension (PyCaret adds it automatically)
        local_path = Path("models/lung_cancer_model")
    
    if local_path.exists():
        # load_model expects the path without .pkl extension if PyCaret added it
        model_path = str(local_path).replace('.pkl', '')
        model = load_model(model_path)
        print(f"Loaded local model from: {local_path}")
        return model
    else:
        raise FileNotFoundError(
            f"Model not found at models/lung_cancer_model. "
            "Please train the model first using: poetry run python src/models/train.py"
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
        # Normalize feature names: replace spaces with underscores
        normalized_dict = {k.replace(' ', '_'): v for k, v in input_data.items()}
        input_df = pd.DataFrame([normalized_dict])
    else:
        input_df = input_data.copy()
    
    # Normalize column names: replace spaces with underscores
    input_df.columns = input_df.columns.str.replace(' ', '_')
    
    # Make prediction using PyCaret's predict_model (handles preprocessing pipeline)
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
        # Get prediction score (confidence for predicted class)
        pred_score = None
        if 'prediction_score' in predictions.columns:
            pred_score = float(predictions['prediction_score'].iloc[0])
        
        # Try to get probabilities for all classes
        probabilities = {}
        try:
            # PyCaret models are pipelines with an 'actual_estimator' step
            # We need to preprocess the data first, then get probabilities
            if hasattr(model, 'named_steps') and 'actual_estimator' in model.named_steps:
                # Get the preprocessing steps (everything except actual_estimator)
                preprocessing_steps = [step for name, step in model.named_steps.items() 
                                     if name != 'actual_estimator']
                
                # Apply preprocessing
                preprocessed_data = input_df.copy()
                for step in preprocessing_steps:
                    if hasattr(step, 'transform'):
                        try:
                            preprocessed_data = step.transform(preprocessed_data)
                        except:
                            # Some steps might not have transform or need different handling
                            pass
                
                # Get the actual estimator
                estimator = model.named_steps['actual_estimator']
                
                # Get probabilities
                if hasattr(estimator, 'predict_proba'):
                    proba_array = estimator.predict_proba(preprocessed_data)
                    # Map numeric classes to labels (0=Low, 1=Medium, 2=High based on data)
                    class_labels = ['Low', 'Medium', 'High']
                    for i, prob in enumerate(proba_array[0]):
                        if i < len(class_labels):
                            probabilities[class_labels[i]] = float(prob)
                        else:
                            probabilities[f'Class_{i}'] = float(prob)
            elif hasattr(model, 'classes_'):
                # Fallback: use prediction_score and show it as confidence
                if pred_score is not None:
                    # Map the prediction to show confidence
                    probabilities['Predicted Class'] = str(pred_label)
                    probabilities['Confidence'] = f"{pred_score:.2%}"
        except Exception as e:
            # Fallback: use prediction_score if available
            if pred_score is not None:
                probabilities['Predicted Class'] = str(pred_label)
                probabilities['Confidence'] = f"{pred_score:.2%}"
        
        # If we still don't have probabilities, use prediction_score
        if not probabilities and pred_score is not None:
            probabilities['Predicted Class'] = str(pred_label)
            probabilities['Confidence'] = f"{pred_score:.2%}"
        
        result = {
            'prediction': str(pred_label),
            'probabilities': probabilities
        }
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
    
    # Normalize column names: replace spaces with underscores
    input_df = input_df.copy()
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
