# Heart Disease Prediction Pipeline

This document describes the complete MLOps pipeline for the Heart Disease Prediction dataset, following Task 2 requirements from the IT3385 Assignment.

## Dataset Information

- **Dataset**: Heart Failure Prediction Dataset
- **Source**: Kaggle - https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- **Location**: `data/raw/heart.csv`
- **Size**: 918 rows, 12 columns
- **Target**: `HeartDisease` (binary classification: 0 = No Heart Disease, 1 = Heart Disease)
- **Class Distribution**: 
  - Heart Disease: 508 (55.3%)
  - No Heart Disease: 410 (44.7%)

### Features

**Numeric Features:**
- `Age`: Age of the patient
- `RestingBP`: Resting blood pressure (mm Hg)
- `Cholesterol`: Serum cholesterol (mm/dl)
- `FastingBS`: Fasting blood sugar (1 = >120 mg/dl, 0 = ≤120 mg/dl)
- `MaxHR`: Maximum heart rate achieved
- `Oldpeak`: ST depression induced by exercise relative to rest

**Categorical Features:**
- `Sex`: Sex (M = Male, F = Female)
- `ChestPainType`: Chest pain type (ATA, NAP, ASY, TA)
- `RestingECG`: Resting electrocardiographic results (Normal, ST, LVH)
- `ExerciseAngina`: Exercise-induced angina (Y = Yes, N = No)
- `ST_Slope`: Slope of the peak exercise ST segment (Up, Flat, Down)

## Pipeline Overview

The pipeline implements all requirements from Task 2:

1. ✅ **Preprocessing Pipeline**: Scaling, normalization, feature engineering, binning
2. ✅ **Model Training**: Multiple models with k-fold cross-validation
3. ✅ **Model Evaluation**: Comprehensive performance metrics and visualizations
4. ✅ **Hyperparameter Tuning**: Optimized hyperparameters
5. ✅ **Model Saving**: Complete pipeline saved (including preprocessing)
6. ✅ **MLflow Logging**: Experiment tracking and model registry
7. ✅ **Documentation**: Comprehensive code documentation

## Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Activate Environment

```bash
poetry shell
```

### 3. Run Training Pipeline

```bash
python src/models/train_heart.py
```

This will:
- Load data from `data/raw/heart.csv`
- Initialize PyCaret with preprocessing
- Train and compare multiple models
- Perform hyperparameter tuning
- Generate evaluation plots
- Save the model to `models/heart_disease_model.pkl`
- Log everything to MLflow

### 4. View Results

**MLflow UI:**
```bash
mlflow ui --backend-store-uri ./mlruns
```
Then open: http://localhost:5000

**Generated Plots:**
- Confusion Matrix: `reports/figures/Confusion Matrix.png`
- Classification Report: `reports/figures/Class Report.png`
- Feature Importance: `reports/figures/Feature Importance.png`
- AUC Curve: `reports/figures/AUC.png`
- Precision-Recall Curve: `reports/figures/PR Curve.png`
- Learning Curve: `reports/figures/Learning Curve.png`

## Configuration

The pipeline uses Hydra for configuration management. Main config file: `config/config_heart.yaml`

### Key Configuration Parameters

**Data Configuration:**
- `raw_path`: Path to raw data file
- `test_size`: Proportion for test set (0.2 = 20%)
- `random_state`: Random seed for reproducibility

**Model Configuration:**
- `experiment_name`: MLflow experiment name
- `target_column`: Target variable (`HeartDisease`)
- `metric`: Optimization metric (`Accuracy`)
- `fold`: Cross-validation folds (5)
- `n_select`: Number of top models to select (3)

**Preprocessing Configuration:**
- `normalize`: Enable normalization (true)
- `feature_selection`: Enable feature selection (true)
- `remove_multicollinearity`: Remove correlated features (true)
- `multicollinearity_threshold`: Correlation threshold (0.95)
- `bin_numeric_features`: Features to bin (e.g., `Age`)

**Tuning Configuration:**
- `optimize`: Metric to optimize (`Accuracy`)
- `n_iter`: Hyperparameter search iterations (10)

### Override Configuration

You can override any parameter from the command line:

```bash
# Change test size
python src/models/train_heart.py data.test_size=0.3

# Change number of CV folds
python src/models/train_heart.py model.fold=10

# Change optimization metric
python src/models/train_heart.py model.metric="F1" tuning.optimize="F1"

# Change model stage
python src/models/train_heart.py mlflow.model_stage="Production"
```

## Pipeline Steps

### Step 1: MLflow Setup
- Sets tracking URI and experiment name
- Initializes experiment tracking

### Step 2: Data Loading
- Loads data from CSV
- Inspects data shape, types, and distributions
- Checks for missing values

### Step 3: PyCaret Initialization
- **Normalization**: Z-score normalization
- **Feature Selection**: Classic feature selection method
- **Multicollinearity Removal**: Removes features with correlation > 0.95
- **Binning**: Bins continuous features (e.g., Age) into intervals
- **Feature Interactions**: Creates interaction features
- Creates complete transformation pipeline

### Step 4: Model Comparison
- Compares multiple models: LightGBM, Random Forest, XGBoost, Logistic Regression, Naive Bayes, Decision Tree, SVM
- Uses k-fold cross-validation (default: 5 folds)
- Selects top N models (default: 3)
- Ranks by optimization metric (default: Accuracy)

### Step 5: Hyperparameter Tuning
- Tunes best model's hyperparameters
- Uses Bayesian optimization
- Optimizes specified metric
- Runs for N iterations (default: 10)

### Step 6: Model Finalization
- Trains final model on entire dataset (train + test)
- Includes all preprocessing transformations

### Step 7: Model Evaluation
- Generates comprehensive performance metrics
- Uses `evaluate_model()` function
- Displays metrics: Accuracy, AUC, Precision, Recall, F1, Kappa, MCC

### Step 8: Visualization
- Generates multiple plots:
  - Confusion Matrix
  - Classification Report
  - Feature Importance
  - AUC Curve
  - Precision-Recall Curve
  - Learning Curve
- Saves all plots to `reports/figures/`

### Step 9: Predictions
- Generates predictions on test set
- Calculates test accuracy
- Saves predictions to CSV

### Step 10: Save Pipeline
- Saves complete pipeline (model + preprocessing) to pickle file
- File: `models/heart_disease_model.pkl`
- Can be loaded and used directly for predictions

### Step 11: MLflow Logging
- Logs all parameters (config values)
- Logs all metrics (from model evaluation)
- Logs model artifact
- Logs visualization plots
- Creates MLflow run with timestamp

### Step 12: Model Registration
- Registers model in MLflow Model Registry
- Transitions model to specified stage (default: Staging)
- Model can be promoted to Production later

## Model Usage

### Load and Predict

```python
import pickle
import pandas as pd

# Load the saved pipeline
with open("models/heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare new data (must match training features)
new_data = pd.DataFrame({
    'Age': [63],
    'Sex': ['M'],
    'ChestPainType': ['TA'],
    'RestingBP': [145],
    'Cholesterol': [233],
    'FastingBS': [1],
    'RestingECG': ['Normal'],
    'MaxHR': [150],
    'ExerciseAngina': ['N'],
    'Oldpeak': [2.3],
    'ST_Slope': ['Down']
})

# Make prediction (pipeline handles preprocessing automatically)
prediction = model.predict(new_data)
print(f"Prediction: {prediction[0]}")
```

### Using MLflow Model

```python
import mlflow
import pandas as pd

# Load model from MLflow
model_uri = "models:/heart_disease_model/Staging"
model = mlflow.pyfunc.load_model(model_uri)

# Make predictions
new_data = pd.DataFrame({...})
predictions = model.predict(new_data)
```

## Output Files

After running the pipeline, you'll have:

1. **Model File**: `models/heart_disease_model.pkl`
2. **Predictions**: `data/processed/heart_predictions.csv`
3. **Plots**: `reports/figures/*.png`
4. **MLflow Runs**: `mlruns/` directory
5. **Logs**: Console output with detailed information

## MLflow Model Registry

The model is automatically registered in MLflow. To manage models:

```bash
# List registered models
mlflow models list

# View model versions
mlflow models get-model-version --name heart_disease_model --version 1

# Promote to Production
mlflow models transition-model-stage \
  --name heart_disease_model \
  --version 1 \
  --stage Production
```

## Troubleshooting

### Common Issues

1. **Data file not found**
   - Ensure `data/raw/heart.csv` exists
   - Check path in `config/config_heart.yaml`

2. **MLflow connection errors**
   - Ensure `mlruns/` directory exists
   - Check tracking URI in config

3. **Memory errors**
   - Reduce `model.n_select` (fewer models to compare)
   - Reduce `tuning.n_iter` (fewer tuning iterations)
   - Reduce `data.test_size` (smaller test set)

4. **Import errors**
   - Ensure Poetry environment is activated: `poetry shell`
   - Reinstall dependencies: `poetry install`

## Requirements Compliance

This pipeline fully complies with Task 2 requirements:

✅ **Requirement 1**: Training environment initialized with preprocessing (scaling, normalization, feature engineering, binning)  
✅ **Requirement 2**: Models trained and evaluated with k-fold cross-validation  
✅ **Requirement 3**: Performance analyzed using `plot_model` and `evaluate_model`  
✅ **Requirement 4**: Predictions generated on unseen data with `predict_model`  
✅ **Requirement 5**: Complete pipeline saved (model + preprocessing)  
✅ **Requirement 6**: MLflow experiment logging enabled  
✅ **Requirement 7**: Comprehensive documentation provided  
✅ **Requirement 8**: Model registered with MLflow  

## Next Steps

After training:
1. Review MLflow UI for experiment comparison
2. Analyze generated plots for model insights
3. Deploy model to web application (Task 3)
4. Set up CI/CD pipeline (Task 4)

## References

- [PyCaret Documentation](https://pycaret.readthedocs.io/)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Dataset Source](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
