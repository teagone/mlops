# MLOps Pipeline - Lung Cancer Risk Prediction

This project implements a complete MLOps pipeline for lung cancer risk prediction using PyCaret, MLflow, and Hydra.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Pipeline Commands](#pipeline-commands)
- [Configuration](#configuration)
- [Monitoring and Tracking](#monitoring-and-tracking)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before running the pipeline, ensure you have:

- Python 3.10 or higher
- Poetry (for dependency management)
- Git (for version control)

## Project Structure (Cookiecutter MLOps Template)

This project follows the CookieCutter MLOps structure for standardized project organization:

```
mlops/
├── data/
│   ├── raw/                    # Raw, immutable data files (DVC-tracked)
│   ├── processed/              # Processed/cleaned data
│   ├── external/               # External data sources
│   └── interim/                # Intermediate data files
├── src/
│   ├── models/                 # Model training and prediction
│   │   ├── train.py           # Training script
│   │   ├── predict.py         # Prediction functions
│   │   └── utils.py           # Utility functions
│   └── webapp/                # Streamlit web application
│       └── app.py             # Web app interface
├── notebooks/                  # Jupyter notebooks for exploration
├── tests/                      # Unit and integration tests
├── reports/                    # Generated reports and visualizations
│   └── figures/                # Charts and plots (Confusion Matrix, etc.)
├── models/                     # Saved trained models (.pkl files)
├── outputs/                    # Training outputs and logs
├── mlruns/                     # MLflow experiment tracking
│   └── models/                 # MLflow model registry
├── config/                     # Hydra configuration files
│   └── config.yaml            # Main configuration
├── pyproject.toml              # Poetry dependencies
├── poetry.lock                 # Poetry dependency lock file
└── cookiecutter.json           # CookieCutter template configuration
```

## Setup and Installation

### 1. Install Poetry (if not already installed)

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**macOS/Linux:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Install Project Dependencies

Navigate to the project root and install dependencies:

```bash
poetry install
```

This will:
- Create a virtual environment
- Install all dependencies from `poetry.lock`
- Set up the project environment

### 3. Activate the Poetry Virtual Environment

**Windows (PowerShell):**
```powershell
poetry shell
```

**macOS/Linux:**
```bash
poetry shell
```

Alternatively, you can run commands using `poetry run`:

```bash
poetry run python src/models/train.py
```

## Pipeline Commands

### Data Preparation

#### 1. Verify Data Location

Ensure your raw data is in the correct location:

```bash
# Check if raw data exists
ls data/raw/lung_cancer.csv
```

The pipeline expects data at: `data/raw/lung_cancer.csv`

#### 2. Data Processing

Data processing is handled automatically by the training pipeline, but you can verify processed data:

```bash
# Check processed data (generated after first run)
ls data/processed/
```

### Model Training

#### 3. Run Training Pipeline

The main training command uses Hydra for configuration management:

```bash
# Basic training run
poetry run python src/models/train.py

# Or if using poetry shell
python src/models/train.py
```

**What this command does:**
- Loads configuration from `config/config.yaml`
- Reads raw data from `data/raw/lung_cancer.csv`
- Performs data preprocessing (normalization, feature selection, etc.)
- Trains multiple models using PyCaret
- Selects top 3 models based on accuracy
- Performs hyperparameter tuning
- Saves the best model
- Logs experiments to MLflow
- Generates reports and visualizations

#### 4. Training with Configuration Overrides

You can override configuration parameters from the command line:

```bash
# Override test size
poetry run python src/models/train.py data.test_size=0.3

# Override model parameters
poetry run python src/models/train.py model.n_select=5 model.fold=10

# Override multiple parameters
poetry run python src/models/train.py data.test_size=0.25 model.metric="F1" tuning.n_iter=20

# Change model stage in MLflow
poetry run python src/models/train.py mlflow.model_stage="Production"
```

#### 5. Training with Different Configurations

If you have multiple config files:

```bash
# Use a different config file
poetry run python src/models/train.py --config-name=config_production

# Use a different config path
poetry run python src/models/train.py --config-path=configs --config-name=experiment1
```

### Model Management with MLflow

#### 6. Start MLflow UI

View experiment tracking results:

```bash
# Start MLflow UI (default port 5000)
mlflow ui --backend-store-uri ./mlruns

# Or with custom port
mlflow ui --backend-store-uri ./mlruns --port 5001
```

Then open your browser to: `http://localhost:5000`

**What you can do in MLflow UI:**
- View all experiment runs
- Compare model metrics
- View model artifacts
- Register models for deployment
- Track model versions

#### 7. List MLflow Experiments

```bash
# List all experiments
mlflow experiments list --experiment-name lung_cancer_experiment
```

#### 8. Register Model in MLflow

Models are automatically registered during training. To manually register:

```bash
# Register a specific model version
mlflow models register-model \
  --model-uri "runs:/<run-id>/model" \
  --name lung_cancer_risk_model
```

#### 9. Transition Model Stage

```bash
# Promote model to Production
mlflow models transition-model-stage \
  --name lung_cancer_risk_model \
  --version <version-number> \
  --stage Production
```

### Model Inference

#### 10. Load and Use Trained Model

```python
import mlflow
import pandas as pd

# Load model from MLflow
model_uri = "models:/lung_cancer_risk_model/Staging"
model = mlflow.pyfunc.load_model(model_uri)

# Make predictions
data = pd.read_csv("data/processed/test_features.csv")
predictions = model.predict(data)
```

Or using the saved pickle file:

```python
import pickle
import pandas as pd

# Load model
with open("models/lung_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions
data = pd.read_csv("data/processed/test_features.csv")
predictions = model.predict(data)
```

### Testing

#### 11. Run Tests

```bash
# Run all tests
poetry run pytest tests/

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_features.py
```

### Data and Model Inspection

#### 12. View Training Logs

```bash
# View latest training log
cat outputs/$(Get-Date -Format "yyyy-MM-dd")/$(Get-ChildItem "outputs/$(Get-Date -Format 'yyyy-MM-dd')" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name/train.log

# Or manually navigate to the latest output directory
cat outputs/2026-02-21/18-26-35/train.log
```

#### 13. View Generated Reports

After training, check the generated reports:

```bash
# View report figures
ls reports/figures/

# Open reports (Windows)
start reports/figures/Confusion\ Matrix.png
start reports/figures/Feature\ Importance.png
start reports/figures/Class\ Report.png
```

### Cleanup and Maintenance

#### 14. Clean Output Directories

```bash
# Remove old output directories (be careful!)
Remove-Item -Recurse -Force outputs/*

# Remove old MLflow runs (optional)
Remove-Item -Recurse -Force mlruns/*
```

#### 15. Update Dependencies

```bash
# Update all dependencies
poetry update

# Update specific package
poetry update pandas

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name
```

## Configuration

### Configuration File Structure

The main configuration file is located at `config/config.yaml` (or as specified in Hydra config). Key sections:

```yaml
data:
  raw_path: data/raw/lung_cancer.csv
  processed_path: data/processed/lung_cancer_processed.csv
  test_size: 0.2
  random_state: 42

model:
  experiment_name: lung_cancer_experiment
  target_column: Level
  metric: Accuracy
  fold: 5
  n_select: 3

preprocessing:
  normalize: true
  feature_selection: true
  remove_multicollinearity: true
  multicollinearity_threshold: 0.95

tuning:
  optimize: Accuracy
  n_iter: 10
  random_state: 42

mlflow:
  tracking_uri: ./mlruns
  registry_uri: ./mlruns
  model_name: lung_cancer_risk_model
  model_stage: Staging
```

### Configuration Parameters Explained

**Data Configuration:**
- `raw_path`: Path to raw data file
- `processed_path`: Path where processed data will be saved
- `test_size`: Proportion of data for testing (0.2 = 20%)
- `random_state`: Random seed for reproducibility

**Model Configuration:**
- `experiment_name`: MLflow experiment name
- `target_column`: Name of the target variable to predict
- `metric`: Metric to optimize (Accuracy, F1, AUC, etc.)
- `fold`: Number of cross-validation folds
- `n_select`: Number of top models to select

**Preprocessing Configuration:**
- `normalize`: Whether to normalize features
- `feature_selection`: Whether to perform feature selection
- `remove_multicollinearity`: Remove highly correlated features
- `multicollinearity_threshold`: Correlation threshold (0.95 = 95%)

**Tuning Configuration:**
- `optimize`: Metric to optimize during hyperparameter tuning
- `n_iter`: Number of iterations for hyperparameter search
- `random_state`: Random seed for tuning

**MLflow Configuration:**
- `tracking_uri`: Local path for MLflow tracking
- `registry_uri`: Local path for model registry
- `model_name`: Name for registered models
- `model_stage`: Model stage (Staging, Production, Archived)

## Monitoring and Tracking

### MLflow Experiment Tracking

All training runs are automatically logged to MLflow. Each run includes:

- **Metrics**: Accuracy, AUC, F1, Precision, Recall, Kappa, MCC
- **Parameters**: Model hyperparameters
- **Artifacts**: Trained models, plots, reports
- **Tags**: Run metadata

### Viewing Results

1. **Via MLflow UI:**
   ```bash
   mlflow ui --backend-store-uri ./mlruns
   ```

2. **Via Command Line:**
   ```bash
   mlflow runs list --experiment-id 0
   ```

3. **Via Python:**
   ```python
   import mlflow
   
   # List experiments
   experiments = mlflow.search_experiments()
   
   # Get runs for an experiment
   runs = mlflow.search_runs(experiment_names=["lung_cancer_experiment"])
   ```

## Troubleshooting

### Common Issues

#### 1. Poetry Command Not Found

**Solution:**
```bash
# Add Poetry to PATH (Windows)
$env:Path += ";$env:USERPROFILE\.local\bin"

# Or reinstall Poetry
```

#### 2. Module Not Found Errors

**Solution:**
```bash
# Ensure you're in the Poetry environment
poetry shell

# Reinstall dependencies
poetry install
```

#### 3. MLflow UI Not Starting

**Solution:**
```bash
# Check if port is already in use
netstat -ano | findstr :5000

# Use a different port
mlflow ui --backend-store-uri ./mlruns --port 5001
```

#### 4. Data File Not Found

**Solution:**
```bash
# Verify data file exists
Test-Path data/raw/lung_cancer.csv

# Check configuration path
cat config/config.yaml | grep raw_path
```

#### 5. Out of Memory Errors

**Solution:**
- Reduce `model.n_select` in config
- Reduce `data.test_size` to use less data
- Reduce `tuning.n_iter` for faster tuning

#### 6. Hydra Configuration Errors

**Solution:**
```bash
# Validate configuration
poetry run python src/models/train.py --cfg job

# Check config structure
poetry run python src/models/train.py --help
```

### Getting Help

- Check logs: `outputs/<date>/<time>/train.log`
- Check MLflow runs: `mlflow ui`
- Validate config: `poetry run python src/models/train.py --cfg job`

## Quick Reference

### Most Common Commands

```bash
# 1. Install dependencies
poetry install

# 2. Activate environment
poetry shell

# 3. Run training
python src/models/train.py

# 4. View MLflow UI
mlflow ui --backend-store-uri ./mlruns

# 5. Run tests
poetry run pytest tests/
```

### Command Cheat Sheet

| Task | Command |
|------|---------|
| Install dependencies | `poetry install` |
| Activate environment | `poetry shell` |
| Run training | `python src/models/train.py` |
| Override config | `python src/models/train.py data.test_size=0.3` |
| Start MLflow UI | `mlflow ui --backend-store-uri ./mlruns` |
| Run tests | `poetry run pytest tests/` |
| Update dependencies | `poetry update` |
| View logs | `cat outputs/<date>/<time>/train.log` |

## 🏗️ CookieCutter Structure

This project follows the **CookieCutter MLOps** template structure for standardized organization:

- **`data/raw/`** - Immutable raw data (DVC-tracked)
- **`data/processed/`** - Processed/cleaned data
- **`data/external/`** - External data sources
- **`data/interim/`** - Intermediate processing files
- **`src/models/`** - Model training and prediction code
- **`src/webapp/`** - Streamlit web application
- **`notebooks/`** - Jupyter notebooks for EDA
- **`tests/`** - Unit and integration tests
- **`reports/figures/`** - Generated visualizations
- **`models/`** - Trained model artifacts
- **`outputs/`** - Training logs and outputs
- **`mlruns/`** - MLflow experiment tracking
- **`config/`** - Hydra configuration files

This structure ensures consistency and follows MLOps best practices.

---

## ✅ Current Project Status

**Status:** ✅ **FULLY OPERATIONAL**

- ✅ All dependencies installed via Poetry
- ✅ Python 3.11 environment configured
- ✅ Training pipeline tested and working
- ✅ Model registered in MLflow
- ✅ Plots generated in `reports/figures/`
- ✅ Web application functional
- ✅ MLflow tracking working

**Services:**
- 🌐 MLflow UI: http://localhost:5000
- 🖥️ Streamlit App: http://localhost:8501

---

## Additional Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [PyCaret Documentation](https://pycaret.readthedocs.io/)

---

**Note:** This README assumes the training script is located at `src/models/train.py`. If your script is in a different location, update the commands accordingly.

**Last Updated:** February 21, 2025
