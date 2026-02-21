# MLOps Assignment: Health Risk Prediction System

A comprehensive MLOps project for predicting lung cancer risk and heart failure using machine learning, with full CI/CD pipeline, monitoring, and deployment capabilities.

## 👥 Team Members

- **Team Member 1**: [Your Name] - Lung Cancer Risk Prediction
- **Team Member 2**: [Teammate Name] - Heart Failure Prediction

## 📋 Project Overview

This project implements a complete MLOps pipeline for health risk prediction, including:

- **Task 1**: Exploratory Data Analysis (EDA) on lung cancer dataset
- **Task 2**: ML pipeline with PyCaret, MLflow tracking, and model registration
- **Task 3**: Integrated Streamlit web application for real-time predictions
- **Task 4**: Complete MLOps environment with DVC, Poetry, Hydra, CI/CD, and monitoring

## 📁 Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── lung_cancer.csv          # DVC-tracked dataset
│   ├── processed/                   # Processed/cleaned data
│   ├── interim/                     # Intermediate data
│   └── external/                    # External data sources
├── notebooks/
│   └── eda_lung_cancer.ipynb        # Task 1: EDA notebook
├── src/                             # Original source code structure
│   ├── features/
│   │   └── build_features.py        # Feature engineering
│   ├── models/
│   │   ├── train.py                 # Task 2: Model training
│   │   ├── predict.py               # Prediction functions
│   │   └── utils.py                 # Utility functions
│   └── webapp/
│       └── app.py                   # Task 3: Streamlit app
├── mlops_assignment/                 # Integrated package structure
│   ├── __init__.py
│   ├── config.py                    # Path configuration
│   ├── dataset.py                   # Data processing CLI
│   ├── features.py                  # Feature generation CLI
│   ├── plots.py                     # Visualization CLI
│   └── modeling/
│       ├── __init__.py
│       ├── train.py                 # Model training CLI
│       └── predict.py               # Prediction CLI
├── config/
│   ├── config.yaml                  # Hydra configuration
│   └── db/
│       └── mlflow.db                 # Local MLflow database
├── tests/
│   ├── test_features.py
│   ├── test_predict.py
│   └── test_webapp.py
├── .github/
│   └── workflows/
│       ├── ci.yml                   # CI pipeline
│       └── cd.yml                   # CD pipeline
├── monitoring/
│   └── drift_detection.py           # Task 4: Data drift monitoring
├── reports/
│   └── figures/                     # Model evaluation plots
├── models/                           # Trained models
├── dvc.yaml                          # DVC pipeline definition
├── Makefile                          # Make commands for common tasks
├── pyproject.toml                    # Poetry dependencies
├── requirements.txt                  # Pip requirements
├── Dockerfile                        # Container configuration
└── README.md                         # This file
```

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10 or 3.11** (required - PyCaret supports 3.9-3.11, but Streamlit requires >=3.10)
- Poetry (for dependency management) or pip
- DVC (for data version control)
- Git

**Note:** Python 3.12+ is not supported due to PyCaret compatibility limitations. For data processing and feature generation, Python 3.12 works, but model training and predictions require Python 3.10/3.11.

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd mlops
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   # Or on Windows:
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Pull data with DVC**:
   ```bash
   dvc pull
   ```
   Note: Ensure DVC is configured with your remote storage. If the data is already in `data/raw/`, you can skip this step.

5. **Activate the virtual environment**:
   ```bash
   poetry env activate
   ```

## 📊 Task 1: Exploratory Data Analysis

Run the EDA notebook to explore the lung cancer dataset:

```bash
poetry run jupyter lab
```

Navigate to `notebooks/eda_lung_cancer.ipynb` and run all cells.

The notebook includes:
- Data loading and basic information
- Target variable analysis
- Numerical and categorical feature distributions
- Correlation analysis
- Outlier detection
- Key insights and recommendations

### Using the mlops_assignment Package

You can also use the integrated package for data processing:

```bash
# Process raw data
python mlops_assignment/dataset.py --input-path data/raw/lung_cancer.csv --output-path data/processed/lung_cancer_processed.csv

# Generate features
python mlops_assignment/features.py --input-path data/processed/lung_cancer_processed.csv --output-path data/processed/features.csv

# Or use Makefile commands
make data
```

## 🤖 Task 2: Model Training

Train the machine learning model using PyCaret with MLflow tracking:

### Using Original Structure

```bash
poetry run python src/models/train.py
```

### Using mlops_assignment Package

```bash
# Using the integrated package CLI
python mlops_assignment/modeling/train.py

# Or using Makefile
make train
```

**Note:** Both methods require Python 3.10 or 3.11 due to PyCaret compatibility.

This will:
1. Load data from `data/raw/lung_cancer.csv`
2. Set up PyCaret environment with preprocessing
3. Compare multiple models (Random Forest, XGBoost, LightGBM, etc.)
4. Perform hyperparameter tuning
5. Evaluate the best model
6. Save the model and register it in MLflow

### View MLflow UI

To view experiment tracking and model registry:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open `http://localhost:5000` in your browser.

### Make Predictions

#### Using Original Structure

```bash
poetry run python src/models/predict.py
```

#### Using mlops_assignment Package

```bash
# Make predictions on a CSV file
python mlops_assignment/modeling/predict.py \
    --features-path data/processed/lung_cancer_processed.csv \
    --predictions-path data/processed/predictions.csv

# Or using Makefile
make predict
```

#### Using Python API

```python
from src.models.predict import predict

input_data = {
    'Age': 45,
    'Gender': 1,
    'Air Pollution': 5,
    # ... other features
}

result = predict(input_data, return_proba=True)
print(result)
```

## 🌐 Task 3: Web Application

Run the Streamlit web application:

```bash
poetry run streamlit run src/webapp/app.py
```

The app will be available at `http://localhost:8501`.

### Features

- **Lung Cancer Risk Prediction**: 
  - Single patient prediction with interactive form
  - Batch prediction via CSV upload
  - Real-time risk level assessment
  
- **Heart Failure Prediction** (Placeholder):
  - Ready for teammate's model integration
  - Dummy prediction interface

### Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t mlops-app .

# Run container
docker run -p 8501:8501 mlops-app
```

## 🔧 Task 4: MLOps Environment

### DVC Pipeline

Run the complete DVC pipeline:

```bash
dvc repro
```

This executes:
- `prepare`: Feature engineering
- `train`: Model training

### Hydra Configuration

All configuration is managed through `config/config.yaml`. Modify parameters like:
- Data paths
- Model hyperparameters
- Preprocessing options
- MLflow settings

### CI/CD Pipeline

#### Continuous Integration (CI)

The CI pipeline (`.github/workflows/ci.yml`) runs on pull requests:
- Code linting with `flake8`
- Type checking with `mypy`
- Unit tests with `pytest`
- Code formatting check with `black`

#### Continuous Deployment (CD)

The CD pipeline (`.github/workflows/cd.yml`) runs on push to `main`:
- Builds Docker image
- Deploys to Render (or other PaaS)

**Note**: Configure Render API keys in GitHub Secrets:
- `RENDER_API_KEY`
- `RENDER_SERVICE_ID`

### Monitoring

Run data drift detection:

```bash
poetry run python monitoring/drift_detection.py \
    --reference data/raw/lung_cancer.csv \
    --current data/production/new_data.csv \
    --threshold 0.5 \
    --output reports/drift_report.html
```

This will:
- Compare reference (training) data with current (production) data
- Detect statistical drift in features
- Generate an HTML report
- Alert if significant drift is detected

### Testing

Run all tests:

```bash
poetry run pytest tests/ -v
```

Run specific test file:

```bash
poetry run pytest tests/test_predict.py -v
```

## 📝 Usage Examples

### Retrain Model

```bash
# Using DVC
dvc repro train

# Or directly
poetry run python src/models/train.py
```

### Update Configuration

Edit `config/config.yaml` and rerun training:

```yaml
model:
  experiment_name: "lung_cancer_experiment_v2"
  metric: "F1"  # Change from Accuracy to F1
```

### Monitor Production Data

Set up a cron job or scheduled task to run drift detection:

```bash
# Example cron job (runs daily at 2 AM)
0 2 * * * cd /path/to/mlops && poetry run python monitoring/drift_detection.py --current /path/to/production/data.csv
```

## 🐛 Troubleshooting

### Common Issues

1. **DVC data not found**:
   ```bash
   dvc pull
   ```

2. **MLflow model not found**:
   - Ensure you've trained the model first: `python src/models/train.py`
   - Check MLflow tracking URI in `config/config.yaml`

3. **Poetry installation issues**:
   ```bash
   poetry install --no-root
   poetry install
   ```

4. **Import errors**:
   - Ensure you're in the Poetry virtual environment: `poetry shell`
   - Check that all dependencies are installed: `poetry install`

5. **Python version compatibility**:
   - Ensure Python 3.10 or 3.11 is installed
   - Poetry will automatically use the correct Python version from `pyproject.toml`
   - If using Python 3.12+, you'll need to install Python 3.11 and configure Poetry to use it

6. **Feature name errors**:
   - The training script automatically normalizes feature names (spaces to underscores)
   - This is handled automatically - no action needed

## 📚 Documentation

- **PyCaret Documentation**: https://pycaret.readthedocs.io/
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **Streamlit Documentation**: https://docs.streamlit.io/
- **DVC Documentation**: https://dvc.org/doc
- **Hydra Documentation**: https://hydra.cc/


## 📦 Package Structure: mlops_assignment

The project includes an integrated `mlops_assignment` package that provides CLI interfaces for all pipeline components:

### Available Commands

```bash
# Data processing
python mlops_assignment/dataset.py --help

# Feature generation
python mlops_assignment/features.py --help

# Model training (requires Python 3.10/3.11)
python mlops_assignment/modeling/train.py --help

# Predictions (requires Python 3.10/3.11)
python mlops_assignment/modeling/predict.py --help

# Plotting
python mlops_assignment/plots.py --help
```

### Makefile Commands

The project includes a Makefile for convenience:

```bash
make help          # Show all available commands
make requirements   # Install dependencies
make data          # Process dataset
make train         # Train model
make predict       # Make predictions
make test          # Run tests
make lint          # Lint code
make format        # Format code
make clean         # Clean Python cache files
```

## ✅ Project Status

**Current Status:** ✅ **Fully Operational**

- ✅ All dependencies installed and tested
- ✅ Training pipeline working correctly
- ✅ MLflow model registration functional
- ✅ Plot generation working (saves to `reports/figures/`)
- ✅ Web application tested and functional
- ✅ All compatibility issues resolved
- ✅ Integrated `mlops_assignment` package structure
- ✅ CLI interfaces for all pipeline components
- ✅ Makefile with common commands

**Verified:** February 22, 2025

**Integration Status:**
- ✅ Friend's code structure integrated into `mlops_assignment/` package
- ✅ Both `src/` and `mlops_assignment/` structures coexist
- ✅ All modules tested and functional
- ⚠️ Model training/predictions require Python 3.10/3.11 (PyCaret limitation)

---

**Last Updated**: February 22, 2025
