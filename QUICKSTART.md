# Quick Start Guide

This guide will help you get started with the MLOps assignment project quickly.

## Prerequisites Check

Before starting, ensure you have:
- Python 3.9+ installed
- Git installed
- Access to the repository

## Step 1: Install Poetry

```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Linux/Mac
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to PATH if needed.

## Step 2: Clone and Setup

```bash
# Navigate to project directory
cd mlops

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Step 3: Get Data

```bash
# If using DVC (recommended)
dvc pull

# Or ensure data is in data/raw/lung_cancer.csv
```

## Step 4: Run EDA (Task 1)

```bash
# Start Jupyter
poetry run jupyter notebook

# Open notebooks/eda_lung_cancer.ipynb and run all cells
```

## Step 5: Train Model (Task 2)

```bash
# From project root
poetry run python src/models/train.py

# View MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

## Step 6: Run Web App (Task 3)

```bash
# From project root
poetry run streamlit run src/webapp/app.py
```

Visit `http://localhost:8501` in your browser.

## Step 7: Test Everything

```bash
# Run tests
poetry run pytest tests/ -v

# Run drift detection (requires production data)
poetry run python monitoring/drift_detection.py --current data/production/new_data.csv
```

## Common Issues

### Issue: Module not found
**Solution**: Ensure you're in the Poetry virtual environment (`poetry shell`)

### Issue: Data file not found
**Solution**: Run `dvc pull` or ensure CSV is in `data/raw/lung_cancer.csv`

### Issue: MLflow model not found
**Solution**: Train the model first: `python src/models/train.py`

### Issue: Import errors
**Solution**: Install dependencies: `poetry install`

## Next Steps

1. Review the full [README.md](README.md) for detailed documentation
2. Check [presentation_outline.md](presentation_outline.md) for demo preparation
3. Customize `config/config.yaml` for your needs
4. Integrate teammate's heart failure model when ready

## Getting Help

- Check the README.md for detailed instructions
- Review error messages carefully
- Ensure all dependencies are installed
- Verify data paths in configuration

Good luck! 🚀
