# CookieCutter Template Usage

This project uses CookieCutter to generate standardized MLOps project structures.

## Installation

First, install cookiecutter if you haven't already:

```bash
pip install cookiecutter
```

## Using This Template

To create a new project from this template:

```bash
cookiecutter .
```

You'll be prompted to enter values for:
- `project_name`: Name of your project
- `author_name`: Your name or organization
- `description`: Brief description of the project
- `open_source_license`: Choose a license
- `python_version`: Python version to use
- `use_poetry`: Whether to use Poetry for dependency management (y/n)
- `use_docker`: Whether to include Docker configuration (y/n)
- `use_mlflow`: Whether to use MLflow for experiment tracking (y/n)
- `use_hydra`: Whether to use Hydra for configuration management (y/n)

## Project Structure

The generated project will follow this structure:

```
{{ cookiecutter.repo_name }}/
├── data/
│   ├── raw/           # Raw, immutable data
│   ├── processed/     # Processed data
│   ├── external/      # External data sources
│   └── interim/       # Intermediate data
├── notebooks/         # Jupyter notebooks
├── src/               # Source code
│   └── models/        # Model definitions
├── tests/             # Test files
├── reports/           # Generated reports
│   └── figures/       # Generated figures
├── models/            # Trained models
├── outputs/           # Training outputs and logs
├── config/            # Configuration files
└── mlruns/            # MLflow experiment tracking
```

## Customization

Edit `cookiecutter.json` to customize the template variables and defaults.
