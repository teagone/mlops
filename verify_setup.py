"""
Verification script to check project setup and correctness.
"""

import sys
from pathlib import Path

def check_file_exists(filepath):
    """Check if a file exists."""
    path = Path(filepath)
    exists = path.exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status} {filepath}")
    return exists

def check_data_file():
    """Check data file structure."""
    print("\n=== Checking Data File ===")
    try:
        import pandas as pd
        df = pd.read_csv('data/raw/lung_cancer.csv')
        print(f"  [OK] Data file loaded successfully")
        print(f"  [OK] Shape: {df.shape}")
        print(f"  [OK] Columns: {len(df.columns)} columns")
        print(f"  [OK] Target column 'Level' exists: {'Level' in df.columns}")
        if 'Level' in df.columns:
            print(f"  [OK] Target distribution:\n{df['Level'].value_counts().to_string()}")
        return True
    except Exception as e:
        print(f"  [ERROR] Error loading data: {e}")
        return False

def check_imports():
    """Check if required packages can be imported."""
    print("\n=== Checking Package Imports ===")
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'mlflow': 'mlflow',
        'pycaret': 'pycaret',
        'streamlit': 'streamlit',
        'hydra': 'hydra-core',
        'pytest': 'pytest'
    }
    
    results = {}
    for module, package_name in packages.items():
        try:
            __import__(module)
            print(f"  [OK] {package_name} imported successfully")
            results[package_name] = True
        except ImportError:
            print(f"  [MISSING] {package_name} NOT installed")
            results[package_name] = False
    
    return results

def check_code_structure():
    """Check code structure and syntax."""
    print("\n=== Checking Code Structure ===")
    files_to_check = [
        'src/models/train.py',
        'src/models/predict.py',
        'src/models/utils.py',
        'src/features/build_features.py',
        'src/webapp/app.py',
        'config/config.yaml'
    ]
    
    all_exist = True
    for filepath in files_to_check:
        if not check_file_exists(filepath):
            all_exist = False
    
    # Check syntax
    print("\n  Checking Python syntax...")
    import py_compile
    python_files = [f for f in files_to_check if f.endswith('.py')]
    syntax_ok = True
    for filepath in python_files:
        try:
            py_compile.compile(filepath, doraise=True)
            print(f"    [OK] {filepath} - syntax OK")
        except py_compile.PyCompileError as e:
            print(f"    [ERROR] {filepath} - syntax error: {e}")
            syntax_ok = False
    
    return all_exist and syntax_ok

def check_model_files():
    """Check if model files exist."""
    print("\n=== Checking Model Files ===")
    model_files = [
        'models/lung_cancer_model.pkl',
    ]
    
    all_exist = True
    for filepath in model_files:
        if not check_file_exists(filepath):
            all_exist = False
    
    return all_exist

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("MLOps Project Verification")
    print("=" * 60)
    
    results = {
        'data_file': check_data_file(),
        'code_structure': check_code_structure(),
        'model_files': check_model_files(),
        'imports': check_imports()
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Data file: {'[PASS]' if results['data_file'] else '[FAIL]'}")
    print(f"Code structure: {'[PASS]' if results['code_structure'] else '[FAIL]'}")
    print(f"Model files: {'[PASS]' if results['model_files'] else '[FAIL]'}")
    
    installed_packages = sum(1 for v in results['imports'].values() if v)
    total_packages = len(results['imports'])
    print(f"Packages installed: {installed_packages}/{total_packages}")
    
    if installed_packages < total_packages:
        print("\n[WARNING] Some packages are missing. Install with:")
        print("   pip install -r requirements.txt")
        print("   OR")
        print("   poetry install")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
