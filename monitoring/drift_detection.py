"""
Data drift detection script using Evidently AI.
Monitors data drift between reference (training) data and new production data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    print("Warning: Evidently not installed. Install with: pip install evidently")
    sys.exit(1)


def load_reference_data(data_path: str) -> pd.DataFrame:
    """
    Load reference (training) data.
    
    Args:
        data_path: Path to reference data CSV
        
    Returns:
        DataFrame with reference data
    """
    df = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    cols_to_drop = ['index', 'Patient Id', 'Level']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    return df


def load_current_data(data_path: str) -> pd.DataFrame:
    """
    Load current (production) data.
    
    Args:
        data_path: Path to current data CSV
        
    Returns:
        DataFrame with current data
    """
    df = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    cols_to_drop = ['index', 'Patient Id', 'Level']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    return df


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    drift_threshold: float = 0.5,
    output_path: Optional[str] = None
) -> dict:
    """
    Detect data drift between reference and current data.
    
    Args:
        reference_data: Reference (training) dataset
        current_data: Current (production) dataset
        drift_threshold: Threshold for drift detection (0-1)
        output_path: Optional path to save drift report
        
    Returns:
        Dictionary with drift detection results
    """
    # Define column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = None  # No target in drift detection
    column_mapping.prediction = None
    
    # Create drift report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    # Get drift results
    drift_results = drift_report.as_dict()
    
    # Extract drift metrics
    drift_metrics = drift_results.get('metrics', [])
    drift_detected = False
    drifted_features = []
    
    for metric in drift_metrics:
        if metric.get('metric') == 'DatasetDriftMetric':
            drift_detected = metric.get('result', {}).get('dataset_drift', False)
        elif 'ColumnDriftMetric' in str(metric.get('metric', '')):
            feature_name = metric.get('result', {}).get('column_name', '')
            drift_score = metric.get('result', {}).get('drift_score', 0)
            if drift_score > drift_threshold:
                drifted_features.append({
                    'feature': feature_name,
                    'drift_score': drift_score
                })
    
    # Save report if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        drift_report.save_html(output_path)
        print(f"Drift report saved to: {output_path}")
    
    # Print results
    print("=" * 60)
    print("DATA DRIFT DETECTION RESULTS")
    print("=" * 60)
    print(f"\nDataset Drift Detected: {drift_detected}")
    print(f"Drift Threshold: {drift_threshold}")
    print(f"\nNumber of Drifted Features: {len(drifted_features)}")
    
    if drifted_features:
        print("\nDrifted Features:")
        for feature_info in drifted_features:
            print(f"  - {feature_info['feature']}: {feature_info['drift_score']:.4f}")
    else:
        print("\nNo significant drift detected in individual features.")
    
    print("=" * 60)
    
    # Return results
    result = {
        'drift_detected': drift_detected,
        'drifted_features': drifted_features,
        'drift_threshold': drift_threshold,
        'reference_samples': len(reference_data),
        'current_samples': len(current_data)
    }
    
    # Trigger alert if drift detected
    if drift_detected or len(drifted_features) > 0:
        print("\n⚠️  WARNING: Data drift detected!")
        print("Consider retraining the model with new data.")
        # In production, you could trigger a retraining job here
        # e.g., via GitHub Actions API, webhook, or message queue
    
    return result


def main():
    """Main function to run drift detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect data drift in production data")
    parser.add_argument(
        "--reference",
        type=str,
        default="data/raw/lung_cancer.csv",
        help="Path to reference (training) data"
    )
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to current (production) data"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Drift detection threshold (0-1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/drift_report.html",
        help="Path to save drift report HTML"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading reference data from: {args.reference}")
    reference_data = load_reference_data(args.reference)
    print(f"Reference data shape: {reference_data.shape}")
    
    print(f"\nLoading current data from: {args.current}")
    current_data = load_current_data(args.current)
    print(f"Current data shape: {current_data.shape}")
    
    # Detect drift
    results = detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        drift_threshold=args.threshold,
        output_path=args.output
    )
    
    return results


if __name__ == "__main__":
    main()
