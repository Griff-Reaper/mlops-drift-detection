"""
Run ML Pipeline

Simple script to execute the complete ML pipeline using Prefect.

This orchestrates:
1. Data preprocessing
2. Model training (Random Forest, XGBoost, LightGBM)
3. Model comparison
4. Drift detection
5. Retraining recommendations

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.ml_pipeline import ml_training_pipeline

print("=" * 70)
print("Starting ML Pipeline with Prefect Orchestration")
print("=" * 70)
print()
print("This will:")
print("  1. Load and preprocess data")
print("  2. Train multiple ML models")
print("  3. Compare model performance")
print("  4. Check for data drift")
print("  5. Generate recommendations")
print()
print("All experiments will be logged to MLflow.")
print("Make sure MLflow server is running: mlflow server --host 127.0.0.1 --port 5000")
print()
print("=" * 70)
print()

# Run the pipeline
try:
    result = ml_training_pipeline()
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS!")
    print("=" * 70)
    print()
    print(f"Best model: {result['best_model']['model_name']}")
    print(f"Test F1 Score: {result['best_model']['metrics']['test_f1']:.4f}")
    print()
    print("Check MLflow UI for detailed results:")
    print("  http://localhost:5000")
    print()
    
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ PIPELINE FAILED")
    print("=" * 70)
    print()
    print(f"Error: {str(e)}")
    print()
    print("Troubleshooting:")
    print("  1. Make sure MLflow server is running")
    print("  2. Check that data files exist in data/raw/")
    print("  3. Verify all dependencies are installed")
    print()
    sys.exit(1)
