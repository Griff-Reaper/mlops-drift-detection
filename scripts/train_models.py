"""
Train Multiple Models and Compare

This script trains multiple models on the preprocessed data:
- Random Forest
- XGBoost
- LightGBM (if installed)

All experiments are logged to MLflow for easy comparison.
"""

import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

print("=" * 70)
print("Model Training and Comparison")
print("=" * 70)
print()

# Configuration
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "network_traffic_classification")

# Check if preprocessed data exists
print("🔍 Checking for preprocessed data...")
required_files = ['X_train.npy', 'X_val.npy', 'X_test.npy', 
                 'y_train.npy', 'y_val.npy', 'y_test.npy']

missing_files = [f for f in required_files if not (DATA_PROCESSED_PATH / f).exists()]
if missing_files:
    print("❌ Preprocessed data not found!")
    print(f"   Missing: {', '.join(missing_files)}")
    print("   Run: python scripts/preprocess_data.py")
    sys.exit(1)

print("✅ All data files found")
print()

# Load preprocessed data
print("📥 Loading preprocessed data...")
X_train = np.load(DATA_PROCESSED_PATH / 'X_train.npy')
X_val = np.load(DATA_PROCESSED_PATH / 'X_val.npy')
X_test = np.load(DATA_PROCESSED_PATH / 'X_test.npy')
y_train = np.load(DATA_PROCESSED_PATH / 'y_train.npy')
y_val = np.load(DATA_PROCESSED_PATH / 'y_val.npy')
y_test = np.load(DATA_PROCESSED_PATH / 'y_test.npy')

print(f"✅ Loaded data:")
print(f"   Train: {X_train.shape}")
print(f"   Val:   {X_val.shape}")
print(f"   Test:  {X_test.shape}")
print()

# Set up MLflow
print("🔧 Connecting to MLflow...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    mlflow.search_experiments()
    print(f"✅ Connected to MLflow at {MLFLOW_TRACKING_URI}")
except:
    print(f"⚠️  Could not connect to MLflow server")
    print(f"   Using local tracking: ./mlruns")
    mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment(EXPERIMENT_NAME)
print()


def train_and_evaluate(model, model_name, params):
    """Train a model and log results to MLflow"""
    
    print(f"🎯 Training {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        
        # Train model
        model.fit(X_train, y_train)
        print(f"   ✅ Training complete")
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
            "val_recall": recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
            "val_f1": f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        }
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
            "test_recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
            "test_f1": f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        }
        
        metrics.update(test_metrics)
        
        # Log all metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Print results
        print(f"\n   📊 {model_name} Performance:")
        print(f"      Validation Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"      Test Accuracy:       {metrics['test_accuracy']:.4f}")
        print(f"      Test F1 Score:       {metrics['test_f1']:.4f}")
        
        run = mlflow.active_run()
        if run:
            print(f"      MLflow Run ID: {run.info.run_id}")
        
        print()
        
        return metrics


# Define models to train
print("=" * 70)
print("Training Models")
print("=" * 70)
print()

results = {}

# 1. Random Forest
print("1️⃣  Random Forest")
print("-" * 70)
rf_params = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1
}

rf_model = RandomForestClassifier(**rf_params)
results["Random Forest"] = train_and_evaluate(rf_model, "random_forest", rf_params)

# 2. XGBoost
print("2️⃣  XGBoost")
print("-" * 70)
try:
    import xgboost as xgb
    
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    results["XGBoost"] = train_and_evaluate(xgb_model, "xgboost", xgb_params)
    
except ImportError:
    print("   ⚠️  XGBoost not installed, skipping...")
    print("   Install with: pip install xgboost")
    print()

# 3. LightGBM (bonus)
print("3️⃣  LightGBM")
print("-" * 70)
try:
    import lightgbm as lgb
    
    lgb_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    results["LightGBM"] = train_and_evaluate(lgb_model, "lightgbm", lgb_params)
    
except ImportError:
    print("   ⚠️  LightGBM not installed, skipping...")
    print("   Install with: pip install lightgbm")
    print()

# Print comparison
print("=" * 70)
print("Model Comparison")
print("=" * 70)
print()

print(f"{'Model':<20} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
print("-" * 70)

for model_name, metrics in results.items():
    print(f"{model_name:<20} "
          f"{metrics['val_accuracy']:<12.4f} "
          f"{metrics['test_accuracy']:<12.4f} "
          f"{metrics['test_f1']:<12.4f}")

print()

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['test_f1'])
print(f"🏆 Best Model: {best_model[0]}")
print(f"   Test F1 Score: {best_model[1]['test_f1']:.4f}")
print()

print("=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print()
print("Next steps:")
print(f"  1. Check MLflow UI: {MLFLOW_TRACKING_URI}")
print("  2. Compare models in the 'Runs' tab")
print("  3. Select the best model for deployment")
print()
print("To compare models:")
print("  - Click on the experiment name")
print("  - Check the boxes next to runs you want to compare")
print("  - Click 'Compare' button")
print()
