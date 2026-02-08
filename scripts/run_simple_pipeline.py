"""
Simple ML Pipeline Runner (No Prefect)

Runs the complete ML pipeline without Prefect orchestration.
Good for testing and development.

Usage:
    python scripts/run_simple_pipeline.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    load_raw_data,
    prepare_data,
    NetworkTrafficPreprocessor,
    save_processed_data
)
from src.monitoring.drift_detection import DriftDetector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

print("=" * 70)
print("ML TRAINING PIPELINE - SIMPLE VERSION")
print("=" * 70)
print()

# Configuration
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
REPORTS_PATH = PROJECT_ROOT / "reports"
MODELS_PATH = PROJECT_ROOT / "models"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "network_traffic_classification")

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

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("STEP 1: Load Data")
print("-" * 70)

csv_files = list(DATA_RAW_PATH.glob("*.csv"))
if not csv_files:
    print("❌ No CSV files found in data/raw/")
    print("   Run: python scripts/download_data.py")
    sys.exit(1)

df = load_raw_data(csv_files[0])
print(f"✅ Loaded {len(df):,} samples")
print()

# ============================================================================
# STEP 2: Prepare Data Splits
# ============================================================================
print("STEP 2: Prepare Data Splits")
print("-" * 70)

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
    df,
    target_column='Label',
    test_size=0.2,
    val_size=0.1,
    random_state=42
)
print()

# ============================================================================
# STEP 3: Preprocess Data
# ============================================================================
print("STEP 3: Preprocess Data")
print("-" * 70)

preprocessor = NetworkTrafficPreprocessor(
    variance_threshold=0.01,
    scaling_method="standard"
)

X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
X_val_processed, y_val_processed = preprocessor.transform(X_val, y_val)
X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)

# Save processed data
DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
save_processed_data(
    X_train_processed, X_val_processed, X_test_processed,
    y_train_processed, y_val_processed, y_test_processed,
    DATA_PROCESSED_PATH
)

# Save preprocessor
MODELS_PATH.mkdir(parents=True, exist_ok=True)
preprocessor.save(MODELS_PATH / "preprocessor.pkl")

print(f"✅ Preprocessed {len(X_train_processed):,} training samples")
print(f"✅ Selected {len(preprocessor.selected_feature_names)} features")
print()

# ============================================================================
# STEP 4: Train Models
# ============================================================================
print("STEP 4: Train Models")
print("-" * 70)

def train_model(model_name, model, params):
    """Train a model and log to MLflow."""
    print(f"🎯 Training {model_name}...")
    
    with mlflow.start_run(run_name=f"{model_name}_simple_pipeline"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        
        # Train
        model.fit(X_train_processed, y_train_processed)
        
        # Evaluate
        y_val_pred = model.predict(X_val_processed)
        y_test_pred = model.predict(X_test_processed)
        
        metrics = {
            "val_accuracy": accuracy_score(y_val_processed, y_val_pred),
            "val_f1": f1_score(y_val_processed, y_val_pred, average='weighted', zero_division=0),
            "test_accuracy": accuracy_score(y_test_processed, y_test_pred),
            "test_precision": precision_score(y_test_processed, y_test_pred, average='weighted', zero_division=0),
            "test_recall": recall_score(y_test_processed, y_test_pred, average='weighted', zero_division=0),
            "test_f1": f1_score(y_test_processed, y_test_pred, average='weighted', zero_division=0)
        }
        
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"   ✅ Test F1: {metrics['test_f1']:.4f}")
        
        return {"model_name": model_name, "metrics": metrics, "model": model}

# Train models
results = []

# Random Forest
rf_params = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42
}
rf_model = RandomForestClassifier(**rf_params, n_jobs=-1)
results.append(train_model("random_forest", rf_model, rf_params))

# XGBoost
try:
    import xgboost as xgb
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    xgb_model = xgb.XGBClassifier(**xgb_params, n_jobs=-1)
    results.append(train_model("xgboost", xgb_model, xgb_params))
except ImportError:
    print("   ⚠️  XGBoost not available")

# LightGBM
try:
    import lightgbm as lgb
    lgb_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": 42,
        "verbose": -1
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params, n_jobs=-1)
    results.append(train_model("lightgbm", lgb_model, lgb_params))
except ImportError:
    print("   ⚠️  LightGBM not available")

print()

# ============================================================================
# STEP 5: Model Comparison
# ============================================================================
print("STEP 5: Model Comparison")
print("-" * 70)

print(f"\n{'Model':<20} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
print("-" * 70)

for result in results:
    name = result['model_name']
    metrics = result['metrics']
    print(f"{name:<20} "
          f"{metrics['val_accuracy']:<12.4f} "
          f"{metrics['test_accuracy']:<12.4f} "
          f"{metrics['test_f1']:<12.4f}")

best_model = max(results, key=lambda x: x['metrics']['test_f1'])
print(f"\n🏆 Best Model: {best_model['model_name']}")
print(f"   Test F1 Score: {best_model['metrics']['test_f1']:.4f}\n")

# ============================================================================
# STEP 6: Drift Detection
# ============================================================================
print("STEP 6: Drift Detection")
print("-" * 70)

# Create DataFrames
n_features = X_train_processed.shape[1]
feature_names = [f'feature_{i}' for i in range(n_features)]

df_train = pd.DataFrame(X_train_processed, columns=feature_names)
df_train['target'] = y_train_processed

df_test = pd.DataFrame(X_test_processed, columns=feature_names)
df_test['target'] = y_test_processed

# Initialize detector
detector = DriftDetector(drift_threshold=0.5, stattest='ks')

# Detect data drift
print("\n🔍 Checking for data drift...")
data_has_drift, data_metrics = detector.detect_data_drift(df_train, df_test)

print(f"   Dataset drift: {'YES ⚠️' if data_has_drift else 'NO ✅'}")
print(f"   Drift share: {data_metrics['drift_share']:.2%}")

# Detect target drift
print("\n🔍 Checking for target drift...")
target_has_drift, target_metrics = detector.detect_target_drift(
    df_train, df_test, target_column='target'
)

print(f"   Target drift: {'YES ⚠️' if target_has_drift else 'NO ✅'}")

# Save report
REPORTS_PATH.mkdir(parents=True, exist_ok=True)
report_path = detector.save_report(
    df_train, df_test, REPORTS_PATH, "simple_pipeline_drift_report"
)

# Retraining decision
should_retrain, reason = detector.should_retrain(
    data_drift=data_has_drift,
    target_drift=target_has_drift,
    prediction_drift=False,
    drift_share=data_metrics['drift_share'],
    retrain_threshold=0.3
)

print(f"\n🤖 Retraining: {'RECOMMENDED ⚠️' if should_retrain else 'NOT NEEDED ✅'}")
print(f"   {reason}\n")

# ============================================================================
# PIPELINE COMPLETE
# ============================================================================
print("=" * 70)
print("✅ PIPELINE COMPLETE!")
print("=" * 70)
print()
print(f"Best Model: {best_model['model_name']}")
print(f"Test F1 Score: {best_model['metrics']['test_f1']:.4f}")
print(f"Drift Status: {'Retraining recommended' if should_retrain else 'No action needed'}")
print(f"Drift Report: {report_path}")
print()
print("Next steps:")
if should_retrain:
    print("  ⚠️  Drift detected - consider retraining with fresh data")
else:
    print("  ✅ Model is stable - ready for deployment")
print()
print(f"Check MLflow UI: {MLFLOW_TRACKING_URI}")
print()
