"""
ML Pipeline Orchestration with Prefect

This Prefect flow orchestrates the complete ML pipeline:
1. Data preprocessing
2. Model training (multiple models in parallel)
3. Model comparison and selection
4. Drift detection
5. Retraining recommendations

Run with: python src/pipelines/ml_pipeline.py
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prefect import flow, task
import mlflow

from src.data.preprocessing import (
    load_raw_data,
    prepare_data,
    NetworkTrafficPreprocessor,
    save_processed_data
)
from src.monitoring.drift_detection import DriftDetector

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()


@task(name="Load Raw Data", log_prints=True)
def load_data_task(data_path: Path) -> pd.DataFrame:
    """Task: Load raw network traffic data."""
    print(f"📥 Loading data from {data_path}")
    
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    
    df = load_raw_data(csv_files[0])
    print(f"✅ Loaded {len(df):,} samples")
    return df


@task(name="Prepare Data Splits", log_prints=True)
def prepare_data_splits_task(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Task: Split data into train/val/test sets."""
    print("🔧 Splitting data into train/val/test...")
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        df,
        target_column='Label',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print("✅ Data split complete")
    return X_train, X_val, X_test, y_train, y_val, y_test


@task(name="Preprocess Data", log_prints=True)
def preprocess_data_task(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: Path
) -> NetworkTrafficPreprocessor:
    """Task: Preprocess features and save processed data."""
    print("🔧 Preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = NetworkTrafficPreprocessor(
        variance_threshold=0.01,
        scaling_method="standard"
    )
    
    # Fit and transform
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_val_processed, y_val_processed = preprocessor.transform(X_val, y_val)
    X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
    
    # Save processed data
    output_dir.mkdir(parents=True, exist_ok=True)
    save_processed_data(
        X_train_processed, X_val_processed, X_test_processed,
        y_train_processed, y_val_processed, y_test_processed,
        output_dir
    )
    
    # Save preprocessor
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(models_dir / "preprocessor.pkl")
    
    print(f"✅ Preprocessed {len(X_train_processed)} training samples")
    print(f"✅ Selected {len(preprocessor.selected_feature_names)} features")
    
    return preprocessor


@task(name="Train Model", log_prints=True)
def train_model_task(
    model_name: str,
    model,
    params: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str
) -> Dict:
    """Task: Train a single model and log to MLflow."""
    print(f"🎯 Training {model_name}...")
    
    # Set up MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{model_name}_pipeline"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate on validation
        y_val_pred = model.predict(X_val)
        val_metrics = {
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
            "val_recall": recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
            "val_f1": f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        }
        
        # Evaluate on test
        y_test_pred = model.predict(X_test)
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
            "test_recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
            "test_f1": f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        }
        
        # Combine metrics
        all_metrics = {**val_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ {model_name} - Test F1: {test_metrics['test_f1']:.4f}")
        
        return {
            "model_name": model_name,
            "metrics": all_metrics,
            "model": model
        }


@task(name="Compare Models", log_prints=True)
def compare_models_task(results: list) -> Dict:
    """Task: Compare model results and select best."""
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    
    print(f"\n{'Model':<20} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 70)
    
    for result in results:
        name = result['model_name']
        metrics = result['metrics']
        print(f"{name:<20} "
              f"{metrics['val_accuracy']:<12.4f} "
              f"{metrics['test_accuracy']:<12.4f} "
              f"{metrics['test_f1']:<12.4f}")
    
    # Find best model
    best_result = max(results, key=lambda x: x['metrics']['test_f1'])
    
    print(f"\n🏆 Best Model: {best_result['model_name']}")
    print(f"   Test F1 Score: {best_result['metrics']['test_f1']:.4f}\n")
    
    return best_result


@task(name="Check Drift", log_prints=True)
def check_drift_task(
    data_processed_path: Path,
    reports_path: Path
) -> Tuple[bool, Dict]:
    """Task: Check for data and target drift."""
    print("\n" + "=" * 70)
    print("Drift Detection")
    print("=" * 70)
    
    # Load processed data
    X_train = np.load(data_processed_path / 'X_train.npy')
    X_test = np.load(data_processed_path / 'X_test.npy')
    y_train = np.load(data_processed_path / 'y_train.npy')
    y_test = np.load(data_processed_path / 'y_test.npy')
    
    # Create DataFrames
    n_features = X_train.shape[1]
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['target'] = y_train
    
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['target'] = y_test
    
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
    reports_path.mkdir(parents=True, exist_ok=True)
    report_path = detector.save_report(
        df_train, df_test, reports_path, "pipeline_drift_report"
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
    
    drift_info = {
        'data_drift': data_has_drift,
        'target_drift': target_has_drift,
        'drift_share': data_metrics['drift_share'],
        'should_retrain': should_retrain,
        'reason': reason,
        'report_path': str(report_path)
    }
    
    return should_retrain, drift_info


@flow(
    name="ML Training Pipeline",
    log_prints=True
)
def ml_training_pipeline():
    """
    Complete ML pipeline orchestrated by Prefect.
    
    Pipeline Steps:
    1. Load raw data
    2. Split into train/val/test
    3. Preprocess and save
    4. Train multiple models in parallel
    5. Compare and select best model
    6. Check for drift
    7. Recommend next steps
    """
    
    print("\n" + "=" * 70)
    print("ML TRAINING PIPELINE - PREFECT ORCHESTRATION")
    print("=" * 70)
    print()
    
    # Configuration
    data_raw_path = PROJECT_ROOT / "data" / "raw"
    data_processed_path = PROJECT_ROOT / "data" / "processed"
    reports_path = PROJECT_ROOT / "reports"
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "network_traffic_classification")
    
    # Step 1: Load data
    print("STEP 1: Load Data")
    print("-" * 70)
    df = load_data_task(data_raw_path)
    
    # Step 2: Prepare splits
    print("\nSTEP 2: Prepare Data Splits")
    print("-" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits_task(df)
    
    # Step 3: Preprocess
    print("\nSTEP 3: Preprocess Data")
    print("-" * 70)
    preprocessor = preprocess_data_task(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        data_processed_path
    )
    
    # Load processed data for training
    X_train_processed = np.load(data_processed_path / 'X_train.npy')
    X_val_processed = np.load(data_processed_path / 'X_val.npy')
    X_test_processed = np.load(data_processed_path / 'X_test.npy')
    y_train_processed = np.load(data_processed_path / 'y_train.npy')
    y_val_processed = np.load(data_processed_path / 'y_val.npy')
    y_test_processed = np.load(data_processed_path / 'y_test.npy')
    
    # Step 4: Train models
    print("\nSTEP 4: Train Models")
    print("-" * 70)
    
    # Define models
    models = [
        {
            "name": "random_forest",
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            }
        }
    ]
    
    # Try to add XGBoost and LightGBM
    try:
        import xgboost as xgb
        models.append({
            "name": "xgboost",
            "model": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            }
        })
    except ImportError:
        print("   ⚠️  XGBoost not available")
    
    try:
        import lightgbm as lgb
        models.append({
            "name": "lightgbm",
            "model": lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "learning_rate": 0.1,
                "num_leaves": 31
            }
        })
    except ImportError:
        print("   ⚠️  LightGBM not available")
    
    # Train all models
    results = []
    for model_config in models:
        result = train_model_task(
            model_name=model_config["name"],
            model=model_config["model"],
            params=model_config["params"],
            X_train=X_train_processed,
            y_train=y_train_processed,
            X_val=X_val_processed,
            y_val=y_val_processed,
            X_test=X_test_processed,
            y_test=y_test_processed,
            experiment_name=experiment_name
        )
        results.append(result)
    
    # Step 5: Compare models
    print("\nSTEP 5: Model Comparison")
    print("-" * 70)
    best_model = compare_models_task(results)
    
    # Step 6: Check drift
    print("\nSTEP 6: Drift Detection")
    print("-" * 70)
    should_retrain, drift_info = check_drift_task(data_processed_path, reports_path)
    
    # Step 7: Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print()
    print(f"✅ Best Model: {best_model['model_name']}")
    print(f"✅ Test F1 Score: {best_model['metrics']['test_f1']:.4f}")
    print(f"✅ Drift Detection: {'Retraining recommended' if should_retrain else 'No action needed'}")
    print(f"✅ Drift Report: {drift_info['report_path']}")
    print()
    print("Next steps:")
    if should_retrain:
        print("  ⚠️  Drift detected - consider retraining with fresh data")
    else:
        print("  ✅ Model is stable - deploy to production")
    print()
    
    return {
        "best_model": best_model,
        "drift_info": drift_info,
        "pipeline_status": "success"
    }


if __name__ == "__main__":
    """
    Run the complete ML pipeline.
    
    This orchestrates:
    - Data loading and preprocessing
    - Model training and comparison
    - Drift detection and reporting
    """
    
    # Run the pipeline
    result = ml_training_pipeline()
    
    print("\n" + "=" * 70)
    print("🎉 Pipeline execution complete!")
    print("=" * 70)
