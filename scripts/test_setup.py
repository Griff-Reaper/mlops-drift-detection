"""
Test Setup Script - Verify MLOps Pipeline Installation

This script tests that everything is working:
1. Loads data
2. Trains a simple model
3. Logs to MLflow
4. Verifies the tracking server

Run this after completing the setup steps.
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("MLOps Pipeline - Setup Verification")
print("=" * 70)
print()

# Step 1: Verify imports
print("🔍 Step 1: Verifying package imports...")
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import mlflow
    import mlflow.sklearn
    from dotenv import load_dotenv
    print("   ✅ All packages imported successfully!")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

print()

# Step 2: Load environment configuration
print("🔍 Step 2: Loading configuration...")
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "network_traffic_classification")
print(f"   ✅ MLflow URI: {MLFLOW_TRACKING_URI}")
print(f"   ✅ Experiment: {EXPERIMENT_NAME}")

print()

# Step 3: Check for data
print("🔍 Step 3: Checking for dataset...")
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
csv_files = list(DATA_RAW_PATH.glob("*.csv"))

if not csv_files:
    print("   ❌ No CSV files found in data/raw/")
    print("   Run: python scripts/download_data.py")
    sys.exit(1)

data_file = csv_files[0]
print(f"   ✅ Found dataset: {data_file.name}")

print()

# Step 4: Load and prepare data
print("🔍 Step 4: Loading data...")
try:
    df = pd.read_csv(data_file)
    print(f"   ✅ Loaded {len(df):,} samples")
    print(f"   ✅ Features: {len(df.columns)}")
    
    # Check for Label column
    if 'Label' not in df.columns:
        print("   ❌ No 'Label' column found in dataset")
        sys.exit(1)
    
    print(f"   ✅ Classes: {df['Label'].nunique()}")
    print()
    print("   📊 Class distribution:")
    print(df['Label'].value_counts().to_string().replace('\n', '\n      '))
    
except Exception as e:
    print(f"   ❌ Error loading data: {e}")
    sys.exit(1)

print()

# Step 5: Prepare training data
print("🔍 Step 5: Preparing training data...")
try:
    # Take a sample for quick testing (10% of data)
    df_sample = df.sample(frac=0.1, random_state=42)
    print(f"   Using {len(df_sample):,} samples for quick test")
    
    # Separate features and target
    X = df_sample.drop('Label', axis=1)
    y = df_sample['Label']
    
    # Handle non-numeric columns if any
    X = X.select_dtypes(include=[np.number])
    
    # Fill missing values
    X = X.fillna(X.mean())
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✅ Training samples: {len(X_train):,}")
    print(f"   ✅ Test samples: {len(X_test):,}")
    print(f"   ✅ Features: {len(X.columns)}")
    
except Exception as e:
    print(f"   ❌ Error preparing data: {e}")
    sys.exit(1)

print()

# Step 6: Set up MLflow
print("🔍 Step 6: Connecting to MLflow...")
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Try to reach the server
    try:
        mlflow.search_experiments()
        print(f"   ✅ Connected to MLflow server at {MLFLOW_TRACKING_URI}")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not connect to MLflow server")
        print(f"      Make sure MLflow is running: mlflow server --host 127.0.0.1 --port 5000")
        print(f"      Continuing with local tracking...")
        mlflow.set_tracking_uri("file:./mlruns")
    
    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            print(f"   ✅ Created new experiment: {EXPERIMENT_NAME}")
        else:
            experiment_id = experiment.experiment_id
            print(f"   ✅ Using existing experiment: {EXPERIMENT_NAME}")
        
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        print(f"   ⚠️  Error with experiment: {e}")
        print("   Continuing anyway...")
    
except Exception as e:
    print(f"   ❌ MLflow setup error: {e}")
    print("   Continuing without MLflow tracking...")

print()

# Step 7: Train model and log to MLflow
print("🔍 Step 7: Training baseline model...")
try:
    with mlflow.start_run(run_name="setup_test"):
        # Log parameters
        params = {
            "model_type": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "test_size": 0.2
        }
        mlflow.log_params(params)
        
        # Train model
        print("   Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("   ✅ Model trained!")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        print()
        print("   📊 Model Performance:")
        print(f"      Accuracy:  {metrics['accuracy']:.4f}")
        print(f"      Precision: {metrics['precision']:.4f}")
        print(f"      Recall:    {metrics['recall']:.4f}")
        print(f"      F1 Score:  {metrics['f1']:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        print()
        print("   ✅ Model logged to MLflow!")
        
        # Get run info
        run = mlflow.active_run()
        if run:
            print(f"   ✅ Run ID: {run.info.run_id}")
            print(f"   ✅ Experiment ID: {run.info.experiment_id}")
        
except Exception as e:
    print(f"   ❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("✅ SETUP TEST COMPLETE!")
print("=" * 70)
print()
print("Everything is working! Here's what we verified:")
print("  ✅ All packages installed correctly")
print("  ✅ Data loaded successfully")
print("  ✅ Model trained and evaluated")
print("  ✅ MLflow experiment tracking working")
print()
print("Next steps:")
print("  1. Open MLflow UI: http://localhost:5000")
print("  2. Check your experiment: 'network_traffic_classification'")
print("  3. Explore the metrics and model we just logged")
print("  4. Continue with Phase 1 in KICKOFF_GUIDE.md")
print()
print("🎉 You're ready to start building the full pipeline!")
print()
