"""
Check for Drift

This script checks for drift between training data and test data:
1. Load baseline (training) and current (test) data
2. Run drift detection
3. Generate HTML drift report
4. Recommend whether to retrain

Run this script to monitor your production model!
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.drift_detection import DriftDetector

print("=" * 70)
print("Drift Detection System")
print("=" * 70)
print()

# Configuration
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Check if processed data exists
print("🔍 Checking for processed data...")
required_files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']
missing_files = [f for f in required_files if not (DATA_PROCESSED_PATH / f).exists()]

if missing_files:
    print("❌ Processed data not found!")
    print(f"   Missing: {', '.join(missing_files)}")
    print("   Run: python scripts/preprocess_data.py")
    sys.exit(1)

print("✅ All data files found")
print()

# Load data
print("📥 Loading data...")
X_train = np.load(DATA_PROCESSED_PATH / 'X_train.npy')
X_test = np.load(DATA_PROCESSED_PATH / 'X_test.npy')
y_train = np.load(DATA_PROCESSED_PATH / 'y_train.npy')
y_test = np.load(DATA_PROCESSED_PATH / 'y_test.npy')

print(f"✅ Loaded data:")
print(f"   Reference (train): {X_train.shape}")
print(f"   Current (test):    {X_test.shape}")
print()

# Convert to DataFrames for Evidently
print("🔧 Preparing data for drift detection...")

# Create feature names
n_features = X_train.shape[1]
feature_names = [f'feature_{i}' for i in range(n_features)]

# Create DataFrames
df_train = pd.DataFrame(X_train, columns=feature_names)
df_train['target'] = y_train

df_test = pd.DataFrame(X_test, columns=feature_names)
df_test['target'] = y_test

print(f"✅ Data prepared with {n_features} features")
print()

# Initialize drift detector
print("🔧 Initializing drift detector...")
detector = DriftDetector(
    drift_threshold=0.5,  # 50% of features must drift to trigger dataset drift
    stattest='ks',        # Kolmogorov-Smirnov test
    confidence_level=0.95
)
print("✅ Drift detector initialized")
print()

# Detect data drift
print("=" * 70)
print("1️⃣  Data Drift Detection")
print("=" * 70)
print()

data_has_drift, data_metrics = detector.detect_data_drift(
    reference_data=df_train,
    current_data=df_test
)

print()
print("📊 Data Drift Results:")
print(f"   Dataset drift detected: {data_has_drift}")
print(f"   Drift share: {data_metrics['drift_share']:.2%}")
print(f"   Drifted features: {data_metrics['n_drifted_features']}/{data_metrics['n_features']}")
print()

# Save data drift report
data_report_path = detector.save_report(
    df_train,
    df_test,
    REPORTS_PATH,
    "data_drift_report"
)
print()

# Detect target drift
print("=" * 70)
print("2️⃣  Target Drift Detection")
print("=" * 70)
print()

target_has_drift, target_metrics = detector.detect_target_drift(
    reference_data=df_train,
    current_data=df_test,
    target_column='target'
)

print()
print("📊 Target Drift Results:")
print(f"   Target drift detected: {target_has_drift}")
if 'drift_score' in target_metrics:
    print(f"   Drift score: {target_metrics['drift_score']:.4f}")
print()

# Determine if retraining is needed
print("=" * 70)
print("3️⃣  Retraining Recommendation")
print("=" * 70)
print()

should_retrain, reason = detector.should_retrain(
    data_drift=data_has_drift,
    target_drift=target_has_drift,
    prediction_drift=False,  # We don't have predictions in this example
    drift_share=data_metrics['drift_share'],
    retrain_threshold=0.3  # Retrain if >30% of features drift
)

print(f"🤖 Retraining Status: {'⚠️  RECOMMENDED' if should_retrain else '✅ NOT NEEDED'}")
print(f"   Reason: {reason}")
print()

# Summary
print("=" * 70)
print("Summary")
print("=" * 70)
print()

print("📄 Drift Report Generated:")
print(f"   - Data drift: {data_report_path}")
print()

print("🔍 Key Findings:")
print(f"   - Data drift: {'YES ⚠️' if data_has_drift else 'NO ✅'}")
print(f"   - Target drift: {'YES ⚠️' if target_has_drift else 'NO ✅'}")
print(f"   - Retrain model: {'YES ⚠️' if should_retrain else 'NO ✅'}")
print()

print("=" * 70)
print("✅ DRIFT DETECTION COMPLETE!")
print("=" * 70)
print()

print("Next steps:")
print(f"  1. Open drift reports in your browser:")
print(f"     {data_report_path}")
print(f"     {target_report_path}")
print()
print("  2. Review the visualizations:")
print("     - Feature distributions")
print("     - Drift scores")
print("     - Statistical tests")
print()

if should_retrain:
    print("  3. Retrain your model:")
    print("     python scripts/train_models.py")
    print()

print("💡 Tip: In production, you would:")
print("   - Schedule this to run daily/weekly")
print("   - Send alerts when drift is detected")
print("   - Automatically trigger retraining")
print("   - Track drift metrics over time")
print()
