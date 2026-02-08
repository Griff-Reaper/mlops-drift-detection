"""
Simulate Data Drift

This script creates "drifted" data to test the drift detection system.
It artificially modifies the test data to simulate what happens when
production data changes over time.

Run this to see drift detection in action!
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
print("Simulate Data Drift - Test Drift Detection System")
print("=" * 70)
print()

# Configuration
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Load data
print("📥 Loading baseline data...")
X_train = np.load(DATA_PROCESSED_PATH / 'X_train.npy')
X_test = np.load(DATA_PROCESSED_PATH / 'X_test.npy')
y_train = np.load(DATA_PROCESSED_PATH / 'y_train.npy')
y_test = np.load(DATA_PROCESSED_PATH / 'y_test.npy')

print(f"✅ Loaded:")
print(f"   Train: {X_train.shape}")
print(f"   Test:  {X_test.shape}")
print()

# Create feature names
n_features = X_train.shape[1]
feature_names = [f'feature_{i}' for i in range(n_features)]

# Convert to DataFrames
df_train = pd.DataFrame(X_train, columns=feature_names)
df_train['target'] = y_train

df_test_original = pd.DataFrame(X_test, columns=feature_names)
df_test_original['target'] = y_test

print("=" * 70)
print("Creating Drifted Data")
print("=" * 70)
print()

print("🔧 Applying drift transformations...")
print()

# Create drifted version of test data
X_test_drifted = X_test.copy()

# Drift Scenario 1: Feature scaling drift (common in production)
# Simulate scenario where feature scales change over time
print("1️⃣  Applying feature scaling drift...")
print("   Simulating: New data has different feature scales")

# Scale 30% of features by 1.5x
n_drift_features = int(0.3 * n_features)
drift_features = np.random.choice(n_features, size=n_drift_features, replace=False)

for feat_idx in drift_features:
    X_test_drifted[:, feat_idx] *= 1.5

print(f"   Modified {n_drift_features} features ({n_drift_features/n_features:.0%})")
print()

# Drift Scenario 2: Add noise to simulate data quality issues
print("2️⃣  Adding noise drift...")
print("   Simulating: Data quality degradation")

noise_features = np.random.choice(n_features, size=int(0.2 * n_features), replace=False)
for feat_idx in noise_features:
    noise = np.random.normal(0, 0.1 * np.std(X_test_drifted[:, feat_idx]), size=len(X_test_drifted))
    X_test_drifted[:, feat_idx] += noise

print(f"   Added noise to {len(noise_features)} features")
print()

# Drift Scenario 3: Shift means (distribution shift)
print("3️⃣  Applying distribution shift...")
print("   Simulating: Population distribution change")

shift_features = np.random.choice(n_features, size=int(0.15 * n_features), replace=False)
for feat_idx in shift_features:
    shift = 0.2 * np.mean(X_test_drifted[:, feat_idx])
    X_test_drifted[:, feat_idx] += shift

print(f"   Shifted distribution of {len(shift_features)} features")
print()

# Drift Scenario 4: Target drift (class imbalance change)
print("4️⃣  Simulating target drift...")
print("   Original class distribution:")

unique, counts = np.unique(y_test, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"      Class {cls}: {count} samples ({count/len(y_test):.1%})")

# Create imbalanced test set by oversampling one class
y_test_drifted = y_test.copy()
majority_class = unique[np.argmax(counts)]

# Oversample majority class to create imbalance
oversample_indices = np.where(y_test == majority_class)[0]
oversample_size = int(0.3 * len(y_test))
oversample_selection = np.random.choice(oversample_indices, size=oversample_size, replace=True)

X_test_drifted = np.vstack([X_test_drifted, X_test_drifted[oversample_selection]])
y_test_drifted = np.concatenate([y_test_drifted, y_test_drifted[oversample_selection]])

print()
print("   New class distribution:")
unique_new, counts_new = np.unique(y_test_drifted, return_counts=True)
for cls, count in zip(unique_new, counts_new):
    print(f"      Class {cls}: {count} samples ({count/len(y_test_drifted):.1%})")

print()
print(f"✅ Drifted dataset created: {X_test_drifted.shape}")
print()

# Create drifted DataFrame
df_test_drifted = pd.DataFrame(X_test_drifted, columns=feature_names)
df_test_drifted['target'] = y_test_drifted

# Run drift detection
print("=" * 70)
print("Running Drift Detection")
print("=" * 70)
print()

detector = DriftDetector(
    drift_threshold=0.5,
    stattest='ks'
)

# Detect data drift
print("🔍 Checking for data drift...")
data_has_drift, data_metrics = detector.detect_data_drift(
    reference_data=df_train,
    current_data=df_test_drifted
)

print()
print("📊 Data Drift Results:")
print(f"   Dataset drift: {'YES ⚠️' if data_has_drift else 'NO ✅'}")
print(f"   Drift share: {data_metrics['drift_share']:.2%}")
print(f"   Drifted features: {data_metrics['n_drifted_features']}/{data_metrics['n_features']}")
print()

# Detect target drift
print("🔍 Checking for target drift...")
target_has_drift, target_metrics = detector.detect_target_drift(
    reference_data=df_train,
    current_data=df_test_drifted,
    target_column='target'
)

print()
print("📊 Target Drift Results:")
print(f"   Target drift: {'YES ⚠️' if target_has_drift else 'NO ✅'}")
if 'drift_score' in target_metrics:
    print(f"   Drift score: {target_metrics['drift_score']:.4f}")
print()

# Save reports
print("💾 Saving drift reports...")
data_report_path = detector.save_report(
    df_train,
    df_test_drifted,
    REPORTS_PATH,
    "simulated_data_drift_report"
)
print()

# Retraining recommendation
should_retrain, reason = detector.should_retrain(
    data_drift=data_has_drift,
    target_drift=target_has_drift,
    prediction_drift=False,
    drift_share=data_metrics['drift_share'],
    retrain_threshold=0.3
)

print("=" * 70)
print("Retraining Decision")
print("=" * 70)
print()

print(f"🤖 Should retrain? {should_retrain}")
print(f"   {reason}")
print()

# Summary
print("=" * 70)
print("✅ DRIFT SIMULATION COMPLETE!")
print("=" * 70)
print()

print("📄 Generated Report:")
print(f"   {data_report_path}")
print()

print("🔍 What We Simulated:")
print("   1. Feature scaling drift (30% of features)")
print("   2. Data quality drift (added noise)")
print("   3. Distribution shift (15% of features)")
print("   4. Target class imbalance")
print()

print("💡 Next Steps:")
print("   1. Open the HTML reports to see detailed visualizations")
print("   2. Compare simulated drift vs. real drift:")
print("      python scripts/check_drift.py")
print("   3. In production, you would:")
print("      - Monitor these metrics continuously")
print("      - Alert when thresholds are exceeded")
print("      - Automatically trigger retraining")
print()

print("🎯 Key Takeaway:")
print("   The drift detection system successfully identified:")
print(f"   - Data drift in {data_metrics['drift_share']:.0%} of features")
print(f"   - Target drift: {target_has_drift}")
print(f"   - Recommended retraining: {should_retrain}")
print()
