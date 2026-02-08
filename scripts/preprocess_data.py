"""
Preprocess Raw Data

This script runs the full preprocessing pipeline:
1. Load raw network traffic data
2. Split into train/val/test sets
3. Fit preprocessor on training data
4. Transform all sets
5. Save processed data and preprocessor
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    load_raw_data,
    prepare_data,
    NetworkTrafficPreprocessor,
    save_processed_data
)

print("=" * 70)
print("Network Traffic Data - Preprocessing Pipeline")
print("=" * 70)
print()

# Configuration
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"

# Find CSV file
print("🔍 Looking for data files...")
csv_files = list(DATA_RAW_PATH.glob("*.csv"))

if not csv_files:
    print("❌ No CSV files found in data/raw/")
    print("   Run: python scripts/download_data.py")
    sys.exit(1)

data_file = csv_files[0]
print(f"✅ Found: {data_file.name}")
print()

# Load data
df = load_raw_data(data_file)
print()

# Prepare splits
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
    df,
    target_column='Label',
    test_size=0.2,
    val_size=0.1,
    random_state=42
)
print()

# Initialize preprocessor
print("🔧 Initializing preprocessor...")
preprocessor = NetworkTrafficPreprocessor(
    variance_threshold=0.01,
    scaling_method="standard"
)
print()

# Fit and transform training data
print("🎯 Fitting preprocessor on training data...")
X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
print()

# Transform validation and test data
print("🔄 Transforming validation and test data...")
X_val_processed, y_val_processed = preprocessor.transform(X_val, y_val)
X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
print()

# Save processed data
print("💾 Saving processed data...")
save_processed_data(
    X_train_processed, X_val_processed, X_test_processed,
    y_train_processed, y_val_processed, y_test_processed,
    DATA_PROCESSED_PATH
)
print()

# Save preprocessor
print("💾 Saving preprocessor...")
MODELS_PATH.mkdir(parents=True, exist_ok=True)
preprocessor.save(MODELS_PATH / "preprocessor.pkl")
print()

# Print summary
print("=" * 70)
print("Preprocessing Summary")
print("=" * 70)
print()

feature_info = preprocessor.get_feature_importance_info()
print(f"📊 Feature Statistics:")
print(f"   Total features:    {len(feature_info)}")
print(f"   Selected features: {feature_info['selected'].sum()}")
print(f"   Removed features:  {(~feature_info['selected']).sum()}")
print()

print(f"📦 Data Shapes:")
print(f"   Train: {X_train_processed.shape}")
print(f"   Val:   {X_val_processed.shape}")
print(f"   Test:  {X_test_processed.shape}")
print()

print(f"🎯 Classes:")
for i, class_name in enumerate(preprocessor.label_classes):
    count = (y_train_processed == i).sum()
    print(f"   {class_name}: {count:,} samples")
print()

print("📁 Output Locations:")
print(f"   Processed data: {DATA_PROCESSED_PATH}")
print(f"   Preprocessor:   {MODELS_PATH / 'preprocessor.pkl'}")
print()

print("=" * 70)
print("✅ PREPROCESSING COMPLETE!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Run: python scripts/train_models.py")
print("  2. Check MLflow UI: http://localhost:5000")
print()
