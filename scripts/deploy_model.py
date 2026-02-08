"""
Automated Model Deployment Script

This script:
1. Connects to MLflow
2. Finds the best performing model
3. Loads the model and preprocessor
4. Packages everything for deployment
5. Creates inference scripts

Usage:
    python scripts/deploy_model.py
"""

import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os
import shutil
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
load_dotenv()

print("=" * 70)
print("AUTOMATED MODEL DEPLOYMENT")
print("=" * 70)
print()

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "network_traffic_classification")
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
MODELS_DIR = PROJECT_ROOT / "models"

# Connect to MLflow
print("🔧 Connecting to MLflow...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    mlflow.search_experiments()
    print(f"✅ Connected to MLflow at {MLFLOW_TRACKING_URI}")
except:
    print(f"❌ Could not connect to MLflow")
    print("   Make sure MLflow server is running!")
    sys.exit(1)

print()

# ============================================================================
# STEP 1: Find Best Model
# ============================================================================
print("STEP 1: Find Best Model from MLflow")
print("-" * 70)

# Get experiment
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    print(f"❌ Experiment '{EXPERIMENT_NAME}' not found!")
    sys.exit(1)

# Search for best run
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_f1 DESC"],
    max_results=1
)

if runs.empty:
    print("❌ No runs found in experiment!")
    sys.exit(1)

best_run = runs.iloc[0]
run_id = best_run['run_id']
model_name = best_run['params.model_type'] if 'params.model_type' in best_run else 'unknown'
test_f1 = best_run['metrics.test_f1']
test_accuracy = best_run['metrics.test_accuracy']

print(f"✅ Found best model:")
print(f"   Run ID: {run_id}")
print(f"   Model: {model_name}")
print(f"   Test F1: {test_f1:.4f}")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print()

# ============================================================================
# STEP 2: Load Model and Preprocessor
# ============================================================================
print("STEP 2: Load Model and Preprocessor")
print("-" * 70)

# Load model from MLflow
print("📥 Loading model from MLflow...")
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)
print("✅ Model loaded")

# Load preprocessor
print("📥 Loading preprocessor...")
preprocessor_path = MODELS_DIR / "preprocessor.pkl"

if not preprocessor_path.exists():
    print("❌ Preprocessor not found!")
    print(f"   Expected: {preprocessor_path}")
    sys.exit(1)

from src.data.preprocessing import NetworkTrafficPreprocessor
preprocessor = NetworkTrafficPreprocessor.load(preprocessor_path)
print("✅ Preprocessor loaded")
print()

# ============================================================================
# STEP 3: Create Deployment Package
# ============================================================================
print("STEP 3: Create Deployment Package")
print("-" * 70)

# Create deployment directory
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

# Create timestamped deployment folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
deploy_folder = DEPLOYMENT_DIR / f"deployment_{timestamp}"
deploy_folder.mkdir(parents=True, exist_ok=True)

print(f"📁 Created deployment folder: {deploy_folder}")

# Save model
print("💾 Saving model...")
import joblib
model_path = deploy_folder / "model.pkl"
joblib.dump(model, model_path)
print(f"   ✅ Model saved: {model_path}")

# Copy preprocessor
print("💾 Copying preprocessor...")
preprocessor_deploy_path = deploy_folder / "preprocessor.pkl"
shutil.copy(preprocessor_path, preprocessor_deploy_path)
print(f"   ✅ Preprocessor saved: {preprocessor_deploy_path}")

# Create metadata
print("📝 Creating deployment metadata...")
metadata = {
    "deployment_timestamp": timestamp,
    "mlflow_run_id": run_id,
    "model_name": model_name,
    "test_f1_score": float(test_f1),
    "test_accuracy": float(test_accuracy),
    "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    "experiment_name": EXPERIMENT_NAME,
    "feature_count": len(preprocessor.selected_feature_names),
    "label_classes": preprocessor.label_classes.tolist()
}

metadata_path = deploy_folder / "metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✅ Metadata saved: {metadata_path}")
print()

# ============================================================================
# STEP 4: Create Inference Script
# ============================================================================
print("STEP 4: Create Inference Script")
print("-" * 70)

inference_script = f'''"""
Model Inference Script

This script loads the deployed model and makes predictions on new data.

Usage:
    from inference import NetworkTrafficClassifier
    
    # Initialize
    classifier = NetworkTrafficClassifier()
    
    # Make predictions on CSV file
    predictions = classifier.predict_from_csv("new_data.csv")
    
    # Or on DataFrame
    predictions = classifier.predict(df)
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json

class NetworkTrafficClassifier:
    """
    Production-ready network traffic classifier.
    
    Loads the trained model and preprocessor for making predictions.
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the classifier by loading model and preprocessor.
        
        Args:
            model_dir: Path to deployment directory (defaults to current directory)
        """
        if model_dir is None:
            model_dir = Path(__file__).parent
        else:
            model_dir = Path(model_dir)
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load model and preprocessor
        self.model = joblib.load(model_dir / "model.pkl")
        self.preprocessor = joblib.load(model_dir / "preprocessor.pkl")
        
        print("✅ Model loaded successfully")
        print(f"   Model: {{self.metadata['model_name']}}")
        print(f"   Test F1: {{self.metadata['test_f1_score']:.4f}}")
        print(f"   Features: {{self.metadata['feature_count']}}")
    
    def predict(self, X):
        """
        Make predictions on a DataFrame.
        
        Args:
            X: DataFrame with feature columns (without target)
            
        Returns:
            predictions: Array of predicted class labels
            probabilities: Array of prediction probabilities
        """
        # Preprocess
        X_processed, _ = self.preprocessor.transform(X, y=None)
        
        # Predict
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed) if hasattr(self.model, 'predict_proba') else None
        
        # Convert back to original labels
        predictions_labels = self.preprocessor.inverse_transform_labels(predictions)
        
        return predictions_labels, probabilities
    
    def predict_from_csv(self, csv_path, target_column='Label'):
        """
        Make predictions on data from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            target_column: Name of target column (will be dropped if present)
            
        Returns:
            DataFrame with original data + predictions
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Separate features and target if present
        if target_column in df.columns:
            X = df.drop(target_column, axis=1)
            y_true = df[target_column]
            has_labels = True
        else:
            X = df
            y_true = None
            has_labels = False
        
        # Make predictions
        predictions, probabilities = self.predict(X)
        
        # Create results DataFrame
        results = df.copy()
        results['predicted_label'] = predictions
        
        if probabilities is not None:
            # Add probability columns for each class
            for i, class_name in enumerate(self.metadata['label_classes']):
                results[f'prob_{{class_name}}'] = probabilities[:, i]
        
        if has_labels:
            results['correct'] = (predictions == y_true.values)
            accuracy = results['correct'].mean()
            print(f"\\n📊 Prediction Accuracy: {{accuracy:.4f}}")
        
        return results
    
    def predict_single(self, sample_dict):
        """
        Make a prediction on a single sample.
        
        Args:
            sample_dict: Dictionary with feature values
            
        Returns:
            prediction: Predicted class label
            probabilities: Prediction probabilities for each class
        """
        # Convert to DataFrame
        df = pd.DataFrame([sample_dict])
        
        # Predict
        predictions, probabilities = self.predict(df)
        
        return predictions[0], probabilities[0] if probabilities is not None else None


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Network Traffic Classifier - Inference Demo")
    print("=" * 70)
    print()
    
    # Initialize classifier
    classifier = NetworkTrafficClassifier()
    
    print("\\nModel ready for predictions!")
    print("\\nExample usage:")
    print("  predictions = classifier.predict_from_csv('new_data.csv')")
    print("  predictions.to_csv('predictions.csv', index=False)")
'''

inference_path = deploy_folder / "inference.py"
with open(inference_path, 'w', encoding='utf-8') as f:
    f.write(inference_script)

print(f"✅ Inference script created: {inference_path}")
print()

# ============================================================================
# STEP 5: Create README
# ============================================================================
print("STEP 5: Create Deployment README")
print("-" * 70)

readme_content = f'''# Model Deployment Package

**Deployed:** {timestamp}  
**Model:** {model_name}  
**Test F1 Score:** {test_f1:.4f}  
**Test Accuracy:** {test_accuracy:.4f}

## Contents

- `model.pkl` - Trained {model_name} model
- `preprocessor.pkl` - Data preprocessor with feature scaling and selection
- `metadata.json` - Deployment metadata and model performance metrics
- `inference.py` - Production inference script
- `README.md` - This file

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn joblib
```

### 2. Make Predictions

```python
from inference import NetworkTrafficClassifier

# Initialize classifier
classifier = NetworkTrafficClassifier()

# Predict from CSV file
results = classifier.predict_from_csv("new_data.csv")
results.to_csv("predictions.csv", index=False)

# Or predict from DataFrame
import pandas as pd
df = pd.read_csv("new_data.csv")
predictions, probabilities = classifier.predict(df)
```

### 3. Single Prediction Example

```python
sample = {{
    'Flow Duration': 120000,
    'Total Fwd Packets': 8,
    'Total Backward Packets': 0,
    # ... other features ...
}}

prediction, probs = classifier.predict_single(sample)
print(f"Prediction: {{prediction}}")
```

## Model Details

- **Algorithm:** {model_name}
- **Features:** {len(preprocessor.selected_feature_names)} (after feature selection)
- **Classes:** {', '.join(preprocessor.label_classes.tolist())}
- **Preprocessing:** StandardScaler + Variance Threshold (0.01)

## Performance Metrics

- **Test Accuracy:** {test_accuracy:.4f}
- **Test F1 Score:** {test_f1:.4f}

## Deployment Notes

This model was trained on CICIDS2017 network traffic data and is designed to classify network traffic patterns including:
- Benign traffic
- DDoS attacks
- Port scans
- Botnet activity

**Important:** Input data must have the same features as training data. The preprocessor will handle scaling and feature selection automatically.

## MLflow Integration

This model was tracked in MLflow:
- **Run ID:** {run_id}
- **Experiment:** {EXPERIMENT_NAME}
- **Tracking URI:** {MLFLOW_TRACKING_URI}

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install numpy pandas scikit-learn joblib

CMD ["python", "inference.py"]
```

### REST API (Optional)

For production API deployment, consider wrapping this in FastAPI or Flask.

## Support

For issues or questions, refer to the main project repository.
'''

readme_path = deploy_folder / "README.md"
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)

print(f"✅ README created: {readme_path}")
print()

# ============================================================================
# STEP 6: Create Symlink to Latest
# ============================================================================
print("STEP 6: Create 'latest' Symlink")
print("-" * 70)

latest_link = DEPLOYMENT_DIR / "latest"

# Remove old symlink if exists
if latest_link.exists() or latest_link.is_symlink():
    latest_link.unlink()

# Create new symlink (on Windows, this might require admin rights or dev mode)
try:
    latest_link.symlink_to(deploy_folder, target_is_directory=True)
    print(f"✅ Created symlink: {latest_link} -> {deploy_folder}")
except OSError:
    print(f"⚠️  Could not create symlink (Windows requires special permissions)")
    print(f"   You can manually reference: {deploy_folder}")

print()

# ============================================================================
# COMPLETE
# ============================================================================
print("=" * 70)
print("✅ DEPLOYMENT COMPLETE!")
print("=" * 70)
print()
print(f"📦 Deployment Package: {deploy_folder}")
print()
print("Package Contents:")
print(f"  ✅ model.pkl - Trained {model_name} model")
print(f"  ✅ preprocessor.pkl - Data preprocessor")
print(f"  ✅ metadata.json - Deployment metadata")
print(f"  ✅ inference.py - Production inference script")
print(f"  ✅ README.md - Deployment documentation")
print()
print("Next Steps:")
print(f"  1. Test inference:")
print(f"     cd {deploy_folder}")
print(f"     python inference.py")
print()
print(f"  2. Make predictions:")
print(f"     from inference import NetworkTrafficClassifier")
print(f"     classifier = NetworkTrafficClassifier('{deploy_folder}')")
print(f"     predictions = classifier.predict_from_csv('data.csv')")
print()
print(f"  3. Deploy to production:")
print(f"     - Copy deployment folder to production server")
print(f"     - Run inference.py or integrate with API")
print()
print("🎉 Model is ready for production deployment!")
print()
