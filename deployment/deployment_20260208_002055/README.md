# Model Deployment Package

**Deployed:** 20260208_002055  
**Model:** random_forest  
**Test F1 Score:** 0.7111  
**Test Accuracy:** 0.8000

## Contents

- `model.pkl` - Trained random_forest model
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
sample = {
    'Flow Duration': 120000,
    'Total Fwd Packets': 8,
    'Total Backward Packets': 0,
    # ... other features ...
}

prediction, probs = classifier.predict_single(sample)
print(f"Prediction: {prediction}")
```

## Model Details

- **Algorithm:** random_forest
- **Features:** 29 (after feature selection)
- **Classes:** BENIGN, Bot, DDoS, PortScan
- **Preprocessing:** StandardScaler + Variance Threshold (0.01)

## Performance Metrics

- **Test Accuracy:** 0.8000
- **Test F1 Score:** 0.7111

## Deployment Notes

This model was trained on CICIDS2017 network traffic data and is designed to classify network traffic patterns including:
- Benign traffic
- DDoS attacks
- Port scans
- Botnet activity

**Important:** Input data must have the same features as training data. The preprocessor will handle scaling and feature selection automatically.

## MLflow Integration

This model was tracked in MLflow:
- **Run ID:** ca550779aa434536932a360bb0b31de0
- **Experiment:** network_traffic_classification
- **Tracking URI:** http://127.0.0.1:5000

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
