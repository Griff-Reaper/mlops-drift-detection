"""
FastAPI Model Serving Endpoint (Optional)

This creates a REST API for the deployed model.

Install dependencies:
    pip install fastapi uvicorn

Run the API:
    python scripts/serve_api.py
    
    # Or with uvicorn directly:
    uvicorn scripts.serve_api:app --reload --host 0.0.0.0 --port 8000

API Endpoints:
    GET  /health              - Health check
    GET  /model/info          - Model metadata
    POST /predict             - Make predictions
    POST /predict/batch       - Batch predictions
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("❌ FastAPI not installed!")
    print("   Install with: pip install fastapi uvicorn")
    sys.exit(1)

# Find latest deployment
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment" / "latest"

if not DEPLOYMENT_DIR.exists():
    DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
    deployments = sorted(DEPLOYMENT_DIR.glob("deployment_*"))
    if deployments:
        DEPLOYMENT_DIR = deployments[-1]
    else:
        print("❌ No deployment found!")
        print("   Run: python scripts/deploy_model.py")
        sys.exit(1)

# Import inference module
sys.path.insert(0, str(DEPLOYMENT_DIR))
from inference import NetworkTrafficClassifier

# Initialize FastAPI app
app = FastAPI(
    title="Network Traffic Classification API",
    description="REST API for network traffic classification using ML",
    version="1.0.0"
)

# Load model at startup
classifier = None

@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    global classifier
    print("🚀 Loading model...")
    classifier = NetworkTrafficClassifier(DEPLOYMENT_DIR)
    print("✅ Model loaded and ready!")


# Request/Response Models
class PredictionRequest(BaseModel):
    """Single prediction request."""
    features: Dict[str, float]
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "Flow Duration": 120000,
                    "Total Fwd Packets": 8,
                    "Total Backward Packets": 0,
                    "Fwd Packet Length Max": 1500,
                    "Bwd Packet Length Max": 0
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    samples: List[Dict[str, float]]


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: str
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_samples: int


class ModelInfo(BaseModel):
    """Model information."""
    model_name: str
    test_f1_score: float
    test_accuracy: float
    feature_count: int
    classes: List[str]
    deployment_timestamp: str


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Network Traffic Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": classifier.metadata["model_name"]
    }


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model information and metadata."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=classifier.metadata["model_name"],
        test_f1_score=classifier.metadata["test_f1_score"],
        test_accuracy=classifier.metadata["test_accuracy"],
        feature_count=classifier.metadata["feature_count"],
        classes=classifier.metadata["label_classes"],
        deployment_timestamp=classifier.metadata["deployment_timestamp"]
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction on a single sample.
    
    Send a POST request with feature values.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        prediction, probabilities = classifier.predict_single(request.features)
        
        # Format response
        response = {
            "prediction": str(prediction)
        }
        
        if probabilities is not None:
            # Convert to dict
            prob_dict = {
                str(class_name): float(prob)
                for class_name, prob in zip(classifier.metadata["label_classes"], probabilities)
            }
            response["probabilities"] = prob_dict
            response["confidence"] = float(max(probabilities))
        
        return PredictionResponse(**response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions on multiple samples.
    
    Send a POST request with a list of feature dictionaries.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(request.samples)
        
        # Make predictions
        predictions, probabilities = classifier.predict(df)
        
        # Format responses
        results = []
        for i, pred in enumerate(predictions):
            response_dict = {"prediction": str(pred)}
            
            if probabilities is not None:
                prob_dict = {
                    str(class_name): float(prob)
                    for class_name, prob in zip(classifier.metadata["label_classes"], probabilities[i])
                }
                response_dict["probabilities"] = prob_dict
                response_dict["confidence"] = float(max(probabilities[i]))
            
            results.append(PredictionResponse(**response_dict))
        
        return BatchPredictionResponse(
            predictions=results,
            total_samples=len(predictions)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


# Run the API
if __name__ == "__main__":
    print("=" * 70)
    print("Starting FastAPI Model Serving")
    print("=" * 70)
    print()
    print(f"📦 Loading model from: {DEPLOYMENT_DIR}")
    print()
    print("Starting server...")
    print("  API Docs: http://localhost:8000/docs")
    print("  Health Check: http://localhost:8000/health")
    print("  Model Info: http://localhost:8000/model/info")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
