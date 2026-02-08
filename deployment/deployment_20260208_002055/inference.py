"""
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
        print(f"   Model: {self.metadata['model_name']}")
        print(f"   Test F1: {self.metadata['test_f1_score']:.4f}")
        print(f"   Features: {self.metadata['feature_count']}")
    
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
                results[f'prob_{class_name}'] = probabilities[:, i]
        
        if has_labels:
            results['correct'] = (predictions == y_true.values)
            accuracy = results['correct'].mean()
            print(f"\n📊 Prediction Accuracy: {accuracy:.4f}")
        
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
    
    print("\nModel ready for predictions!")
    print("\nExample usage:")
    print("  predictions = classifier.predict_from_csv('new_data.csv')")
    print("  predictions.to_csv('predictions.csv', index=False)")
