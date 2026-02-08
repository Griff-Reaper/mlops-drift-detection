"""
Data Preprocessing Pipeline for Network Traffic Classification

This module handles all data preprocessing steps:
- Missing value imputation
- Feature scaling
- Label encoding
- Feature selection
- Data validation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkTrafficPreprocessor:
    """
    Preprocessing pipeline for network traffic data.
    
    This class encapsulates all preprocessing steps needed to prepare
    network traffic data for machine learning models.
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.01,
        scaling_method: str = "standard"
    ):
        """
        Initialize the preprocessor.
        
        Args:
            variance_threshold: Minimum variance for feature selection
            scaling_method: Type of scaling ('standard', 'minmax', or 'none')
        """
        self.variance_threshold = variance_threshold
        self.scaling_method = scaling_method
        
        # Initialize preprocessing components
        self.scaler = StandardScaler() if scaling_method == "standard" else None
        self.label_encoder = LabelEncoder()
        self.variance_selector = VarianceThreshold(threshold=variance_threshold)
        
        # Store feature names for later use
        self.feature_names: Optional[List[str]] = None
        self.selected_feature_names: Optional[List[str]] = None
        self.label_classes: Optional[np.ndarray] = None
        
        # Track if fitted
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NetworkTrafficPreprocessor':
        """
        Fit the preprocessor to training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series with class labels
            
        Returns:
            self: Fitted preprocessor
        """
        logger.info("Fitting preprocessor...")
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X_clean = self._handle_missing_values(X)
        
        # Fit variance selector
        logger.info(f"Selecting features with variance > {self.variance_threshold}")
        X_selected = self.variance_selector.fit_transform(X_clean)
        
        # Store selected feature names
        selected_mask = self.variance_selector.get_support()
        self.selected_feature_names = [
            name for name, selected in zip(self.feature_names, selected_mask)
            if selected
        ]
        logger.info(f"Selected {len(self.selected_feature_names)} features")
        
        # Fit scaler if enabled
        if self.scaler is not None:
            logger.info(f"Fitting {self.scaling_method} scaler")
            self.scaler.fit(X_selected)
        
        # Fit label encoder
        logger.info("Encoding labels")
        self.label_encoder.fit(y)
        self.label_classes = self.label_encoder.classes_
        logger.info(f"Found {len(self.label_classes)} classes: {self.label_classes}")
        
        self.is_fitted = True
        logger.info("✅ Preprocessor fitted successfully!")
        
        return self
    
    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Feature DataFrame
            y: Optional target Series
            
        Returns:
            X_transformed: Preprocessed features
            y_transformed: Encoded labels (if y provided)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform!")
        
        logger.info("Transforming data...")
        
        # Handle missing values
        X_clean = self._handle_missing_values(X)
        
        # Select features
        X_selected = self.variance_selector.transform(X_clean)
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_selected)
        else:
            X_scaled = X_selected
        
        # Encode labels if provided
        y_encoded = None
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
        
        logger.info(f"✅ Transformed {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        
        return X_scaled, y_encoded
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            X_transformed: Preprocessed features
            y_transformed: Encoded labels
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded labels back to original labels.
        
        Args:
            y_encoded: Encoded label array
            
        Returns:
            Original labels
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform!")
        
        return self.label_encoder.inverse_transform(y_encoded)
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Strategy:
        - For numeric columns: Fill with mean
        - For categorical columns: Fill with mode
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with no missing values
        """
        X_clean = X.copy()
        
        # Check for missing values
        missing_count = X_clean.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values, imputing...")
            
            # Fill numeric columns with mean
            numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X_clean[col].isnull().any():
                    X_clean[col].fillna(X_clean[col].mean(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if X_clean[col].isnull().any():
                    X_clean[col].fillna(X_clean[col].mode()[0], inplace=True)
        
        return X_clean
    
    def get_feature_importance_info(self) -> pd.DataFrame:
        """
        Get information about selected features.
        
        Returns:
            DataFrame with feature names and selection status
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first!")
        
        selected_mask = self.variance_selector.get_support()
        variances = self.variance_selector.variances_
        
        info = pd.DataFrame({
            'feature': self.feature_names,
            'selected': selected_mask,
            'variance': variances
        })
        
        return info.sort_values('variance', ascending=False)
    
    def save(self, filepath: Path) -> None:
        """
        Save the fitted preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor!")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        logger.info(f"✅ Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'NetworkTrafficPreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            Loaded preprocessor
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor not found at {filepath}")
        
        preprocessor = joblib.load(filepath)
        logger.info(f"✅ Preprocessor loaded from {filepath}")
        
        return preprocessor


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """
    Load raw network traffic data from CSV.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame with raw data
    """
    logger.info(f"Loading data from {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"✅ Loaded {len(df):,} samples with {len(df.columns)} columns")
    
    return df


def prepare_data(
    df: pd.DataFrame,
    target_column: str = 'Label',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Prepare data for training by splitting into train/val/test sets.
    
    Args:
        df: Full dataset
        target_column: Name of the target column
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Splitting data into train/val/test sets...")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: separate validation set from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    logger.info(f"✅ Data split complete:")
    logger.info(f"   Train: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    logger.info(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
    logger.info(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> None:
    """
    Save processed data to disk.
    
    Args:
        X_train, X_val, X_test: Feature arrays
        y_train, y_val, y_test: Target arrays
        output_dir: Directory to save processed data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to {output_dir}")
    
    # Save as numpy arrays for faster loading
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'X_val.npy', X_val)
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_val.npy', y_val)
    np.save(output_dir / 'y_test.npy', y_test)
    
    logger.info("✅ Processed data saved successfully!")


if __name__ == "__main__":
    """
    Example usage of the preprocessing pipeline.
    """
    import sys
    from pathlib import Path
    
    # Add project root to path
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Configuration
    DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
    
    # Find CSV file
    csv_files = list(DATA_RAW_PATH.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in data/raw/")
        print("   Run: python scripts/download_data.py")
        sys.exit(1)
    
    data_file = csv_files[0]
    
    # Load data
    df = load_raw_data(data_file)
    
    # Prepare splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)
    
    # Initialize and fit preprocessor
    preprocessor = NetworkTrafficPreprocessor(
        variance_threshold=0.01,
        scaling_method="standard"
    )
    
    # Fit on training data
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Transform validation and test data
    X_val_processed, y_val_processed = preprocessor.transform(X_val, y_val)
    X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
    
    # Save processed data
    save_processed_data(
        X_train_processed, X_val_processed, X_test_processed,
        y_train_processed, y_val_processed, y_test_processed,
        DATA_PROCESSED_PATH
    )
    
    # Save preprocessor
    preprocessor.save(PROJECT_ROOT / "models" / "preprocessor.pkl")
    
    # Print feature info
    print("\n" + "=" * 60)
    print("Feature Selection Summary")
    print("=" * 60)
    feature_info = preprocessor.get_feature_importance_info()
    print(f"\nTotal features: {len(feature_info)}")
    print(f"Selected features: {feature_info['selected'].sum()}")
    print(f"Removed features: {(~feature_info['selected']).sum()}")
    print("\nTop 10 features by variance:")
    print(feature_info.head(10).to_string(index=False))
    
    print("\n✅ Preprocessing complete!")
    print(f"   Processed data saved to: {DATA_PROCESSED_PATH}")
    print(f"   Preprocessor saved to: {PROJECT_ROOT / 'models' / 'preprocessor.pkl'}")
