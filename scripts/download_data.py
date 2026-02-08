"""
Data Download Script for CICIDS2017 Dataset

This script downloads the CICIDS2017 network intrusion detection dataset.
We'll use a subset for initial development to keep things manageable.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)

# Dataset URLs (using smaller Friday dataset for manageable size)
CICIDS_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"

print("=" * 60)
print("CICIDS2017 Dataset Download")
print("=" * 60)
print()
print("⚠️  NOTE: The full CICIDS2017 dataset is ~7GB.")
print("For this project, we'll download the Friday subset (~500MB)")
print()
print("Manual Download Instructions:")
print("1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html")
print("2. Download: 'Friday-WorkingHours.pcap_ISCX.csv'")
print("3. Place the file in: data/raw/")
print()
print(f"Target directory: {DATA_RAW_PATH.absolute()}")
print()

# Check if data already exists
existing_files = list(DATA_RAW_PATH.glob("*.csv"))
if existing_files:
    print(f"✅ Found existing CSV files in {DATA_RAW_PATH}:")
    for f in existing_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size_mb:.1f} MB)")
    print()
    response = input("Do you want to download anyway? (y/N): ")
    if response.lower() != 'y':
        print("Skipping download.")
        exit(0)

print("=" * 60)
print("Alternative: Using Sample Data for Quick Start")
print("=" * 60)
print()
print("I can create a sample dataset for you to start immediately.")
print("This will let you:")
print("- Test the pipeline without waiting for large downloads")
print("- Verify everything works")
print("- Replace with real data later")
print()

use_sample = input("Create sample dataset for quick start? (Y/n): ")

if use_sample.lower() != 'n':
    print("\n🔨 Creating sample dataset...")
    
    import pandas as pd
    import numpy as np
    
    # Create a synthetic network traffic dataset
    np.random.seed(42)
    n_samples = 10000
    
    # Features similar to CICIDS2017
    data = {
        'Flow Duration': np.random.exponential(scale=1000000, size=n_samples),
        'Total Fwd Packets': np.random.poisson(lam=10, size=n_samples),
        'Total Backward Packets': np.random.poisson(lam=8, size=n_samples),
        'Total Length of Fwd Packets': np.random.exponential(scale=5000, size=n_samples),
        'Total Length of Bwd Packets': np.random.exponential(scale=3000, size=n_samples),
        'Fwd Packet Length Max': np.random.exponential(scale=1000, size=n_samples),
        'Fwd Packet Length Min': np.random.exponential(scale=100, size=n_samples),
        'Fwd Packet Length Mean': np.random.normal(loc=500, scale=200, size=n_samples),
        'Fwd Packet Length Std': np.random.exponential(scale=200, size=n_samples),
        'Bwd Packet Length Max': np.random.exponential(scale=800, size=n_samples),
        'Bwd Packet Length Min': np.random.exponential(scale=80, size=n_samples),
        'Bwd Packet Length Mean': np.random.normal(loc=400, scale=150, size=n_samples),
        'Bwd Packet Length Std': np.random.exponential(scale=150, size=n_samples),
        'Flow Bytes/s': np.random.exponential(scale=100000, size=n_samples),
        'Flow Packets/s': np.random.exponential(scale=100, size=n_samples),
        'Flow IAT Mean': np.random.exponential(scale=10000, size=n_samples),
        'Flow IAT Std': np.random.exponential(scale=5000, size=n_samples),
        'Flow IAT Max': np.random.exponential(scale=50000, size=n_samples),
        'Flow IAT Min': np.random.exponential(scale=100, size=n_samples),
        'Fwd IAT Total': np.random.exponential(scale=100000, size=n_samples),
        'Fwd IAT Mean': np.random.exponential(scale=10000, size=n_samples),
        'Fwd IAT Std': np.random.exponential(scale=5000, size=n_samples),
        'Fwd IAT Max': np.random.exponential(scale=50000, size=n_samples),
        'Fwd IAT Min': np.random.exponential(scale=100, size=n_samples),
        'Bwd IAT Total': np.random.exponential(scale=80000, size=n_samples),
        'Bwd IAT Mean': np.random.exponential(scale=8000, size=n_samples),
        'Bwd IAT Std': np.random.exponential(scale=4000, size=n_samples),
        'Bwd IAT Max': np.random.exponential(scale=40000, size=n_samples),
        'Bwd IAT Min': np.random.exponential(scale=80, size=n_samples),
    }
    
    # Create labels (20% attack traffic, 80% benign)
    labels = np.random.choice(['BENIGN', 'DDoS', 'PortScan', 'Bot'], 
                             size=n_samples, 
                             p=[0.8, 0.1, 0.05, 0.05])
    data['Label'] = labels
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = DATA_RAW_PATH / "network_traffic_sample.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Sample dataset created: {output_file}")
    print(f"   - Samples: {len(df):,}")
    print(f"   - Features: {len(df.columns) - 1}")
    print(f"   - Classes: {df['Label'].nunique()}")
    print(f"   - File size: {output_file.stat().st_size / 1024:.1f} KB")
    print()
    print("📊 Class distribution:")
    print(df['Label'].value_counts())
    print()
    print("🎯 You can now proceed with the test_setup.py script!")
    print()
    print("💡 To use real CICIDS2017 data later:")
    print("   1. Download from the link above")
    print("   2. Place CSV files in data/raw/")
    print("   3. Update data paths in your code")

else:
    print("\n📥 Download Instructions:")
    print("1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html")
    print("2. Scroll to 'Download' section")
    print("3. Download one or more CSV files")
    print("4. Extract them to:", DATA_RAW_PATH.absolute())
    print()
    print("Recommended for quick start:")
    print("  - Friday-WorkingHours.pcap_ISCX.csv (~500MB)")
    print()
    print("For full dataset:")
    print("  - Download all files (~7GB total)")

print()
print("=" * 60)
print("Next Steps:")
print("=" * 60)
print("1. Run: python scripts/test_setup.py")
print("2. Check MLflow UI: http://localhost:5000")
print("3. Continue with KICKOFF_GUIDE.md")
print()
