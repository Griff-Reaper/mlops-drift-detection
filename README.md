# MLOps Pipeline with Drift Detection

Production-grade MLOps pipeline demonstrating continuous training, model monitoring, and automated drift detection for network traffic classification.

## 🎯 Project Overview

This project implements a complete MLOps lifecycle for a machine learning model that classifies network traffic. It showcases:

- **Automated Training Pipeline**: Orchestrated ML workflows with experiment tracking
- **Drift Detection**: Monitoring for data drift and model performance degradation
- **CI/CD Integration**: Automated testing and deployment
- **Model Registry**: Versioned models with metadata and lineage
- **Production Monitoring**: Real-time performance tracking and alerting

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources                                 │
│              (Network Traffic Dataset)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Prefect Orchestration                           │
│                  (Workflow Management)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Data       │  │   Feature    │  │   Model      │
│  Ingestion   │→ │  Engineering │→ │   Training   │
└──────────────┘  └──────────────┘  └──────────────┘
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │   MLflow     │
                                    │  Experiment  │
                                    │   Tracking   │
                                    └──────────────┘
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │    Model     │
                                    │   Registry   │
                                    └──────────────┘
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │ Evidently AI │
                                    │    Drift     │
                                    │  Detection   │
                                    └──────────────┘
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │  Automated   │
                                    │  Retraining  │
                                    │   Trigger    │
                                    └──────────────┘
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | Prefect | Workflow management and scheduling |
| **Experiment Tracking** | MLflow | Model versioning and metrics |
| **Drift Detection** | Evidently AI | Data and model drift monitoring |
| **Data Versioning** | DVC | Dataset version control |
| **CI/CD** | GitHub Actions | Automated testing and deployment |
| **ML Framework** | Scikit-learn, XGBoost | Model training |
| **Infrastructure** | Docker | Containerization |

## 📊 Dataset

**CICIDS2017** - Intrusion Detection Evaluation Dataset
- 2.8M network flow records
- 8 attack types (DDoS, Brute Force, Botnet, Web Attacks, etc.)
- 80+ features extracted from packet captures
- Realistic network traffic scenarios

Alternative: **UNSW-NB15** if CICIDS2017 unavailable

## 📁 Project Structure

```
mlops-drift-detection/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and preprocessed
│   └── .dvc/                   # DVC tracking
├── src/
│   ├── data/
│   │   ├── ingestion.py       # Data loading
│   │   └── preprocessing.py   # Feature engineering
│   ├── models/
│   │   ├── train.py           # Training logic
│   │   └── evaluate.py        # Evaluation metrics
│   ├── monitoring/
│   │   ├── drift_detection.py # Evidently integration
│   │   └── alerts.py          # Alert system
│   └── pipelines/
│       ├── training_pipeline.py   # Prefect training flow
│       └── monitoring_pipeline.py # Prefect monitoring flow
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_monitoring.py
├── config/
│   ├── model_config.yaml      # Model hyperparameters
│   └── monitoring_config.yaml # Drift thresholds
├── .github/
│   └── workflows/
│       ├── train.yml          # CI/CD for training
│       └── test.yml           # CI/CD for testing
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory analysis
│   └── 02_baseline.ipynb     # Baseline models
├── mlruns/                    # MLflow artifacts
├── .dvc/                      # DVC configuration
├── .env.example
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional)
- 4GB RAM minimum
- 10GB disk space

### Installation

1. **Clone and setup**
```bash
git clone https://github.com/YOUR_USERNAME/mlops-drift-detection.git
cd mlops-drift-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download dataset**
```bash
python scripts/download_data.py
```

3. **Initialize DVC**
```bash
dvc init
dvc add data/raw/
```

4. **Start MLflow tracking server**
```bash
mlflow server --host 127.0.0.1 --port 5000
```

5. **Run training pipeline**
```bash
python src/pipelines/training_pipeline.py
```

## 📈 Key Features

### 1. Automated Training Pipeline
- Scheduled retraining every 24 hours
- Experiment tracking with MLflow
- Hyperparameter optimization
- Model versioning and registry

### 2. Drift Detection
- **Data Drift**: Monitors feature distribution changes
- **Model Drift**: Tracks performance degradation
- **Prediction Drift**: Detects output distribution shifts
- Automated retraining triggers on drift

### 3. CI/CD Integration
- Automated testing on pull requests
- Model validation before deployment
- Docker containerization
- GitHub Actions workflows

### 4. Monitoring Dashboard
- Real-time metrics visualization
- Drift reports and alerts
- Model performance tracking
- Data quality checks

## 🎯 Success Metrics

- [ ] Pipeline runs automatically on schedule
- [ ] Drift detection accurately identifies data shifts
- [ ] Automated retraining triggers on drift
- [ ] Model achieves >95% accuracy on test set
- [ ] Complete CI/CD with automated testing
- [ ] <5 minute pipeline execution time

## 📚 Documentation

- [Setup Guide](docs/SETUP.md) - Detailed installation instructions
- [Architecture](docs/ARCHITECTURE.md) - System design decisions
- [API Reference](docs/API.md) - Code documentation
- [Deployment](docs/DEPLOYMENT.md) - Production deployment guide

## 🔄 Development Workflow

1. Make changes to code
2. Run tests: `pytest tests/`
3. Train model: `python src/pipelines/training_pipeline.py`
4. Check MLflow UI: http://localhost:5000
5. Review drift reports in `reports/`
6. Commit and push (CI/CD runs automatically)

## 🎓 Learning Outcomes

By completing this project, you'll demonstrate:
- MLOps best practices and tooling
- Production ML system design
- Drift detection and monitoring
- CI/CD for machine learning
- Model lifecycle management
- Automated retraining workflows

## 🚧 Roadmap

- [x] Phase 1: Core Pipeline Setup
- [x] Phase 2: Model Development  
- [x] Phase 3: Monitoring & Drift Detection
- [ ] Phase 4: CI/CD & Automation
- [ ] Phase 5: Documentation & Deployment
- [ ] Phase 6: Demo & Portfolio Integration

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome!

## 📧 Contact

Jace Griffith - [Your LinkedIn/Email]

---

**Status**: 🚧 In Active Development  
**Target Completion**: March 21, 2026  
**Portfolio**: AI Security Engineering Projects
