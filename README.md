# FraudSys - Real-time Fraud Detection System

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://bandit.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready fraud detection system built with modern MLOps practices, featuring real-time predictions, comprehensive feature engineering, and enterprise-grade monitoring.

## Overview

FraudSys is a complete end-to-end machine learning system for credit card fraud detection that combines:

- **Real-time Predictions**: FastAPI service with <100ms response times
- **Feature Store**: Feast integration for batch and streaming features
- **Model Registry**: MLflow for experiment tracking and model versioning
- **Streaming Pipeline**: Kafka-based real-time data processing
- **Comprehensive Monitoring**: Prometheus metrics and Grafana dashboards
- **Cloud-Native**: S3 integration and containerized deployment

The system processes over 1M transactions from the Kaggle fraud detection dataset and implements sophisticated time-windowed features for accurate fraud detection.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Infrastructure Setup](#infrastructure-setup)
- [Usage](#usage)
  - [Training Pipeline](#training-pipeline)
  - [Real-time Prediction Service](#real-time-prediction-service)
  - [Monitoring](#monitoring)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Development](#development)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Features

### 🚀 Production-Ready ML Pipeline
- Complete training workflow from data extraction to model deployment
- Automated hyperparameter tuning with XGBoost
- Model validation and promotion with MLflow aliases
- Time series cross-validation for temporal data

### 🎯 Real-time Fraud Detection
- FastAPI service with automatic OpenAPI documentation
- Real-time feature processing with Kafka streams
- <100ms prediction latency with confidence scoring
- Comprehensive input validation and error handling

### 📊 Advanced Feature Engineering
- Time-windowed customer transaction statistics (1h-30d)
- Merchant risk indicators and fraud rate tracking
- Customer behavioral patterns and velocity analysis
- Geographic and temporal pattern detection

### 🔧 Enterprise Infrastructure
- Containerized microservices with Docker Compose
- Kafka streaming for real-time data processing
- Redis for online feature storage
- PostgreSQL backend for MLflow experiments
- Prometheus + Grafana monitoring stack

### 📈 Comprehensive Observability
- Real-time fraud rate monitoring
- Prediction throughput and latency metrics
- Model performance tracking
- Business intelligence dashboards

## Project Structure

```
fraudsys/
├── src/fraudsys/           # Main Python package
│   ├── core/               # Core ML components
│   │   ├── features.py     # Feature engineering logic
│   │   ├── models.py       # Model training and evaluation
│   │   ├── pipelines.py    # ML workflow orchestration
│   │   ├── schemas.py      # Data validation schemas
│   │   └── metrics.py      # Performance metrics
│   ├── io/                 # Input/output handling
│   │   ├── configs.py      # Configuration management
│   │   ├── datasets.py     # Data loaders and writers
│   │   ├── kafka.py        # Kafka integration
│   │   ├── registries.py   # MLflow integration
│   │   └── runtimes.py     # Execution environments
│   ├── jobs/               # ML pipeline jobs
│   │   ├── base.py         # Base job classes
│   │   └── *.py           # Individual job implementations
│   ├── services/           # Long-running services
│   │   ├── base.py         # Base service classes
│   │   ├── api.py          # FastAPI prediction service
│   │   ├── feature.py      # Feature processing service
│   │   ├── monitoring.py   # Metrics collection service
│   │   └── producer.py     # Data simulation service
│   └── utils/              # Utility functions
│       ├── samplers.py     # Data sampling strategies
│       ├── searchers.py    # Hyperparameter optimization
│       ├── signers.py      # Security utilities
│       └── splitters.py    # Time-aware data splitting
├── confs/                  # Configuration files
│   ├── jobs/               # Job configurations (YAML)
│   ├── services/           # Service configurations (YAML)
│   └── env/                # Environment-specific configs
├── data/                   # Data storage
│   ├── training/           # Training datasets
│   ├── prod/               # Production simulation data
│   └── raw/                # Raw downloaded data
├── infra/                  # Infrastructure as code
│   ├── terraform/          # AWS infrastructure
│   └── feast/              # Feature store configuration
├── notebooks/              # Jupyter notebooks for analysis
├── docs/                   # Documentation
├── tasks/                  # Just task definitions
├── tests/                  # Test suite
└── docker-compose.yaml     # Service orchestration
```

## Quick Start

### Prerequisites

- **Python 3.13+**
- **Docker & Docker Compose**
- **AWS CLI** (configured with credentials)
- **Just** (task runner) - `cargo install just` or `brew install just`

### 1. Clone and Setup

```bash
git clone <repository-url>
cd fraudsys
just install  # Installs dependencies and git hooks
```

### 2. Infrastructure Setup

```bash
# Start local infrastructure
docker-compose up -d postgres redis kafka zookeeper mlflow

# Setup AWS infrastructure (optional)
just terraform-apply
```

### 3. Run Complete Training Pipeline

```bash
# Download data and train model
just pipeline-training
```

This executes the complete ML pipeline:
1. **Extract**: Downloads fraud detection dataset from Kaggle
2. **Feature Engineering**: Computes time-windowed features
3. **Training**: Trains XGBoost model with hyperparameter tuning
4. **Evaluation**: Validates model performance
5. **Promotion**: Promotes model to Champion status

### 4. Start Real-time Services

```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8000/health
```

### 5. Make Predictions

```bash
# Send test transaction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C123456789",
    "merchant_name": "fraud_Kirlin and Sons",
    "category": "grocery_pos",
    "amount_usd": 100.50,
    "transaction_time": "2024-01-01 12:00:00"
  }'
```

## Setup

### Prerequisites

**System Requirements:**
- Python 3.13+
- Docker Desktop
- AWS CLI (for cloud features)
- 8GB+ RAM for local development

**External Services:**
- AWS Account (for S3 and IAM)
- Kaggle Account (for dataset download)

### Installation

1. **Install UV Package Manager**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd fraudsys
   ```

3. **Install Dependencies**:
   ```bash
   just install
   # Or manually:
   uv sync
   uv run pre-commit install
   ```

4. **Configure Environment**:
   ```bash
   # Copy environment template
   cp .env.example .env

   # Add your Kaggle credentials
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
   ```

### Infrastructure Setup

**Local Development (Required):**
```bash
# Start core infrastructure
docker-compose up -d postgres redis kafka zookeeper mlflow

# Verify services are running
docker-compose ps
```

**AWS Infrastructure (Optional):**
```bash
# Configure AWS credentials
aws configure

# Deploy infrastructure
just terraform-init
just terraform-apply

# This creates:
# - S3 buckets for feature store
# - IAM user with appropriate permissions
```

## Usage

### Training Pipeline

The complete training pipeline can be executed with:

```bash
just pipeline-training
```

Or run individual steps:

```bash
# 1. Download and prepare data
fraudsys job extract

# 2. Engineer features for feature store
fraudsys job feature_engineering

# 3. Enrich training data with features
fraudsys job offline_features

# 4. Train and evaluate model
fraudsys job training
fraudsys job training_evaluation

# 5. Promote model to production
fraudsys job offline_promotion
```

### Real-time Prediction Service

**Start Services:**
```bash
docker-compose up -d
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Make Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C1234567890",
    "merchant_name": "xyz_grocery",
    "category": "grocery_pos",
    "amount_usd": 67.89,
    "transaction_time": "2024-01-01T14:30:00"
  }'

# Response:
{
  "transaction_id": "txn_abc123",
  "is_fraud": 0,
  "model_version": "1.0.0",
  "timestamp": "2024-01-01T14:30:01.123Z"
}
```

### Monitoring

**Access Monitoring Dashboards:**
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000
- **Kafka UI**: http://localhost:8080

**Key Metrics:**
- Real-time fraud detection rate
- Prediction latency and throughput
- Model confidence distributions
- System health and performance

### Available Commands

```bash
# Code Quality
just check-code          # Run ruff linting
just check-format        # Check code formatting
just check-type          # Run mypy type checking
just check-security      # Run bandit security scan
just format              # Format code with ruff

# Testing
uv run pytest           # Run test suite
uv run pytest -v        # Verbose test output

# Infrastructure
just terraform-plan     # Plan infrastructure changes
just terraform-apply    # Apply infrastructure
just terraform-destroy  # Destroy infrastructure

# Services
fraudsys list jobs       # List available jobs
fraudsys list services   # List available services
fraudsys job <name>      # Run specific job
fraudsys service <name>  # Run specific service
```

## Architecture

For detailed architecture documentation, see [docs/architecture_overview.md](docs/architecture_overview.md).

**High-Level Architecture:**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Raw Data        │───▶│ Feature          │───▶│ Feature Store   │
│ (Kaggle/Files)  │    │ Engineering      │    │ (S3 + Redis)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐           │
│ Live Stream     │───▶│ Feature Service  │───────────┤
│ (Producer)      │    │ (Kafka Consumer) │           │
└─────────────────┘    └──────────────────┘           │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Training        │───▶│ ML Pipeline      │    │ Prediction API  │
│ Pipeline        │    │ (MLflow)         │───▶│ (FastAPI)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌──────────────────┐           │
                       │ Monitoring       │◀──────────┘
                       │ (Prometheus)     │
                       └──────────────────┘
```

**Key Components:**
- **ML Pipeline**: Complete training workflow with experiment tracking
- **Feature Store**: Feast-based feature management with S3 offline and Redis online stores
- **Prediction API**: FastAPI service with real-time fraud detection
- **Streaming Pipeline**: Kafka-based real-time feature processing
- **Monitoring**: Prometheus metrics with Grafana visualization

## Configuration

The system uses **OmegaConf** with **Pydantic** for type-safe configuration management.

**Configuration Structure:**
- `confs/jobs/` - ML pipeline job configurations
- `confs/services/` - Service configurations
- `confs/env/` - Environment-specific overrides

**Example Job Configuration:**
```yaml
# confs/jobs/training.yaml
job:
  KIND: training
  model:
    KIND: xgboost
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  data_splitter:
    KIND: time_series
    test_size: 0.2
```

**Environment Variables:**
```bash
# .env file
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
AWS_REGION=eu-west-1
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Development

### Code Quality

The project uses comprehensive code quality tools:

```bash
just check-code      # Ruff linting
just check-format    # Code formatting
just check-type      # MyPy type checking
just check-security  # Bandit security scanning
just format          # Auto-format code
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_features.py -v
```

### Pre-commit Hooks

Automatically installed with `just install`:
- Code formatting with ruff
- Type checking with mypy
- Security scanning with bandit
- Conventional commit validation

### Adding New Components

**New Job:**
1. Create job class in `src/fraudsys/jobs/`
2. Add configuration in `confs/jobs/`
3. Register in job factory

**New Service:**
1. Create service class in `src/fraudsys/services/`
2. Add configuration in `confs/services/`
3. Add to docker-compose.yaml

**New Model:**
1. Implement model class in `src/fraudsys/core/models.py`
2. Add to model factory with KIND discriminator
3. Update configurations

## API Reference

### Prediction Endpoint

**POST** `/predict`

Predicts fraud probability for a transaction.

**Request Body:**
```json
{
  "customer_id": "string",
  "merchant_name": "string",
  "category": "string",
  "amount_usd": "number",
  "transaction_time": "string (ISO 8601)",
  "lat": "number",
  "long": "number",
  "city_pop": "integer",
  "merch_lat": "number",
  "merch_long": "number"
}
```

**Response:**
```json
{
  "transaction_id": "string",
  "is_fraud": "integer (0 or 1)",
  "model_version": "string",
  "timestamp": "string (ISO 8601)"
}
```

**Status Codes:**
- `200` - Successful prediction
- `400` - Invalid input data
- `422` - Validation error
- `500` - Internal server error

### Health Check

**GET** `/health`

Returns service health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "dependencies": {
    "mlflow": "connected",
    "kafka": "connected"
  }
}
```

## Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes with tests**
4. **Run quality checks**: `just check-code check-format check-type`
5. **Commit changes**: `git commit -m 'feat: add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**

**Commit Convention:**
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Test additions/modifications
- `chore:` - Maintenance tasks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using modern MLOps practices**

For questions or support, please open an issue or reach out to the development team.
