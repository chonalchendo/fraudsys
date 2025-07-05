# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fraud detection system built with Python that implements a complete ML pipeline for fraud detection. The project uses a configuration-driven approach with YAML files to define jobs and services, and includes MLflow for model tracking and Feast for feature management.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

- **`src/fraudsys/`**: Main package containing all core functionality
  - **`core/`**: Core ML components (features, models, pipelines, schemas, metrics)
  - **`io/`**: Input/output handling (configs, datasets, Kafka, registries, runtimes)
  - **`jobs/`**: Job definitions for different pipeline stages (training, inference, evaluation, etc.)
  - **`services/`**: Service definitions (API, feature service, monitoring, producer)
  - **`utils/`**: Utility functions (samplers, searchers, signers, splitters)
- **`confs/`**: Configuration files for jobs and services
- **`tasks/`**: Just task definitions for various operations
- **`infra/`**: Infrastructure code (Feast, Terraform)
- **`notebooks/`**: Jupyter notebooks for development and analysis

## Common Development Commands

### Setup and Installation
```bash
# Install dependencies
uv sync

# Install git hooks
just install-hooks
```

### Code Quality
```bash
# Run all checks
just check-code check-format check-type check-security

# Format code
just format

# Individual checks
just check-code      # Lint with ruff
just check-format    # Check formatting
just check-type      # Type checking with mypy
just check-security  # Security scan with bandit
```

### Running Jobs and Services
```bash
# List available jobs and services
fraudsys list jobs
fraudsys list services

# Run specific jobs
fraudsys job training
fraudsys job offline_promotion
fraudsys job training_evaluation

# Run services
fraudsys service api
fraudsys service monitoring
```

### ML Pipeline
```bash
# Run complete training pipeline
just pipeline-training

# This runs the following sequence:
# 1. training
# 2. offline_promotion
# 3. training_evaluation
# 4. offline_inference
# 5. inference_evaluation
```

### Testing
```bash
# Run tests
uv run pytest

# Run tests with specific options
uv run pytest --verbosity=2
```

## Configuration System

The project uses OmegaConf for configuration management. Jobs and services are defined in YAML files under `confs/`:

- **Jobs**: `confs/jobs/` - ML pipeline jobs (training, inference, evaluation, etc.)
- **Services**: `confs/services/` - Long-running services (API, monitoring, etc.)

Configuration files use a `KIND` field to specify the type of component to instantiate, following a factory pattern.

## ML Pipeline Stages

The fraud detection pipeline follows these stages:

1. **Experiment**: Model selection and initial assessment
2. **Tuning**: Hyperparameter optimization
3. **Training**: Final model training
4. **Training Evaluation**: Performance assessment
5. **Offline Promotion**: Model promotion to "Champion" status
6. **Offline Inference**: Batch predictions
7. **Inference Evaluation**: Prediction quality assessment
8. **Online Promotion**: Production deployment

## Key Dependencies

- **ML/Data**: scikit-learn, xgboost, polars, pandas, numpy
- **ML Ops**: MLflow, Feast (feature store)
- **API**: FastAPI, uvicorn
- **Monitoring**: prometheus-client
- **Configuration**: OmegaConf, pydantic
- **Messaging**: kafka-python-ng
- **Development**: ruff, mypy, pytest, bandit

## Environment Setup

The project uses uv for dependency management and includes dependency groups for different purposes:
- `checks`: Code quality tools
- `commit`: Git hooks and commit tools
- `dev`: Development tools
- `doc`: Documentation tools
- `notebooks`: Jupyter kernel

## Docker and Infrastructure

Docker Compose is used for local development with services like MLflow and PostgreSQL. Terraform configurations are available in `infra/terraform/` for infrastructure management.