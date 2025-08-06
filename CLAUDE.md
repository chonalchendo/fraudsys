# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fraud detection system built with Python that implements a complete ML pipeline for fraud detection. The project uses a configuration-driven approach with YAML files to define jobs and services, and includes MLflow for model tracking. The system is designed to operate as a real-time fraud detection system using Docker Compose for orchestration.

## Project Structure

The codebase follows a modular architecture with clear separation of concerns:

```
fraudsys/
├── __init__.py, __main__.py, cli.py, settings.py
├── constants.py                    # System-wide constants
├── schemas.py                      # Data validation schemas (pandera)
├── exceptions.py                   # Custom exceptions
├── app/                            # Application layer
│   ├── jobs/                       # Batch processing jobs
│   │   ├── data/                   # Data processing jobs
│   │   │   ├── extract.py
│   │   │   ├── feature_engineering.py
│   │   │   ├── offline_feature_engineering.py
│   │   │   └── offline_features.py
│   │   └── ml/                     # ML pipeline jobs
│   │       ├── evaluation.py
│   │       ├── experiment.py
│   │       ├── explanation.py
│   │       ├── inference.py
│   │       ├── promotion.py
│   │       ├── training.py
│   │       └── tuning.py
│   └── services/                   # Long-running services
│       ├── api/                    # REST API service
│       ├── monitoring.py           # Monitoring service
│       ├── producer.py             # Kafka producer service
│       └── transformation.py       # Kafka consumer/transformer service
├── config/                         # Configuration management
│   ├── __init__.py
│   └── parser.py                   # YAML/JSON config parsing utilities
├── data/                           # Data handling
│   ├── __init__.py
│   └── datasets.py                 # Dataset loading utilities
├── features/                       # Feature engineering & serving
│   ├── engineering/                # Feature computation
│   │   ├── aggregations.py
│   │   └── transformations.py
│   └── store/                      # Feature store implementation
│       ├── client.py
│       └── definitions.py
├── ml/                             # Machine learning components
│   ├── metrics.py                  # Model evaluation metrics
│   ├── models.py                   # ML model implementations
│   ├── pipelines.py                # ML preprocessing pipelines
│   ├── sampler.py                  # Data sampling utilities
│   ├── searchers.py                # Hyperparameter search utilities
│   └── splitters.py                # Data splitting utilities
└── infra/                          # Infrastructure & external integrations
    ├── kafka.py                    # Kafka integration
    ├── logging.py                  # Logging configuration (Logger class)
    └── mlflow/                     # MLflow integration
        ├── client.py
        ├── registries.py
        └── signers.py
```

## Running the Project

### Command Line Interface

The project uses a configuration-driven CLI system:

```bash
# Run jobs (batch processing)
uv run fraudsys job <job_name>

# Run services (long-running processes)
uv run fraudsys service <service_name>

# List available jobs and services
uv run fraudsys list jobs
uv run fraudsys list services
```

### Available Jobs

**Data Jobs** (located in `confs/jobs/data/`):
- `extract` - Data extraction job
- `feature_engineering` - Feature engineering job
- `offline_feature_engineering` - Offline feature engineering job
- `offline_features` - Offline feature processing job

**ML Jobs** (located in `confs/jobs/ml/`):
- `experiment` - Model experimentation job
- `explanation` - Model explanation job
- `inference_evaluation` - Inference evaluation job
- `offline_inference` - Offline inference job
- `offline_promotion` - Offline model promotion job
- `online_promotion` - Online model promotion job
- `training` - Model training job
- `training_evaluation` - Training evaluation job
- `tuning` - Hyperparameter tuning job

### Available Services

**Services** (located in `confs/services/`):
- `api` - REST API service
- `monitoring` - Monitoring service
- `producer` - Kafka producer service
- `transformation` - Kafka consumer/transformer service

## Setup and Installation

```bash
# Install dependencies
uv sync

# Install git hooks
just install-hooks
```

## Code Quality and Development

### Code Formatting and Linting

The project uses the following tools (configured in `pyproject.toml`):
- **ruff**: Code linting and formatting (line length: 88, target: Python 3.13)
- **mypy**: Type checking
- **bandit**: Security scanning
- **pytest**: Testing

```bash
# Format code
just format

# Run all checks
just check-code check-format check-type check-security

# Individual checks
just check-code      # Lint with ruff
just check-format    # Check formatting
just check-type      # Type checking with mypy
just check-security  # Security scan with bandit
```

### Pre-commit Hooks

The project uses pre-commit hooks (`.pre-commit-config.yaml`) that run automatically on commit:
- **File checks**: Large files, case conflicts, merge conflicts, TOML/YAML validation
- **Code quality**: Ruff linting and formatting, Bandit security scanning
- **Git**: End-of-file fixer, trailing whitespace removal
- **Commits**: Commitizen for conventional commits
- **Terraform**: Formatting and validation (if applicable)

### Testing

```bash
# Run tests
uv run pytest

# Run tests with specific options
uv run pytest --verbosity=2
```

## Configuration System

The project uses OmegaConf for configuration management:
- **Jobs**: `confs/jobs/data/` and `confs/jobs/ml/` - Batch processing jobs
- **Services**: `confs/services/` - Long-running services

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

## Real-time Fraud Detection System

The project is designed as a real-time fraud detection system with the following characteristics:

- **Orchestration**: Docker Compose for service orchestration
- **Docker Configuration**: Dockerfiles will be located in `.docker/` folder (when implemented)
- **Streaming**: Kafka-based real-time data processing
- **API**: FastAPI REST API for real-time fraud predictions
- **Monitoring**: Prometheus-based monitoring and alerting
- **Feature Store**: Custom feature store for real-time feature serving

## Key Dependencies

- **ML/Data**: scikit-learn, xgboost, polars, pandas, numpy
- **ML Ops**: MLflow (model tracking and registry)
- **API**: FastAPI, uvicorn
- **Streaming**: kafka-python-ng
- **Monitoring**: prometheus-client
- **Configuration**: OmegaConf, pydantic
- **Data Validation**: pandera (for schema validation)
- **Development**: ruff, mypy, pytest, bandit

## Environment Setup

The project uses uv for dependency management with the following groups:
- `checks`: Code quality tools (bandit, mypy, pytest, ruff)
- `commit`: Git hooks and commit tools (commitizen, pre-commit)
- `dev`: Development tools (rust-just)
- `doc`: Documentation tools (pdoc)
- `notebooks`: Jupyter kernel (ipykernel)

## Ignored Files

The following files/directories are ignored in git (`.gitignore`):
- Python artifacts: `__pycache__/`, `*.pyc`, `build/`, `dist/`, `*.egg-info`
- Environment: `.venv`, `.env`
- Tool caches: `.ruff_cache`, `.mypy_cache`
- Data: `data/` directory
- Services: `mlruns`, `postgres-data`, `notebooks`
- Logging: `stderr`

## Docker and Infrastructure

- **Docker Compose**: Used for local development and production orchestration
- **Services**: MLflow, PostgreSQL, Kafka, Redis (as needed)
- **Terraform**: Infrastructure as Code configurations available in `infra/terraform/`

## Entry Point

The CLI entry point is configured in `pyproject.toml`:
```toml
[project.scripts]
fraudsys = "fraudsys.cli:execute"
```

This enables the `uv run fraudsys` command to execute the CLI interface.

# Python Package Management with uv

Use uv exclusively for Python package management in this project.

## Package Management Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

## Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
- Launch a Python repl with `uv run python`

## Managing Scripts with PEP 723 Inline Metadata

- Run a Python script with inline metadata (dependencies defined at the top of the file) with: `uv run script.py`
- You can add or remove dependencies manually from the `dependencies =` section at the top of the script, or
- Or using uv CLI:
    - `uv add package-name --script script.py`
    - `uv remove package-name --script script.py`
