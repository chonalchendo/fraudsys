# Fraud Detection System Architecture

## Overview

This is a production-ready online fraud detection system designed to provide real-time fraud predictions for incoming transactions. The system combines batch and streaming features using a modern MLOps stack with Feast feature store, MLflow model registry, Kafka streaming, and comprehensive monitoring.

## System Architecture

### Core Components

1. **ML Pipeline Jobs**: Complete training workflow from data extraction to model promotion
2. **Real-time Services**: API endpoint, feature processing, monitoring, and data simulation
3. **Feature Store (Feast)**: Centralized feature storage for both batch and streaming features
4. **Model Registry (MLflow)**: Version control, tracking, and serving for ML models
5. **Streaming Infrastructure (Kafka)**: Real-time data processing and event streaming
6. **Monitoring Stack**: Prometheus metrics, Grafana dashboards, and health monitoring

### Data Flow Architecture

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

## Code Architecture and Organization

### Source Code Structure (`src/fraudsys/`)

#### **Core ML Components (`core/`)**
- **Features**: Feature definition, cleaning, and validation logic
- **Models**: Model training, evaluation, and serving components
- **Pipelines**: End-to-end ML workflow orchestration
- **Schemas**: Data validation and type checking with Pydantic
- **Metrics**: Model performance evaluation and monitoring metrics

#### **Input/Output Layer (`io/`)**
- **Configs**: Configuration management using OmegaConf
- **Datasets**: Parquet/JSON loaders with S3 support and Polars/Pandas backends
- **Kafka**: Real-time message streaming and event processing
- **Registries**: MLflow integration for model and experiment management
- **Runtimes**: Execution environment and resource management

#### **Jobs (`jobs/`)**
- **Base Job Classes**: Common functionality for context and resource management
- **Data Jobs**: Feature engineering, extraction, and transformation
- **Training Jobs**: Model experimentation, tuning, and training
- **Evaluation Jobs**: Model validation and performance assessment
- **Promotion Jobs**: Model deployment and version management

#### **Services (`services/`)**
- **Base Service Classes**: Long-running process management
- **API Service**: FastAPI-based prediction endpoint
- **Feature Service**: Real-time feature processing pipeline
- **Monitoring Service**: Metrics collection and health monitoring
- **Producer Service**: Data simulation and load testing

#### **Utilities (`utils/`)**
- **Samplers**: Data sampling strategies for training/testing
- **Searchers**: Hyperparameter optimization utilities
- **Signers**: Security and authentication helpers
- **Splitters**: Time-aware data splitting for ML workflows

### Configuration System

**Framework**: OmegaConf with Pydantic validation
**Pattern**: Factory pattern using `KIND` discriminator fields
**Structure**:
- `confs/jobs/` - Job configurations (YAML)
- `confs/services/` - Service configurations (YAML)
- `confs/env/` - Environment-specific overrides

**Features**:
- Type-safe configuration with Pydantic models
- Hierarchical configuration with inheritance
- Environment variable interpolation
- Dynamic object instantiation via factory pattern

## Detailed Job Breakdown

### **Data Pipeline Jobs**

#### 1. **Extract Job** (`extract.yaml`)
- **Purpose**: Downloads fraud detection dataset from Kaggle
- **Dataset**: `kartik2112/fraud-detection` (1M+ credit card transactions)
- **Processing**: Splits into training (70%), testing (20%), production (10%)
- **Outputs**:
  - Training: `inputs_train.parquet`, `targets_train.parquet`
  - Testing: `inputs_test.parquet`, `targets_test.parquet`
  - Production: `inputs_prod.parquet`, `targets_prod.parquet`

#### 2. **Feature Engineering Job** (`feature_engineering.yaml`)
- **Purpose**: Computes time-windowed aggregation features for feature store
- **Technology**: Polars for high-performance aggregations
- **Features Computed**:
  - **Customer Stats**: Transaction counts, amounts, merchant/category diversity (1h-30d windows)
  - **Merchant Stats**: Fraud rates, transaction patterns, customer diversity (1d-30d windows)
  - **Customer Behavior**: Velocity patterns, location analysis, spending habits
- **Time Windows**: 1h, 6h, 1d, 7d, 30d configurable windows
- **Outputs**: S3-stored parquet files for Feast offline store

#### 3. **Offline Features Job** (`offline_features.yaml`)
- **Purpose**: Enriches raw transaction data with pre-computed features
- **Process**: Joins raw data with Feast offline store features
- **Outputs**: Feature-enriched datasets ready for ML training

### **ML Pipeline Jobs**

#### 4. **Experiment Job** (`experiment.yaml`)
- **Purpose**: Model selection and baseline performance assessment
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Validation**: Time series cross-validation with proper temporal splitting
- **Metrics**: Precision, Recall, F1, AUC-ROC, AUC-PR
- **MLflow Integration**: Experiments logged with parameters and metrics

#### 5. **Tuning Job** (`tuning.yaml`)
- **Purpose**: Hyperparameter optimization for XGBoost
- **Method**: Random search with extensive parameter grid
- **Parameters**:
  - Tree structure: `n_estimators`, `max_depth`, `min_child_weight`
  - Learning: `learning_rate`, `subsample`, `colsample_bytree`
  - Regularization: `reg_alpha`, `reg_lambda`
- **Validation**: Time series split with early stopping

#### 6. **Training Job** (`training.yaml`)
- **Purpose**: Final model training with optimized hyperparameters
- **Model**: XGBoost with fraud detection-specific configuration
- **Features**:
  - Time series split (80% train, 20% validation)
  - Class imbalance handling with `scale_pos_weight`
  - Early stopping with validation monitoring
- **MLflow Integration**: Model artifacts, metrics, and lineage tracking

### **Evaluation and Deployment Jobs**

#### 7. **Training Evaluation** (`training_evaluation.yaml`)
- **Purpose**: Comprehensive model performance validation
- **Metrics**: Classification report, confusion matrix, threshold analysis
- **Analysis**: Feature importance, prediction distribution, error analysis

#### 8. **Offline Promotion** (`offline_promotion.yaml`)
- **Purpose**: Promotes validated model to "Champion" alias in MLflow
- **Process**: Model registry update with metadata and tags
- **Gate**: Only promotes models meeting performance thresholds

#### 9. **Offline Inference** (`offline_inference.yaml`)
- **Purpose**: Batch prediction generation for model validation
- **Input**: Test dataset with features
- **Output**: Predictions with confidence scores saved to parquet

#### 10. **Inference Evaluation** (`inference_evaluation.yaml`)
- **Purpose**: Validates batch prediction quality
- **Analysis**: Prediction accuracy, calibration, and consistency checks

#### 11. **Online Promotion** (`online_promotion.yaml`)
- **Purpose**: Promotes Champion model to "Production" alias
- **Trigger**: After successful offline validation
- **Effect**: Makes model available for real-time API serving

#### 12. **Explanation Job** (`explanation.yaml`)
- **Purpose**: Model interpretability using SHAP values
- **Analysis**: Global feature importance, local explanations, interaction effects
- **Outputs**: SHAP values and feature importance rankings

## Detailed Service Breakdown

### **Core Application Services**

#### 1. **API Service** (`api.yaml`)
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Port**: 8000
- **Endpoints**:
  - `GET /health` - Service health check with dependency validation
  - `POST /predict` - Real-time fraud prediction endpoint

**Prediction Flow**:
1. **Input Validation**: Pydantic `RawTransaction` model validates incoming JSON
2. **Event Publishing**: Raw transaction published to `raw-transactions` Kafka topic
3. **Feature Processing**:
   - Converts to Polars DataFrame for efficient processing
   - Applies feature cleaning transformations
   - Converts to Pandas for model compatibility
   - Validates with `InputsSchema` using Pandera
4. **Model Inference**:
   - Lazy-loads Champion model from MLflow registry
   - Generates binary fraud prediction (0=legitimate, 1=fraud)
   - Includes confidence scoring
5. **Response Publishing**: Prediction published to `fraud-predictions` topic
6. **Response**: Returns JSON with transaction ID and prediction

**Dependencies**: MLflow model registry, Kafka messaging, feature processing pipeline

#### 2. **Feature Service** (`feature.yaml`)
- **Purpose**: Real-time feature processing and transformation
- **Architecture**: Kafka consumer-producer pattern
- **Processing Pipeline**:
  1. Consumes raw transactions from `raw-transactions` topic
  2. Applies feature cleaning and transformation logic
  3. Publishes processed transactions to `clean-transactions` topic
- **Benefits**: Decouples raw data ingestion from prediction serving

#### 3. **Producer Service** (`producer.yaml`)
- **Purpose**: Simulates real-time transaction data for testing and demonstration
- **Data Source**: Production parquet file (`inputs_prod.parquet`)
- **Behavior**:
  - Sends transactions to API endpoint at 1-second intervals
  - Includes health check waiting for API availability
  - Loops through production dataset continuously
- **Use Cases**: Load testing, demonstration, performance validation

#### 4. **Monitoring Service** (`monitoring.yaml`)
- **Purpose**: Real-time prediction monitoring with Prometheus metrics
- **Port**: 8001 (metrics endpoint)
- **Metrics Tracked**:
  - `fraud_predictions_total`: Total predictions by type (fraud/legitimate)
  - `fraud_rate_current`: Rolling fraud rate over last 1000 predictions
  - `predictions_per_minute`: Throughput monitoring
  - `prediction_confidence`: Confidence score distribution
- **Integration**: Consumes predictions from `fraud-predictions` Kafka topic

### **Infrastructure Services**

#### 5. **Prometheus** (Port 9090)
- **Purpose**: Metrics collection and alerting
- **Targets**: Monitors all application services
- **Configuration**: Service discovery via docker-compose labels

#### 6. **Grafana** (Port 3000)
- **Purpose**: Metrics visualization and dashboards
- **Integration**: Connected to Prometheus data source
- **Dashboards**: Real-time fraud detection metrics

## Docker Infrastructure

### **Orchestration**: Docker Compose with comprehensive service management

### **Core Infrastructure Stack**:
- **Zookeeper**: Kafka coordination and metadata management
- **Kafka**: Message broker for real-time data streaming (topics: raw-transactions, clean-transactions, fraud-predictions)
- **Redis**: Online feature store and caching layer
- **PostgreSQL**: MLflow backend store for experiments and models
- **MLflow**: Model registry and experiment tracking (port 5000)

### **Application Services**:
- **feature-api**: Main prediction API service
- **feature-producer**: Transaction data simulation
- **feature-cleaner**: Real-time feature processing
- **prediction-monitoring**: Metrics collection service

### **Monitoring and Management**:
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Visualization dashboard (port 3000)
- **Kafka-UI**: Kafka topic monitoring and management (port 8080)

### **Container Strategy**:
- **Base Image**: `ghcr.io/astral-sh/uv:python3.13-bookworm` for all services
- **Package Management**: UV for fast dependency installation
- **Deployment**: Wheel-based deployment for efficiency and reproducibility
- **Health Checks**: Comprehensive health checking across all services

## Feature Engineering Strategy

### **Batch Features (Historical Aggregations)**

**Purpose**: Capture long-term behavioral patterns and establish baselines

**Customer Transaction Statistics**:
- Transaction counts and amounts across time windows (1h, 6h, 1d, 7d, 30d)
- Merchant and category diversity metrics
- Geographic diversity (unique states/cities)
- Timing patterns (night transactions, weekend activity, business hours)

**Merchant Risk Indicators**:
- Historical fraud rates across time windows (1d, 7d, 30d)
- Transaction volume and amount patterns
- Customer diversity and repeat customer rates
- High-amount transaction frequency (95th percentile threshold)

**Customer Behavioral Patterns**:
- Transaction velocity and timing analysis
- Geographic movement patterns and distance calculations
- Spending pattern analysis and amount deviation detection
- Category concentration using Shannon entropy

### **Streaming Features (Real-time)**

**Purpose**: Capture immediate context and detect real-time anomalies

**Implementation**: Via Feature Service consuming Kafka streams
**Examples**:
- Transactions in last 10 minutes
- Current session activity patterns
- Real-time velocity detection
- Immediate geographic anomalies

### **Reference Time Strategy**
- **Training**: `"2020-03-07 00:00:00"` (just after historical data ends)
- **Production**: `null` (current time via `datetime.now()`)
- **Retraining**: End of new training data period

## Real-time Prediction Flow

### **End-to-End Request Processing**:

1. **Transaction Arrival**: POST request to `/predict` endpoint
2. **Input Validation**: Pydantic model validation with type checking
3. **Event Publishing**: Raw transaction sent to Kafka for audit trail
4. **Feature Processing**:
   - Real-time feature extraction and cleaning
   - Integration with pre-computed batch features from feature store
5. **Model Inference**:
   - Champion model loaded from MLflow registry
   - Prediction generated with confidence scoring
6. **Result Publishing**: Prediction sent to monitoring topic
7. **Response**: JSON response with prediction and metadata

### **Feature Integration**:
- **Transaction Features**: Extracted from incoming request
- **Batch Features**: Retrieved from Feast offline store (pre-computed)
- **Streaming Features**: Retrieved from Feast online store (real-time)
- **Combined Dataset**: Merged for model input

### **Performance Characteristics**:
- **Latency Target**: <100ms for prediction response
- **Throughput**: Designed for high-volume transaction processing
- **Scalability**: Horizontal scaling via container orchestration

## Model Lifecycle Management

### **Training Pipeline**:
```
Extract → Feature Engineering → Offline Features → Experiment → Tuning → Training → Evaluation → Offline Promotion
```

### **Deployment Pipeline**:
```
Offline Promotion → Validation → Online Promotion → Production Serving
```

### **Model Versioning**:
- **MLflow Registry**: Centralized model storage with version control
- **Aliases**: "Champion" (validated model), "Production" (serving model)
- **Metadata**: Performance metrics, feature schemas, training lineage

### **Continuous Learning**:
- **Data Accumulation**: Production data collected for retraining
- **Concept Drift Detection**: Model performance monitoring
- **Automated Retraining**: Triggered by performance degradation or data volume

## Monitoring and Observability

### **Application Metrics** (Prometheus):
- Prediction throughput and latency
- Fraud detection rates and accuracy
- Model confidence distributions
- Feature processing performance

### **Infrastructure Metrics**:
- Kafka topic lag and throughput
- API response times and error rates
- Database connection pooling
- Memory and CPU utilization

### **Business Metrics**:
- Real-time fraud rate tracking
- Geographic and temporal fraud patterns
- Merchant and customer risk distributions
- Financial impact assessment

## Key Architectural Strengths

1. **Production-Ready**: Comprehensive monitoring, health checks, and error handling
2. **Scalable Architecture**: Microservices with Kafka messaging for horizontal scaling
3. **Feature Store Integration**: Professional feature management with Feast
4. **Model Registry**: MLflow for experiment tracking and model versioning
5. **Type Safety**: Extensive Pydantic validation and type hints throughout
6. **Configuration-Driven**: Everything parameterized via YAML configurations
7. **Cloud-Native**: S3 integration, containerized deployment, infrastructure as code
8. **Real-time Processing**: Low-latency prediction serving with streaming features
9. **Observability**: Comprehensive metrics, logging, and monitoring capabilities
10. **Reproducibility**: Deterministic training pipelines with experiment tracking

The system demonstrates enterprise-grade ML engineering practices with proper separation of concerns, comprehensive testing infrastructure, and production-ready monitoring capabilities suitable for high-stakes financial fraud detection.
