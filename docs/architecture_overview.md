# Fraud Detection System Architecture

## Overview

This is an online fraud detection system designed to provide real-time fraud predictions for incoming transactions. The system combines batch and streaming features to deliver accurate fraud detection using machine learning models that adapt to changing patterns over time.

## System Architecture

### Core Components

1. **Batch Feature Engineering**: Historical aggregation features computed offline
2. **Streaming Features**: Real-time features computed from live transaction streams  
3. **Feature Store (Feast)**: Centralized feature storage for both batch and streaming features
4. **ML Pipeline**: Training, validation, and model promotion workflows
5. **API Service**: FastAPI endpoint for real-time fraud prediction
6. **Model Registry (MLflow)**: Version control and tracking for models and experiments

### Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Historical Data │───▶│ Batch Features   │───▶│ Feature Store   │
│ (Parquet)       │    │ (S3)             │    │ (Feast)         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐           │
│ Live Stream     │───▶│ Streaming        │───────────┤
│ (Kafka)         │    │ Features         │           │
└─────────────────┘    └──────────────────┘           │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Training Data   │───▶│ ML Pipeline      │    │ Prediction API  │
│ (inputs/targets)│    │ (MLflow)         │    │ (FastAPI)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Feature Strategy

### Batch Features (Historical Aggregations)

**Purpose**: Capture long-term behavioral patterns and trends

**Examples**:
- Customer transaction statistics (1h, 6h, 1d, 7d, 30d windows)
- Merchant risk indicators and fraud rates
- Customer behavioral patterns and spending habits
- Geographic and temporal patterns

**Computation**: 
- Run offline using historical transaction data
- Computed using time windows relative to a reference time
- Stored in feature store for retrieval during inference

### Streaming Features (Real-time)

**Purpose**: Capture immediate context and short-term patterns

**Examples**:
- Transactions in last 10 minutes
- Current session activity
- Real-time velocity patterns
- Immediate geographic anomalies

**Computation**:
- Computed in real-time from live transaction streams
- Updated continuously as new transactions arrive
- Stored in feature store online storage

## Training vs. Inference Workflow

### Offline Model Training

1. **Historical Feature Engineering**:
   - Set `reference_time` to appropriate historical date
   - Compute batch features for all customers/merchants in training period
   - Features represent the "state" as of the reference time

2. **Model Training**:
   - Use `inputs_train.parquet` with computed batch features
   - Validate on `inputs_test.parquet` 
   - Track experiments in MLflow

3. **Model Promotion**:
   - Evaluate model performance
   - Promote best model to "Champion" status
   - Deploy to online feature store

### Online Inference

1. **Feature Retrieval**:
   - **Batch features**: Retrieved from feature store (pre-computed)
   - **Streaming features**: Retrieved from feature store (real-time)
   - **Transaction features**: Extracted from incoming transaction

2. **Prediction**:
   - Combine all feature types
   - Score transaction using deployed model
   - Return fraud probability/decision

3. **Feature Updates**:
   - Update streaming features with new transaction
   - Trigger any real-time aggregation updates

## Data Lifecycle

### Training Data
- `inputs_train.parquet`: Historical transactions (2019-2020)
- `targets_train.parquet`: Fraud labels for training
- `inputs_test.parquet`: Validation transactions
- `targets_test.parquet`: Validation labels

### Production Simulation
- `inputs_prod.parquet`: Simulates live transaction stream
- `targets_prod.parquet`: Labels for evaluation (not used in real-time)

### Model Retraining
- Periodically retrain model on accumulated production data
- Adapt to concept drift and new fraud patterns
- Use sliding window approach for training data

## Feature Time Windows

### Batch Feature Windows
- **1h**: Immediate recent activity
- **6h**: Short-term patterns  
- **1d**: Daily behavioral patterns
- **7d**: Weekly trends and seasonality
- **30d**: Long-term behavioral baselines

### Reference Time Strategy
- **Training**: Set to end of historical data period
- **Production**: Use current time (`datetime.now()`)
- **Retraining**: Set to end of training period for new model

## Infrastructure

### AWS Components
- **S3**: Feature storage and model artifacts
- **IAM**: Access control for services
- **Terraform**: Infrastructure as code

### Local Development
- **Docker Compose**: Local feature store and database
- **MLflow**: Experiment tracking and model registry
- **PostgreSQL**: Metadata storage

### Feature Store (Feast)
- **Offline Store**: Historical features (S3)
- **Online Store**: Real-time features (Redis/PostgreSQL)
- **Registry**: Feature definitions and metadata

## API Endpoints

### Prediction Service
```
POST /predict
{
  "transaction_id": "...",
  "customer_id": "...", 
  "merchant_name": "...",
  "amount_usd": 100.0,
  "transaction_time": "2024-01-01T12:00:00",
  ...
}

Response:
{
  "fraud_probability": 0.85,
  "is_fraud": true,
  "model_version": "v1.2.3",
  "features_used": {...}
}
```

### Monitoring
- Prometheus metrics for model performance
- Feature drift detection
- Prediction latency and throughput monitoring

## Key Design Principles

1. **Point-in-Time Consistency**: Features represent state at prediction time
2. **No Data Leakage**: Only use data available at prediction time
3. **Low Latency**: <100ms prediction response time
4. **Scalability**: Handle high transaction volumes
5. **Adaptability**: Continuous learning from new data
6. **Observability**: Comprehensive monitoring and alerting

## Development Workflow

1. **Feature Development**: Design and implement new features
2. **Offline Testing**: Validate features on historical data
3. **Model Training**: Train models with new feature sets
4. **A/B Testing**: Compare model versions in production
5. **Deployment**: Roll out successful models to production
6. **Monitoring**: Track performance and detect drift
7. **Retraining**: Adapt models to new patterns