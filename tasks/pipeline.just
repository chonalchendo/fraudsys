# run model training pipeline
[group('pipeline')]
pipeline-training: (docker-up-service "mlflow")
    uv run fraudsys job training
    uv run fraudsys job offline_promotion
    uv run fraudsys job training_evaluation
    uv run fraudsys job offline_inference
    uv run fraudsys job inference_evaluation
