output "mlflow_access_key_id" {
  description = "Access key ID for MLflow user"
  value       = module.iam.mlflow_access_key_id
}

output "mlflow_secret_access_key" {
  description = "Secret access key for MLflow user"
  value       = module.iam.mlflow_secret_access_key
  sensitive   = true
}
