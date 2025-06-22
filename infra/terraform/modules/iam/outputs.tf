output "mlflow_access_key_id" {
  description = "Access key ID for MLflow user"
  value       = aws_iam_access_key.mlflow_user_key.id
}

output "mlflow_secret_access_key" {
  description = "Secret access key for MLflow user"
  value       = aws_iam_access_key.mlflow_user_key.secret
  sensitive   = true
}

output "mlflow_user_arn" {
  description = "ARN of the MLflow user"
  value       = aws_iam_user.name.arn
}