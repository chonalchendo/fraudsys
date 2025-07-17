output "project_access_key_id" {
  description = "Access key ID for project user"
  value       = module.iam.project_access_key_id
}

output "project_secret_access_key" {
  description = "Secret access key for project user"
  value       = module.iam.project_secret_access_key
  sensitive   = true
}

output "project_user_name" {
  description = "Name of the project user"
  value       = module.iam.project_user_name
}

output "mlflow_bucket_name" {
  description = "Name of the MLflow bucket"
  value       = module.s3.mlflow_bucket_name
}

output "feast_bucket_name" {
  description = "Name of the Feast bucket"
  value       = module.s3.feast_bucket_name
}
