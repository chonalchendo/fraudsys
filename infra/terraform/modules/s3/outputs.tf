output "mlflow_bucket_name" {
  value = var.mlflow_bucket_name
}

output "mlflow_bucket_arn" {
  value = aws_s3_bucket.mlflow_artifacts.arn
}

output "feast_bucket_name" {
  value = var.feast_bucket_name
}

output "feast_bucket_arn" {
  value = aws_s3_bucket.feast_offline_store.arn
}
