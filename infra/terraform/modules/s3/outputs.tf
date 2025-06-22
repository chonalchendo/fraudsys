output "mlflow_bucket_name" {
  value = var.mlflow_bucket_name
}

output "mlflow_bucket_arn" {
  value = aws_s3_bucket.mlflow_artifacts.arn
}
