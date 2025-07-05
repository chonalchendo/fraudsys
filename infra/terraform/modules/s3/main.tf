# s3 bucket for mlflow artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = var.mlflow_bucket_name
}

# s3 bucket for feast offline store
resource "aws_s3_bucket" "feast_offline_store" {
  bucket = var.feast_bucket_name
}
