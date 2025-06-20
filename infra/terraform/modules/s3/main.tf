# s3 bucket for mlflow artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = var.mlflow_bucket_name
}

# Block public access
resource "aws_s3_bucket_public_access_block" "mlflow_artifacts_pab" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
