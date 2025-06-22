# s3 bucket for mlflow artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = var.mlflow_bucket_name
}