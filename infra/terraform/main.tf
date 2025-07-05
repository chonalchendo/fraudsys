locals {
  # Project user for all AWS resources
  project_user_name = var.project_name

  # S3 bucket names
  mlflow_bucket_name = "${var.project_name}-${var.environment}-mlflow-artifacts"
  feast_bucket_name  = "${var.project_name}-${var.environment}-feast-offline-store"
}

module "s3" {
  source = "./modules/s3"

  project_name = var.project_name
  environment  = var.environment

  mlflow_bucket_name = local.mlflow_bucket_name
  feast_bucket_name  = local.feast_bucket_name
}

module "iam" {
  source = "./modules/iam"

  project_name      = var.project_name
  environment       = var.environment
  project_user_name = local.project_user_name

  # Bucket ARNs from S3 module
  mlflow_bucket_arn = module.s3.mlflow_bucket_arn
  feast_bucket_arn  = module.s3.feast_bucket_arn
}
