locals {
  mlflow_bucket_name = "${var.project_name}-${var.environment}-mlflow-artifacts"
  feast_bucket_name  = "${var.project_name}-${var.environment}-feast-offline-store"
  mlflow_user_name   = "fraudsys"
}

module "s3" {
  source             = "./modules/s3"
  mlflow_bucket_name = local.mlflow_bucket_name
  feast_bucket_name  = local.feast_bucket_name
}

module "iam" {
  source             = "./modules/iam"
  mlflow_user_name   = local.mlflow_user_name
  mlflow_bucket_name = local.mlflow_bucket_name
  mlflow_bucket_arn  = module.s3.mlflow_bucket_arn
}
