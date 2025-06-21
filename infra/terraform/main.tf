locals {
  mlflow_bucket_name = "${var.project_name}-${var.environment}-mlflow-artifacts"
  tags = {
    project = "fraudsys"
  }
}

module "s3" {
  source             = "./modules/s3"
  mlflow_bucket_name = local.mlflow_bucket_name
  tags               = local.tags
}
