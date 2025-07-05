variable "project_name" {
  type        = string
  description = "The project name"
}

variable "environment" {
  type        = string
  description = "The environment (dev, staging, prod)"
}

variable "mlflow_bucket_name" {
  type        = string
  description = "MLflow artifact bucket name"
}

variable "feast_bucket_name" {
  type        = string
  description = "Feast offline store bucket name"
}
