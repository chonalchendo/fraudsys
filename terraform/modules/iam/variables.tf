variable "project_name" {
  type        = string
  description = "The project name"
}

variable "environment" {
  type        = string
  description = "The environment (dev, staging, prod)"
}

variable "project_user_name" {
  type        = string
  description = "Name of the project user for AWS resources"
}

variable "mlflow_bucket_arn" {
  type        = string
  description = "ARN of the MLflow bucket"
}

variable "feast_bucket_arn" {
  type        = string
  description = "ARN of the Feast bucket"
}
