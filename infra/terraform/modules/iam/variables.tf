variable "mlflow_bucket_name" {
  type        = string
  description = "Mlflow artifact bucket."
  default     = "mlflow-artifacts"
}

variable "mlflow_user_name" {
  type = string
}

variable "mlflow_bucket_arn" {
  type = string
}