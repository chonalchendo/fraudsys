variable "mlflow_bucket_name" {
  type        = string
  description = "Mlflow artifact bucket."
  default     = "mlflow-artifacts"
}

variable "feast_bucket_name" {
  type        = string
  description = "Feast offline store bucket."
  default     = "feast-offline-store"
}
