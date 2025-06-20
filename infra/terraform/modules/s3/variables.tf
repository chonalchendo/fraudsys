variable "mlflow_bucket_name" {
  type        = string
  description = "Mlflow artifact bucket."
  default     = "mlflow-artifacts"
}

variable "tags" {
  type        = map(any)
  description = "The tags to apply to the resources"
}
