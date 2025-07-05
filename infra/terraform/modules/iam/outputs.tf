output "project_access_key_id" {
  description = "Access key ID for project user"
  value       = aws_iam_access_key.project_user_key.id
}

output "project_secret_access_key" {
  description = "Secret access key for project user"
  value       = aws_iam_access_key.project_user_key.secret
  sensitive   = true
}

output "project_user_arn" {
  description = "ARN of the project user"
  value       = aws_iam_user.project_user.arn
}

output "project_user_name" {
  description = "Name of the project user"
  value       = aws_iam_user.project_user.name
}
