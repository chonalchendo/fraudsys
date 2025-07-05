# Project user for all AWS resources
resource "aws_iam_user" "project_user" {
  name = var.project_user_name
  
  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Access key for the project user
resource "aws_iam_access_key" "project_user_key" {
  user = aws_iam_user.project_user.name
}

# Unified IAM policy for all project S3 buckets
resource "aws_iam_user_policy" "project_s3_policy" {
  name = "${var.project_name}-${var.environment}-s3-access"
  user = aws_iam_user.project_user.name

  policy = data.aws_iam_policy_document.project_s3_policy.json
}

# Policy document for S3 access to all project buckets
data "aws_iam_policy_document" "project_s3_policy" {
  statement {
    sid = "ListAllBuckets"
    actions = [
      "s3:ListAllMyBuckets",
      "s3:GetBucketLocation"
    ]
    resources = ["*"]
  }

  statement {
    sid = "ProjectBucketAccess"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketVersioning",
      "s3:PutBucketVersioning"
    ]
    resources = [
      var.mlflow_bucket_arn,
      "${var.mlflow_bucket_arn}/*",
      var.feast_bucket_arn,
      "${var.feast_bucket_arn}/*"
    ]
  }
}
