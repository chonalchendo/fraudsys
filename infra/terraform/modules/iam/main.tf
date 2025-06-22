resource "aws_iam_user" "name" {
  name = var.mlflow_user_name
}

resource "aws_s3_bucket_policy" "mlflow_bucket_policy" {
  bucket = data.aws_s3_bucket.mlflow_bucket.id
  policy = data.aws_iam_policy_document.mlflow_bucket_policy.json
}

resource "aws_iam_access_key" "mlflow_user_key" {
  user = aws_iam_user.name.name
}

data "aws_s3_bucket" "mlflow_bucket" {
  bucket = var.mlflow_bucket_name
}

data "aws_iam_policy_document" "mlflow_bucket_policy" {
  statement {
    principals {
      type        = "AWS"
      identifiers = [aws_iam_user.name.arn]
    }
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket"
    ]
    resources = [
      var.mlflow_bucket_arn,
      "${var.mlflow_bucket_arn}/*"
    ]
  }
}
