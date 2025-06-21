variable "aws_region" {
    type = string
    description = "The AWS region the project deploys to."
    default = "eu-west-1"
}

variable "environment" {
    type = string
    description = "The project environment. For example; dev, staging, or production."
    default = "dev"
}

variable "project_name" {
    type = string
    description = "The project name."
    default = "fraudsys"
}