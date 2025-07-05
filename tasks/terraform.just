# Initialize Terraform
[group('terraform')]
terraform-init:
    cd infra/terraform && terraform init

# Plan Terraform changes
[group('terraform')]
terraform-plan:
    cd infra/terraform && terraform init && terraform plan

# Run Terraform
[group('terraform')]
terraform-apply:
    cd infra/terraform && terraform init && terraform apply -auto-approve

# Output Terraform values
[group('terraform')]
terraform-outputs:
    cd infra/terraform && terraform init && terraform output -json > ../feast/terraform_outputs.json

# Destroy cloud resources
[group('terraform')]
terraform-destroy:
    cd infra/terraform && terraform init && terraform destroy -auto-approve