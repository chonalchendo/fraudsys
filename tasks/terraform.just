# Run Terraform
[group('terraform')]
terraform-apply:
    cd infra/terraform && terraform apply -auto-approve

# Output Terraform values
[group('terraform')]
terraform-outputs:
    cd infra/terraform && terraform output -json > ../feast/terraform_outputs.json

# Destroy cloud resources
[group('terraform')]
terraform-destroy:
    cd infra/terraform && terraform destroy -auto-approve