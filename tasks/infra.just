# Set up the feature store using Terraform and Feast
[group('infra')]
infra-feature-store: terraform-apply terraform-outputs feast-sync-config feast-apply
