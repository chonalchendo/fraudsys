# Apply feast configuration
[group('feast')]
feast-apply:
    cd infra/feast && feast apply

# Validate feast configuration
[group('feast')]
feast-validate:
    cd infra/feast && feast validate

# List feast features
[group('feast')]
feast-list:
    cd infra/feast && feast feature-views list
    cd infra/feast && feast entities list
    cd infra/feast && feast feature-services list

# Run feature engineering pipeline
[group('feast')]
feast-compute-features:
    uv run fraudsys job feature_engineering

# Materialise data to online store
[group('feast')]
feast-materialize:
    cd infra/feast && feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

# Start feast ui
[group('feast')]
feast-ui:
    cd infra/feast && feast ui

# Tear down feast deployment
[group('feast')]
feast-teardown:
    cd infra/feast && feast teardown

# Generate feast config from terraform 
[group('feast')]
feast-sync-config:
    uv run python tools/feast_sync_config.py
