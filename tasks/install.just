# install git hooks
[group('install')]
install-hooks:
    uv run pre-commit install --hook-type=pre-push
    uv run pre-commit install --hook-type=commit-msg