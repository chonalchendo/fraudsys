# bump package
[group('commit')]
commit-bump:
    uv run cz bump

# commit package
[group('commit')]
commit-files: commit-pre-commit
    uv run cz commit

# get commit info
[group('commit')]
commit-info:
    uv run cz info

# run pre-commit on staged files
[group('commit')]
commit-pre-commit:
    uv run pre-commit
