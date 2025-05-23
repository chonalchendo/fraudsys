# run clean tasks
[group('clean')]
clean: clean-build clean-constraints clean-mypy clean-pytest clean-python clean-requirements clean-ruff

# clean build folders
[group('clean')]
clean-build:
    rm -rf dist/
    rm -rf build/

# clean constraints file
[group('clean')]
clean-constraints:
    rm -rf constraints.txt

# clean mypy folders
[group('clean')]
clean-mypy:
    rm -rf .mypy_cache/

# clean pytest cache
[group('clean')]
clean-pytest:
    rm -rf .pytest_cache/

# clean python caches
[group('clean')]
clean-python:
    find . -type f -name '*.py[co]' -delete
    find . -type d -name __pycache__ -exec rm -r {} \+

# clean requirements file
[group('clean')]
clean-requirements:
    rm -f requirements.txt

# clean ruff cache
[group('clean')]
clean-ruff:
    rm -rf .ruff_cache/

# clean venv folder
[confirm]
[group('clean')]
clean-venv:
    rm -rf .venv/

# clean data folder
[group('clean')]
clean-data:
    rm -rf data/