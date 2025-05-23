# code code quality
[group('check')]
check-code:
  uv run ruff check {{SOURCES}} {{TESTS}}

# check code format
[group('check')]
check-format:
  uv run ruff format --check {{SOURCES}} {{TESTS}}

# check code typing
[group('check')]
check-type:
  uv run mypy {{SOURCES}} {{TESTS}}