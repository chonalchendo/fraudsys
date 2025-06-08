FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app

# Copy wheel file and configuration
COPY dist/*.whl /app/
COPY confs/services /app/confs/services/

RUN uv pip install --system *.whl && \
    uv cache clean

# Set up the entry point to use uv run
ENTRYPOINT ["uv", "run"]
CMD ["fraudsys", "service", "feature"]
