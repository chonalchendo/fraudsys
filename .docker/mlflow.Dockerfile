FROM ghcr.io/mlflow/mlflow:v2.20.3

# Install PostgreSQL driver
RUN pip install psycopg2-binary
