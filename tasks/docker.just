# build services
[group('docker')]
docker-build: package
    docker compose build

# spin up all services
[group('docker')]
docker-up: package
    docker compose up -d --build --remove-orphans

# spin down all services
[group('docker')]
docker-down:
    docker compose down --remove-orphans

# spin up a service
[group('docker')]
docker-up-service service:
    docker compose up -d --build {{service}}

# spint down a service
[group('docker')]
docker-down-service service:
    docker compose down {{service}}
