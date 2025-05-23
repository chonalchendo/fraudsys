# build services
[group('docker')]
docker-build: package
    docker compose build

# spin up services
[group('docker')]
docker-up: package
    docker compose up -d --build --remove-orphans

# spin down services
[group('docker')]
docker-down:
    docker compose down --remove-orphans
