#!/usr/bin/env bash
set -Eeuo pipefail

ACR_SERVER="${1:?ACR server is required}"
IMAGE_TAG="${2:?Image tag is required}"

[[ "$ACR_SERVER" =~ ^[a-z0-9.-]+\.azurecr\.io$ ]] || exit 1
[[ "$IMAGE_TAG" =~ ^[a-f0-9]{40}$ ]] || exit 1

cd /home/azureuser

BASE="docker-compose.yml"
OVERRIDE="cd-images.yml"
BACKUP="$(mktemp)"

had_override=0

if [[ -f "$OVERRIDE" ]]; then
    cp "$OVERRIDE" "$BACKUP"
    had_override=1
fi

restore_config() {
    if [[ "$had_override" == "1" ]]; then
        cp "$BACKUP" "$OVERRIDE"
    else
        rm -f "$OVERRIDE"
    fi
}

compose() {
    if [[ -f "$OVERRIDE" ]]; then
        sudo -n docker compose -f "$BASE" -f "$OVERRIDE" "$@"
    else
        sudo -n docker compose -f "$BASE" "$@"
    fi
}

rollback() {
    echo "Deployment failed. Rolling back..."

    restore_config

    compose up -d --no-deps --force-recreate api drift-exporter || true

    rm -f "$BACKUP"
}

trap rollback ERR

cat > "${OVERRIDE}.tmp" <<EOF
services:
  api:
    image: ${ACR_SERVER}/olist-api:${IMAGE_TAG}
  drift-exporter:
    image: ${ACR_SERVER}/olist-api:${IMAGE_TAG}
EOF

mv "${OVERRIDE}.tmp" "$OVERRIDE"

echo "Validating Docker Compose..."
compose config --quiet

echo "Pulling new images..."
compose pull api drift-exporter

echo "Updating API and Drift Exporter..."
compose up -d --no-deps --force-recreate api drift-exporter

echo "Checking API health..."

api_ok=0
for i in {1..12}; do
    if curl -fsS --max-time 5 http://localhost:8000/health >/dev/null; then
        api_ok=1
        break
    fi
    sleep 5
done

if [[ "$api_ok" != "1" ]]; then
    echo "API health check failed"
    false
fi

echo "Checking Drift Exporter..."

exporter_ok=0
for i in {1..12}; do
    if sudo -n docker compose -f "$BASE" -f "$OVERRIDE" exec -T \
        drift-exporter python -c \
        'import urllib.request; urllib.request.urlopen("http://localhost:8001/metrics", timeout=5)' \
        >/dev/null 2>&1; then
        exporter_ok=1
        break
    fi
    sleep 5
done

if [[ "$exporter_ok" != "1" ]]; then
    echo "Drift Exporter health check failed"
    false
fi

trap - ERR
rm -f "$BACKUP"

echo "DEPLOY SUCCESS: ${IMAGE_TAG}"
compose ps api drift-exporter
