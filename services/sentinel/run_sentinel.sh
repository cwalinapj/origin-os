#!/bin/bash
# =============================================================================
# SENTINEL — Run Verification Guardian
# =============================================================================
# Options:
#   --single    Run once and exit (for testing)
#   --daemon    Run continuously (default)
# =============================================================================

set -euo pipefail

source /Users/root1/.env

MODE="${1:-daemon}"

echo "🛡️  SENTINEL VERIFICATION GUARDIAN"
echo ""

# Build container
docker build -t origin-sentinel /Users/root1/services/sentinel

if [ "$MODE" == "--single" ]; then
    echo "Running single verification..."
    docker run --rm \
        -e S3_ENDPOINT="https://s3.us-west-004.backblazeb2.com" \
        -e S3_KEY="$B2_KEY_ID" \
        -e S3_SECRET="$B2_APP_KEY" \
        -e BACKUP_BUCKET="origin-os-backups" \
        -e CODEX_URL="http://host.docker.internal:8001" \
        -e VAULT_URL="http://host.docker.internal:8004" \
        -e SINGLE_RUN="true" \
        origin-sentinel
else
    echo "Starting continuous verification daemon..."
    docker run -d \
        --name origin-sentinel \
        --restart unless-stopped \
        -e S3_ENDPOINT="https://s3.us-west-004.backblazeb2.com" \
        -e S3_KEY="$B2_KEY_ID" \
        -e S3_SECRET="$B2_APP_KEY" \
        -e BACKUP_BUCKET="origin-os-backups" \
        -e CODEX_URL="http://origin-codex:8000" \
        -e VAULT_URL="http://origin-vault:8000" \
        -e CHECK_INTERVAL="3600" \
        --network origin-os_default \
        origin-sentinel
    
    echo "Sentinel started. View logs with: docker logs -f origin-sentinel"
fi
