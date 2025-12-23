#!/bin/bash
# =============================================================================
# ORIGIN OS RESTORE — Runner Script
# =============================================================================
# Usage:
#   ./run_restore.sh <backup_timestamp> <volume_name> <environment> <mode>
#
# Example:
#   ./run_restore.sh 20251223_185941 origin-vault-data sandbox forensic
#
# Environments: sandbox | staging | forensics
# Modes: read_only | clone | forensic
# =============================================================================

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <backup_timestamp> <volume_name> <environment> <mode>"
    echo ""
    echo "Example:"
    echo "  $0 20251223_185941 origin-vault-data sandbox forensic"
    echo ""
    echo "Environments: sandbox | staging | forensics"
    echo "Modes: read_only | clone | forensic"
    exit 1
fi

TIMESTAMP="$1"
VOLUME_NAME="$2"
ENVIRONMENT="$3"
MODE="$4"

source /Users/root1/.env

# First, get the checksum from the manifest
echo "=== Fetching manifest to get checksum ==="

MANIFEST_KEY="backups/${TIMESTAMP}/manifest.json"
MANIFEST_LOCAL="/tmp/manifest_${TIMESTAMP}.json"

docker run --rm \
    -e AWS_ACCESS_KEY_ID="$B2_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$B2_APP_KEY" \
    -v /tmp:/tmp \
    amazon/aws-cli \
    s3 cp "s3://origin-os-backups/${MANIFEST_KEY}" "${MANIFEST_LOCAL}" \
    --endpoint-url https://s3.us-west-004.backblazeb2.com

# Extract checksum for the volume
CHECKSUM=$(cat "${MANIFEST_LOCAL}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for vol in data['volumes']:
    if vol['name'] == '${VOLUME_NAME}':
        print(vol['sha256'])
        break
")

if [ -z "$CHECKSUM" ]; then
    echo "ERROR: Could not find checksum for ${VOLUME_NAME} in manifest"
    exit 1
fi

echo "Found checksum: ${CHECKSUM}"
echo ""

# Build restore container
echo "=== Building restore container ==="
docker build -f /Users/root1/services/backup-b2/Dockerfile.restore \
    -t origin-restore /Users/root1/services/backup-b2

# Create restore output directory
RESTORE_OUTPUT="/Users/root1/restores/${TIMESTAMP}"
mkdir -p "${RESTORE_OUTPUT}"

# Run restore
echo ""
echo "=== Running restore ==="
echo "This will generate a RESTORE_ID. Use it for approval tokens."
echo ""

OBJECT_KEY="backups/${TIMESTAMP}/${VOLUME_NAME}.tar.gz"

docker run --rm \
    -e AWS_ACCESS_KEY_ID="$B2_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$B2_APP_KEY" \
    -e S3_ENDPOINT="https://s3.us-west-004.backblazeb2.com" \
    -e S3_BUCKET="origin-os-backups" \
    -v "${RESTORE_OUTPUT}:/restore" \
    -v "/Users/root1/restores/audit:/audit" \
    origin-restore \
    --bucket origin-os-backups \
    --key "${OBJECT_KEY}" \
    --checksum "${CHECKSUM}" \
    --environment "${ENVIRONMENT}" \
    --container "${VOLUME_NAME}" \
    --mode "${MODE}" \
    --requested-by "human:$(whoami)"

echo ""
echo "Restore output: ${RESTORE_OUTPUT}"
