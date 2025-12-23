#!/bin/bash
# =============================================================================
# ORIGIN OS WORM BACKUP — Runner Script
# =============================================================================
# This script:
# 1. Builds the backup container
# 2. Mounts all volumes READ-ONLY
# 3. Runs backup with Object Lock (WORM)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /Users/root1/.env

# Backblaze S3-compatible credentials (same as AWS format)
# Using the B2 application key in S3-compatible mode
export AWS_ACCESS_KEY_ID="${B2_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${B2_APP_KEY}"

S3_ENDPOINT="https://s3.us-west-004.backblazeb2.com"
S3_BUCKET="origin-os-backups"
RETENTION_DAYS=365

echo "============================================================"
echo "ORIGIN OS WORM BACKUP"
echo "============================================================"
echo ""

# Build backup container
echo "Building backup container..."
docker build -t origin-backup-worm "${SCRIPT_DIR}" >/dev/null 2>&1
echo "  ✓ Container built"
echo ""

# Create temp config directory
CONFIG_DIR=$(mktemp -d)
trap "rm -rf ${CONFIG_DIR}" EXIT

# Copy config files
cp /Users/root1/.env "${CONFIG_DIR}/env.txt" 2>/dev/null || true
cp /Users/root1/docker-compose.yml "${CONFIG_DIR}/" 2>/dev/null || true
cp /Users/root1/.cursor/mcp.json "${CONFIG_DIR}/cursor_mcp.json" 2>/dev/null || true
cp /Users/root1/.config/claude/memory.jsonl "${CONFIG_DIR}/" 2>/dev/null || true
cp "/Users/root1/Library/Application Support/Claude/claude_desktop_config.json" "${CONFIG_DIR}/" 2>/dev/null || true

# Create output directory
BACKUP_OUTPUT="/Users/root1/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_OUTPUT}"

echo "Running WORM backup..."
echo ""

# Run backup container with:
# - All volumes mounted READ-ONLY
# - No docker socket (secure)
# - No network except S3 endpoint
docker run --rm \
    --name origin-backup-runner \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e S3_ENDPOINT="${S3_ENDPOINT}" \
    -e S3_BUCKET="${S3_BUCKET}" \
    -e RETENTION_DAYS="${RETENTION_DAYS}" \
    -e OBJECT_LOCK_MODE="COMPLIANCE" \
    -e RUNNER="backup-container" \
    -e HOSTNAME="$(hostname)" \
    -v origin-vault-data:/volumes/vault:ro \
    -v origin-memory-data:/volumes/memory:ro \
    -v origin-registry-data:/volumes/registry:ro \
    -v origin-orchestrator-data:/volumes/orchestrator:ro \
    -v origin-cad-data:/volumes/cad:ro \
    -v origin-mcp-data:/volumes/mcp:ro \
    -v origin-backup-data:/volumes/backup:ro \
    -v "${CONFIG_DIR}:/configs:ro" \
    -v "${BACKUP_OUTPUT}:/output" \
    origin-backup-worm

echo ""
echo "Local backup copy: ${BACKUP_OUTPUT}"
echo ""

# Cleanup old local backups (keep 7 days)
find /Users/root1/backups -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
