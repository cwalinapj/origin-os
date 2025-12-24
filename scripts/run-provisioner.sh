#!/bin/bash
# =============================================================================
# RUN LLM AGENT PROVISIONER
# =============================================================================
# Creates S3 buckets and Pinecone namespaces for all 5 LLM agents.
# Outputs .env.agents and agents.json to ./output/
#
# Usage:
#   ./run-provisioner.sh
#
# Requires:
#   - AWS credentials in ~/.aws/credentials OR env vars
#   - PINECONE_API_KEY env var (optional, for vector DB setup)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/output"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  LLM AGENT PROVISIONER"
echo "=============================================="
echo ""

# Check for AWS credentials
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    if [ -f ~/.aws/credentials ]; then
        echo "Loading AWS credentials from ~/.aws/credentials"
        export AWS_ACCESS_KEY_ID=$(grep -A2 '\[default\]' ~/.aws/credentials | grep aws_access_key_id | cut -d= -f2 | tr -d ' ')
        export AWS_SECRET_ACCESS_KEY=$(grep -A2 '\[default\]' ~/.aws/credentials | grep aws_secret_access_key | cut -d= -f2 | tr -d ' ')
    else
        echo "ERROR: No AWS credentials found"
        echo "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or configure ~/.aws/credentials"
        exit 1
    fi
fi

export AWS_REGION="${AWS_REGION:-us-east-1}"

echo "AWS Region: $AWS_REGION"
echo "Output Dir: $OUTPUT_DIR"
echo ""

# Build the provisioner image
echo "Building provisioner container..."
docker build -t llm-provisioner -f "$PROJECT_ROOT/docker/Dockerfile.provisioner" "$PROJECT_ROOT"

# Run the provisioner
echo ""
echo "Running provisioner..."
docker run --rm \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -e AWS_REGION="$AWS_REGION" \
    -e PINECONE_API_KEY="${PINECONE_API_KEY:-}" \
    -e OUTPUT_DIR=/output \
    -v "$OUTPUT_DIR:/output" \
    llm-provisioner

echo ""
echo "=============================================="
echo "  PROVISIONING COMPLETE"
echo "=============================================="
echo ""
echo "Generated files:"
echo "  $OUTPUT_DIR/.env.agents"
echo "  $OUTPUT_DIR/agents.json"
echo ""
echo "To use with docker-compose:"
echo "  cat $OUTPUT_DIR/.env.agents >> .env"
echo ""
