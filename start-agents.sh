#!/bin/bash
# =============================================================================
# START FIRST-1000 REGIME — Multi-Agent Stack
# =============================================================================
# Launches all 5 LLM agents with their unique infrastructure
#
# Usage:
#   ./start-agents.sh
#   ./start-agents.sh --provision  # Run provisioner first
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "=============================================="
echo "  FIRST-1000 REGIME — Multi-Agent Stack"
echo "=============================================="
echo ""

# Check for --provision flag
if [ "$1" == "--provision" ]; then
    echo "Running provisioner first..."
    ./scripts/run-provisioner.sh
    echo ""
fi

# Check for .env.agents
ENV_FILE="$PROJECT_ROOT/output/.env.agents"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found"
    echo ""
    echo "Run the provisioner first:"
    echo "  ./scripts/run-provisioner.sh"
    echo ""
    echo "Or run this script with --provision:"
    echo "  ./start-agents.sh --provision"
    exit 1
fi

# Check for global env vars
if [ -z "$PINECONE_API_KEY" ] && [ -z "$VERCEL_API_KEY" ]; then
    echo "WARNING: Global API keys not set in environment"
    echo "Set these before running:"
    echo "  export PINECONE_API_KEY=your-key"
    echo "  export VERCEL_API_KEY=your-key"
    echo "  export OPENROUTER_API_KEY=your-key"
    echo ""
fi

echo "Using env file: $ENV_FILE"
echo ""

# Merge env files
COMBINED_ENV="$PROJECT_ROOT/.env.combined"
cat "$ENV_FILE" > "$COMBINED_ENV"

# Add global keys if set
[ -n "$PINECONE_API_KEY" ] && echo "PINECONE_API_KEY=$PINECONE_API_KEY" >> "$COMBINED_ENV"
[ -n "$VERCEL_API_KEY" ] && echo "VERCEL_API_KEY=$VERCEL_API_KEY" >> "$COMBINED_ENV"
[ -n "$OPENROUTER_API_KEY" ] && echo "OPENROUTER_API_KEY=$OPENROUTER_API_KEY" >> "$COMBINED_ENV"
[ -n "$GA_API_KEY" ] && echo "GA_API_KEY=$GA_API_KEY" >> "$COMBINED_ENV"
[ -n "$ANTHROPIC_API_KEY" ] && echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> "$COMBINED_ENV"
[ -n "$OPENAI_API_KEY" ] && echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> "$COMBINED_ENV"
[ -n "$GOOGLE_API_KEY" ] && echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" >> "$COMBINED_ENV"
[ -n "$MISTRAL_API_KEY" ] && echo "MISTRAL_API_KEY=$MISTRAL_API_KEY" >> "$COMBINED_ENV"

echo "Starting infrastructure..."
docker compose --env-file "$COMBINED_ENV" -f docker-compose.agents.yml up -d redis mongo llm-gateway

echo "Waiting for infrastructure..."
sleep 5

echo "Starting agents..."
docker compose --env-file "$COMBINED_ENV" -f docker-compose.agents.yml up -d \
    claude-agent \
    gpt4-agent \
    gemini-agent \
    llama-agent \
    mistral-agent

echo "Waiting for agents..."
sleep 3

echo "Starting router..."
docker compose --env-file "$COMBINED_ENV" -f docker-compose.agents.yml up -d router

echo "Starting monitoring..."
docker compose --env-file "$COMBINED_ENV" -f docker-compose.agents.yml up -d prometheus grafana

# Cleanup temp env
rm -f "$COMBINED_ENV"

echo ""
echo "=============================================="
echo "  MULTI-AGENT STACK RUNNING"
echo "=============================================="
echo ""
echo "Agents:"
docker ps --filter "name=lam-" --format "  {{.Names}}: {{.Status}}"
echo ""
echo "Endpoints:"
echo "  Router:     http://localhost:8024"
echo "  LLM Gateway: http://localhost:8200"
echo "  Grafana:    http://localhost:3000"
echo "  Prometheus: http://localhost:9090"
echo ""
echo "Verify agent env:"
echo "  docker exec lam-claude printenv | grep S3"
echo "  docker exec lam-gpt4 printenv | grep GTAG"
echo ""
