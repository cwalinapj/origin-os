#!/bin/bash
# =============================================================================
# FIRST-100 REGIME — Initialize and Run
# =============================================================================

set -e

echo "=============================================="
echo "  FIRST-100 REGIME STARTUP"
echo "=============================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose not found"
    exit 1
fi

# Use docker compose v2 or docker-compose v1
COMPOSE_CMD="docker compose"
if ! docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
fi

echo "1. Building containers..."
$COMPOSE_CMD -f docker-compose.first100.yml build

echo ""
echo "2. Starting infrastructure (Redis, MongoDB)..."
$COMPOSE_CMD -f docker-compose.first100.yml up -d redis mongo

echo "   Waiting for databases..."
sleep 5

echo ""
echo "3. Starting intelligence layer (LAM, Governor)..."
$COMPOSE_CMD -f docker-compose.first100.yml up -d lam-inference governor drift-monitor

echo "   Waiting for services..."
sleep 3

echo ""
echo "4. Starting router..."
$COMPOSE_CMD -f docker-compose.first100.yml up -d router

echo "   Waiting for router..."
sleep 2

echo ""
echo "5. Starting page variants..."
$COMPOSE_CMD -f docker-compose.first100.yml up -d page-a page-b

echo ""
echo "6. Starting monitoring..."
$COMPOSE_CMD -f docker-compose.first100.yml up -d prometheus grafana

echo ""
echo "7. Initializing Thompson Sampling priors..."
sleep 2

docker exec redis-state redis-cli <<EOF
# Arm: Baseline (Control)
HSET bandit:arms:site_001:baseline alpha 1 beta 1 samples 0

# Arm: Challenger (LLM-Generated)
HSET bandit:arms:site_001:challenger alpha 1 beta 1 samples 0

# Initialize LoA to Shadow
SET loa:level:default 1

# Initialize drift
SET drift:smoothed:global 0.0

# Circuit breaker
SET circuit:status CLOSED

ECHO "Thompson priors initialized"
EOF

echo ""
echo "=============================================="
echo "  FIRST-100 REGIME ACTIVE"
echo "=============================================="
echo ""
echo "Endpoints:"
echo "  Router:     http://localhost:8024"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo "  Prometheus: http://localhost:9090"
echo ""
echo "Test routing:"
echo "  curl -v http://localhost:8024/route/site_001"
echo ""
echo "View stats:"
echo "  curl http://localhost:8024/stats/site_001"
echo ""
echo "Monitor with:"
echo "  docker logs -f router-api"
echo ""
