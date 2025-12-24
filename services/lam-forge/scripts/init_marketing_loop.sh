#!/bin/bash
# =============================================================================
# INIT MARKETING LOOP — Seed Thompson Sampling Priors
# =============================================================================
# Seeds the 4 initial LLM arms into Redis for Thompson Sampling.
# Run this after deployment verification passes.

set -e

NAMESPACE="ad-engine-prod"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

echo ""
echo "=============================================="
echo "  INIT MARKETING LOOP"
echo "  Seeding Thompson Sampling Priors"
echo "=============================================="
echo ""

ROUTER_POD=$(kubectl get pods -n $NAMESPACE -l app=bandit-router -o jsonpath='{.items[0].metadata.name}')

log_info "Target pod: $ROUTER_POD"

# Seed the 4 initial LLM arms with neutral priors (α=1, β=1)
log_info "Seeding initial Thompson Sampling arms..."

kubectl exec -n $NAMESPACE $ROUTER_POD -- sh -c '
redis-cli -h redis-service <<EOF
# Arm 0: Claude/Anthropic
HSET bandit:arms:default:0 alpha 1 beta 1 model "claude-sonnet" strategy "trust_focused"
# Arm 1: GPT-4
HSET bandit:arms:default:1 alpha 1 beta 1 model "gpt-4" strategy "balanced"
# Arm 2: Gemini
HSET bandit:arms:default:2 alpha 1 beta 1 model "gemini-pro" strategy "creative"
# Arm 3: Baseline (no mutation)
HSET bandit:arms:default:3 alpha 1 beta 1 model "baseline" strategy "control"

# Initialize sample counts
SET bandit:samples:default:0 0
SET bandit:samples:default:1 0
SET bandit:samples:default:2 0
SET bandit:samples:default:3 0

# Set initial LoA to Shadow (Level 1)
SET loa:level:ecommerce 1
SET loa:level:b2b 1
SET loa:level:saas 1

# Initialize accuracy to 0 (will be updated after First-100)
SET loa_accuracy:ecommerce 0.0
SET loa_accuracy:b2b 0.0
SET loa_accuracy:saas 0.0

# Initialize sample counts
SET loa_samples:ecommerce 0
SET loa_samples:b2b 0
SET loa_samples:saas 0

# Initialize drift EMA
SET drift:smoothed:global 0.0
SET drift:samples:global 0

# Set circuit breaker to CLOSED
SET circuit:status CLOSED
SET circuit:failures 0

ECHO "Thompson priors seeded successfully"
EOF
'

log_success "Thompson Sampling arms initialized:"
echo "  - Arm 0: Claude (trust_focused)"
echo "  - Arm 1: GPT-4 (balanced)"
echo "  - Arm 2: Gemini (creative)"
echo "  - Arm 3: Baseline (control)"
echo ""

log_success "LoA initialized to Shadow (Level 1) for all verticals"
log_success "Circuit Breaker set to CLOSED"
log_success "Drift EMA initialized to 0.0"

echo ""
echo "=============================================="
echo "  MARKETING LOOP INITIALIZED"
echo "=============================================="
echo ""
echo "The system is now ready to receive traffic."
echo ""
echo "Monitor the First-100 samples at:"
echo "  kubectl port-forward svc/grafana-service 3000:3000 -n $NAMESPACE"
echo ""
echo "Expected healthy signals:"
echo "  1. Thompson Entropy: High (~25% each arm)"
echo "  2. Surprise Heatmap: Clustering near 0.0"
echo "  3. Circuit Breaker: CLOSED (0 failures)"
echo "  4. LoA: Shadow (Level 1)"
echo ""
