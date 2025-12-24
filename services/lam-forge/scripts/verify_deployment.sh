#!/bin/bash
# =============================================================================
# POST-DEPLOYMENT VERIFICATION — The "Zero-Hour" Log
# =============================================================================

set -e

NAMESPACE="ad-engine-prod"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[TEST]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }

echo ""
echo "=============================================="
echo "  POST-DEPLOYMENT VERIFICATION"
echo "  Zero-Hour Log"
echo "=============================================="
echo ""

ROUTER_POD=$(kubectl get pods -n $NAMESPACE -l app=bandit-router -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
FORGE_POD=$(kubectl get pods -n $NAMESPACE -l app=lam-forge -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
INFERENCE_POD=$(kubectl get pods -n $NAMESPACE -l app=lam-inference -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

TESTS_PASSED=0
TESTS_FAILED=0

# Test 1: Redis Connectivity
log_info "Testing Redis connectivity..."
if kubectl exec -n $NAMESPACE $ROUTER_POD -- sh -c "echo PING | nc redis-service 6379" 2>/dev/null | grep -q PONG; then
    log_success "Redis is reachable"
    ((TESTS_PASSED++))
else
    log_error "Redis connectivity failed"
    ((TESTS_FAILED++))
fi

# Test 2: MongoDB Connectivity
log_info "Testing MongoDB connectivity..."
if kubectl exec -n $NAMESPACE $ROUTER_POD -- sh -c "nc -z mongo-service 27017" 2>/dev/null; then
    log_success "MongoDB is reachable"
    ((TESTS_PASSED++))
else
    log_error "MongoDB connectivity failed"
    ((TESTS_FAILED++))
fi

# Test 3: ServiceMonitor Active
log_info "Checking ServiceMonitor..."
if kubectl get servicemonitor -n $NAMESPACE ad-engine-monitor &>/dev/null; then
    log_success "ServiceMonitor is active"
    ((TESTS_PASSED++))
else
    log_error "ServiceMonitor not found (Prometheus Operator may not be installed)"
    ((TESTS_FAILED++))
fi

# Test 4: GPU Access (if forge pod exists)
log_info "Checking GPU access..."
if [ -n "$FORGE_POD" ]; then
    if kubectl logs $FORGE_POD -n $NAMESPACE 2>/dev/null | grep -qi "cuda\|gpu"; then
        log_success "GPU/CUDA detected in Forge logs"
        ((TESTS_PASSED++))
    else
        log_error "No GPU/CUDA references in Forge logs"
        ((TESTS_FAILED++))
    fi
else
    log_info "LAM Forge pod not running (training job)"
    ((TESTS_PASSED++))
fi

# Test 5: Router Health
log_info "Checking Router health endpoint..."
if kubectl exec -n $NAMESPACE $ROUTER_POD -- curl -s localhost:8024/health 2>/dev/null | grep -q healthy; then
    log_success "Router health check passed"
    ((TESTS_PASSED++))
else
    log_error "Router health check failed"
    ((TESTS_FAILED++))
fi

# Test 6: Governor Health
log_info "Checking Governor health endpoint..."
GOVERNOR_POD=$(kubectl get pods -n $NAMESPACE -l app=governor -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if kubectl exec -n $NAMESPACE $GOVERNOR_POD -- curl -s localhost:8120/health 2>/dev/null | grep -q healthy; then
    log_success "Governor health check passed"
    ((TESTS_PASSED++))
else
    log_error "Governor health check failed"
    ((TESTS_FAILED++))
fi

# Test 7: HPA Active
log_info "Checking HPA status..."
if kubectl get hpa -n $NAMESPACE router-hpa &>/dev/null; then
    REPLICAS=$(kubectl get hpa -n $NAMESPACE router-hpa -o jsonpath='{.status.currentReplicas}')
    log_success "HPA active with $REPLICAS replicas"
    ((TESTS_PASSED++))
else
    log_error "HPA not found"
    ((TESTS_FAILED++))
fi

# Test 8: Network Policy Active
log_info "Checking Network Policies..."
NP_COUNT=$(kubectl get networkpolicy -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
if [ "$NP_COUNT" -gt 0 ]; then
    log_success "$NP_COUNT Network Policies active"
    ((TESTS_PASSED++))
else
    log_error "No Network Policies found"
    ((TESTS_FAILED++))
fi

echo ""
echo "=============================================="
echo "  VERIFICATION SUMMARY"
echo "=============================================="
echo ""
echo -e "  Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "  Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All verification tests passed!${NC}"
    echo ""
    echo "Ready to initialize Thompson priors:"
    echo "  ./scripts/init_marketing_loop.sh"
    exit 0
else
    echo -e "${RED}Some tests failed. Please check the logs.${NC}"
    exit 1
fi
