#!/bin/bash
# =============================================================================
# HIERARCHICAL SEED DEPLOYMENT — Phase A, B, C
# =============================================================================

set -e

NAMESPACE="ad-engine-prod"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$SCRIPT_DIR/../k8s"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "=============================================="
echo "  AUTONOMIC AD-ENGINE DEPLOYMENT"
echo "=============================================="
echo ""

# Pre-flight
if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster."
    exit 1
fi

echo "PHASE A: FOUNDATION"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$K8S_DIR/01-config.yaml" -n $NAMESPACE
kubectl apply -f "$K8S_DIR/02-storage.yaml" -n $NAMESPACE
kubectl rollout status statefulset/redis -n $NAMESPACE --timeout=120s
kubectl rollout status statefulset/mongo -n $NAMESPACE --timeout=180s
log_success "Phase A complete"

echo "PHASE B: INTELLIGENCE"
kubectl apply -f "$K8S_DIR/03-forge.yaml" -n $NAMESPACE
kubectl rollout status deployment/lam-inference -n $NAMESPACE --timeout=180s
kubectl rollout status deployment/governor -n $NAMESPACE --timeout=120s
log_success "Phase B complete"

echo "PHASE C: TRAFFIC GATEWAYS"
kubectl apply -f "$K8S_DIR/04-router.yaml" -n $NAMESPACE
kubectl apply -f "$K8S_DIR/06-network-policy.yaml" -n $NAMESPACE
kubectl apply -f "$K8S_DIR/05-gads-sync.yaml" -n $NAMESPACE
kubectl rollout status deployment/bandit-router -n $NAMESPACE --timeout=120s
log_success "Phase C complete"

echo "MONITORING"
kubectl apply -f "$K8S_DIR/07-monitoring.yaml" -n $NAMESPACE
kubectl apply -f "$K8S_DIR/08-service-monitors.yaml" -n $NAMESPACE 2>/dev/null || true

echo ""
log_success "Deployment complete!"
kubectl get pods -n $NAMESPACE
