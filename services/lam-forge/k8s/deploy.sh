#!/bin/bash
# =============================================================================
# DEPLOYMENT SCRIPT — The Final Push
# =============================================================================
# Deploys the Autonomic Ad-Engine to Kubernetes in order

set -e

NAMESPACE="ad-engine-prod"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "  AUTONOMIC AD-ENGINE DEPLOYMENT"
echo "=============================================="
echo ""

# Check kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl not found"
    exit 1
fi

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
    echo "ERROR: Cannot connect to Kubernetes cluster"
    exit 1
fi

echo "1. Creating Namespace & Config..."
kubectl apply -f "$SCRIPT_DIR/01-config.yaml"

echo "2. Deploying Persistence Layer (Redis, MongoDB)..."
kubectl apply -f "$SCRIPT_DIR/02-storage.yaml"

echo "   Waiting for StatefulSets to be ready..."
kubectl rollout status statefulset/redis -n $NAMESPACE --timeout=120s
kubectl rollout status statefulset/mongo -n $NAMESPACE --timeout=120s

echo "3. Deploying LAM Forge (GPU Training Engine)..."
kubectl apply -f "$SCRIPT_DIR/03-forge.yaml"

echo "   Waiting for LAM services to be ready..."
kubectl rollout status deployment/lam-inference -n $NAMESPACE --timeout=180s
kubectl rollout status deployment/governor -n $NAMESPACE --timeout=120s

echo "4. Deploying Router (Traffic Gatekeeper)..."
kubectl apply -f "$SCRIPT_DIR/04-router.yaml"

echo "   Waiting for Router to be ready..."
kubectl rollout status deployment/bandit-router -n $NAMESPACE --timeout=120s

echo "5. Deploying GAds Sync Service..."
kubectl apply -f "$SCRIPT_DIR/05-gads-sync.yaml"

echo "   Waiting for GAds Sync to be ready..."
kubectl rollout status deployment/gads-sync -n $NAMESPACE --timeout=120s

echo "6. Applying Network Policies..."
kubectl apply -f "$SCRIPT_DIR/06-network-policy.yaml"

echo "7. Deploying Monitoring (Prometheus, Grafana)..."
kubectl apply -f "$SCRIPT_DIR/07-monitoring.yaml"

echo ""
echo "=============================================="
echo "  DEPLOYMENT COMPLETE"
echo "=============================================="
echo ""

# Get service endpoints
echo "Service Endpoints:"
echo "  Router:     $(kubectl get svc router-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8024"
echo "  Grafana:    $(kubectl get svc grafana-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}'):3000"
echo "  Prometheus: $(kubectl get svc prometheus-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}'):9090"
echo ""

# Initialize Thompson priors
echo "Initializing Thompson Sampling priors..."
ROUTER_POD=$(kubectl get pods -n $NAMESPACE -l app=bandit-router -o jsonpath='{.items[0].metadata.name}')

if [ -n "$ROUTER_POD" ]; then
    echo "  Executing init_marketing_loop.sh in $ROUTER_POD..."
    kubectl exec -n $NAMESPACE "$ROUTER_POD" -- /bin/sh -c "if [ -f /app/init_marketing_loop.sh ]; then /app/init_marketing_loop.sh; else echo 'Init script not found'; fi"
fi

echo ""
echo "Deployment successful! The Autonomic Ad-Engine is now running."
echo ""
echo "Next steps:"
echo "  1. Configure Google Ads API credentials in engine-secrets"
echo "  2. Upload training data to training-data-pvc"
echo "  3. Monitor at Grafana: kubectl port-forward svc/grafana-service 3000:3000 -n $NAMESPACE"
