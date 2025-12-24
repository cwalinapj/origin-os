#!/usr/bin/env python3
"""
PROMETHEUS METRICS — Instrumented Observability
===============================================

Exposes metrics for the Surprise Heatmap and Global Drift detection.

CRITICAL: Surprise buckets are SYMMETRIC around zero to distinguish:
- Under-performance Surprise (negative): Model was too optimistic
- Over-performance Surprise (positive): Model was too pessimistic

Bucket interpretation:
- Near-Zero [-0.1, 0.1]: Calibration Zone — LAM is high-fidelity
- Extreme Negative [-1.0, -0.5]: Critical Failures — trigger Inversion
- Extreme Positive [0.5, 1.0]: Exploration Jackpots — Knowledge Mesh candidates
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY
)
from functools import wraps
import time
import asyncio

# =============================================================================
# REGISTRY
# =============================================================================

try:
    from prometheus_client import multiprocess
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
except:
    registry = REGISTRY

# =============================================================================
# SYMMETRIC SURPRISE BUCKETS
# =============================================================================
# These buckets center around 0.0 to enable directional analysis:
# - Negative = Model overestimated (Critical Failures)
# - Positive = Model underestimated (Exploration Jackpots)

SURPRISE_BUCKETS = (
    -1.0, -0.5, -0.25, -0.1,  # Under-performance (model too optimistic)
    0.0,                       # Perfect calibration
    0.1, 0.25, 0.5, 1.0        # Over-performance (model too pessimistic)
)

# =============================================================================
# LAM SURPRISE METRICS — Feeds the Surprise Heatmap
# =============================================================================

lam_surprise_delta = Histogram(
    'lam_surprise_delta',
    'Distribution of surprise deltas (actual - predicted reward). '
    'Symmetric buckets enable directional analysis around zero.',
    labelnames=['vertical', 'site_id'],
    buckets=SURPRISE_BUCKETS,
    registry=registry
)

lam_offline_accuracy = Gauge(
    'lam_offline_accuracy',
    'Offline replay accuracy by vertical',
    labelnames=['vertical'],
    registry=registry
)

lam_calibration_error = Gauge(
    'lam_calibration_error',
    'Mean absolute calibration error',
    labelnames=['vertical'],
    registry=registry
)

# Surprise zone counters for alerting
surprise_critical_failures = Counter(
    'surprise_critical_failures_total',
    'Observations in extreme negative buckets [-1.0, -0.5]',
    labelnames=['vertical', 'site_id'],
    registry=registry
)

surprise_exploration_jackpots = Counter(
    'surprise_exploration_jackpots_total',
    'Observations in extreme positive buckets [0.5, 1.0]',
    labelnames=['vertical', 'site_id'],
    registry=registry
)

surprise_calibrated_total = Counter(
    'surprise_calibrated_total',
    'Observations in calibration zone [-0.1, 0.1]',
    labelnames=['vertical', 'site_id'],
    registry=registry
)

# =============================================================================
# BANDIT METRICS — Thompson Sampling Performance
# =============================================================================

bandit_reward_total = Counter(
    'bandit_reward_total',
    'Total rewards by outcome type',
    labelnames=['page_id', 'outcome'],
    registry=registry
)

bandit_arm_samples = Gauge(
    'bandit_arm_samples',
    'Number of samples per arm',
    labelnames=['site_id', 'arm_id'],
    registry=registry
)

bandit_arm_alpha = Gauge(
    'bandit_arm_alpha',
    'Thompson Sampling alpha parameter',
    labelnames=['site_id', 'arm_id'],
    registry=registry
)

bandit_arm_beta = Gauge(
    'bandit_arm_beta',
    'Thompson Sampling beta parameter',
    labelnames=['site_id', 'arm_id'],
    registry=registry
)

# =============================================================================
# NEURAL DRIFT METRICS — Global Re-Exploration Triggers
# =============================================================================

global_neural_drift_magnitude = Gauge(
    'global_neural_drift_magnitude',
    'Magnitude of centroid shift in latent space',
    labelnames=['cluster_id'],
    registry=registry
)

neural_drift_events = Counter(
    'neural_drift_events_total',
    'Total neural drift events detected',
    labelnames=['vertical', 'severity'],
    registry=registry
)

force_reexplore_triggers = Counter(
    'force_reexplore_total',
    'Total FORCE_REEXPLORE events triggered',
    labelnames=['vertical'],
    registry=registry
)

# =============================================================================
# ROUTER METRICS — Circuit Breaker Monitoring
# =============================================================================

router_decision_latency_seconds = Histogram(
    'router_decision_latency_seconds',
    'Router decision latency by LoA level',
    labelnames=['loa_level'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5, 1.0),
    registry=registry
)

router_requests_total = Counter(
    'router_requests_total',
    'Total router requests',
    labelnames=['vertical', 'loa_level', 'status'],
    registry=registry
)

circuit_breaker_trips = Counter(
    'circuit_breaker_trips_total',
    'Total circuit breaker trip events',
    labelnames=['vertical', 'reason'],
    registry=registry
)

# =============================================================================
# LOA METRICS — Level of Autonomy Tracking
# =============================================================================

loa_current_level = Gauge(
    'loa_current_level',
    'Current Level of Autonomy by vertical',
    labelnames=['vertical'],
    registry=registry
)

loa_samples_total = Gauge(
    'loa_samples_total',
    'Total samples accumulated for LoA calculation',
    labelnames=['vertical'],
    registry=registry
)

loa_downgrade_total = Counter(
    'loa_downgrade_total',
    'Total LoA downgrade events',
    labelnames=['vertical', 'reason'],
    registry=registry
)

loa_upgrade_total = Counter(
    'loa_upgrade_total',
    'Total LoA upgrade events',
    labelnames=['vertical', 'from_level', 'to_level'],
    registry=registry
)

# =============================================================================
# REWARD SHAPING METRICS
# =============================================================================

reward_shaped_value = Histogram(
    'reward_shaped_value',
    'Distribution of shaped reward values',
    labelnames=['vertical', 'stage'],
    buckets=(-1.0, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 1.0),
    registry=registry
)

reward_component_value = Gauge(
    'reward_component_value',
    'Individual reward component values',
    labelnames=['vertical', 'component'],
    registry=registry
)

dark_pattern_detections = Counter(
    'dark_pattern_detections_total',
    'Total dark pattern detections',
    labelnames=['vertical', 'pattern_type'],
    registry=registry
)

# =============================================================================
# TRAINING METRICS
# =============================================================================

training_loss = Gauge(
    'training_loss',
    'Current training loss',
    labelnames=['vertical', 'stage'],
    registry=registry
)

training_epoch = Gauge(
    'training_epoch',
    'Current training epoch',
    labelnames=['stage'],
    registry=registry
)

curriculum_stage = Gauge(
    'curriculum_stage_current',
    'Current curriculum stage',
    labelnames=['vertical'],
    registry=registry
)

# =============================================================================
# CONTAINER METRICS
# =============================================================================

container_spawns_total = Counter(
    'container_spawns_total',
    'Total container spawn events',
    labelnames=['vertical', 'generation_strategy'],
    registry=registry
)

container_tombstones_total = Counter(
    'container_tombstones_total',
    'Total tombstone events',
    labelnames=['vertical', 'reason'],
    registry=registry
)

container_lifetime_seconds = Histogram(
    'container_lifetime_seconds',
    'Container lifetime distribution',
    labelnames=['vertical'],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400),
    registry=registry
)

# =============================================================================
# SURPRISE RECORDING FUNCTION
# =============================================================================

def record_surprise(
    vertical: str,
    site_id: str,
    predicted_reward: float,
    actual_reward: float
):
    """
    Record a surprise delta for the Heatmap.
    
    CRITICAL: surprise = actual - predicted (signed, directional)
    
    - Negative surprise: Model was too optimistic (Critical Failure)
    - Positive surprise: Model was too pessimistic (Exploration Jackpot)
    - Near-zero: Model is well-calibrated
    
    Args:
        vertical: Site vertical (ecommerce, b2b, saas)
        site_id: Unique site identifier
        predicted_reward: What the LAM predicted
        actual_reward: What actually happened
    """
    surprise = actual_reward - predicted_reward
    
    # Record to histogram
    lam_surprise_delta.labels(
        vertical=vertical,
        site_id=site_id
    ).observe(surprise)
    
    # Track zone counters for alerting
    if surprise <= -0.5:
        # Critical Failure: Model was WAY too optimistic
        surprise_critical_failures.labels(
            vertical=vertical,
            site_id=site_id
        ).inc()
    elif surprise >= 0.5:
        # Exploration Jackpot: Model was WAY too pessimistic
        surprise_exploration_jackpots.labels(
            vertical=vertical,
            site_id=site_id
        ).inc()
    elif -0.1 <= surprise <= 0.1:
        # Calibration Zone: Model is accurate
        surprise_calibrated_total.labels(
            vertical=vertical,
            site_id=site_id
        ).inc()


def record_reward(page_id: str, outcome: str):
    """Record a bandit reward event."""
    bandit_reward_total.labels(page_id=page_id, outcome=outcome).inc()


def record_drift(cluster_id: str, magnitude: float):
    """Record neural drift magnitude."""
    global_neural_drift_magnitude.labels(cluster_id=cluster_id).set(magnitude)


def record_loa(vertical: str, level: int, samples: int, accuracy: float):
    """Record LoA metrics."""
    loa_current_level.labels(vertical=vertical).set(level)
    loa_samples_total.labels(vertical=vertical).set(samples)
    lam_offline_accuracy.labels(vertical=vertical).set(accuracy)


# =============================================================================
# HELPER DECORATORS
# =============================================================================

def track_latency(metric, labels_fn=None):
    """Decorator to track function latency."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            labels = labels_fn(*args, **kwargs) if labels_fn else {}
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metric.labels(**labels).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            labels = labels_fn(*args, **kwargs) if labels_fn else {}
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metric.labels(**labels).observe(duration)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# =============================================================================
# FASTAPI INTEGRATION
# =============================================================================

from fastapi import Response


async def metrics_endpoint():
    """FastAPI endpoint for /metrics."""
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


def setup_metrics_route(app):
    """Add /metrics route to FastAPI app."""
    @app.get("/metrics")
    async def metrics():
        return await metrics_endpoint()
