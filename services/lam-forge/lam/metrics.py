#!/usr/bin/env python3
"""
PROMETHEUS METRICS — Low-Cardinality Observability
===================================================

CRITICAL: Cardinality Guardrail
- Durable Labels (High Value): vertical, site_id
- Transient Labels (EXCLUDED): page_id, session_id, visitor_id

High-cardinality details → MongoDB/Loki (for autopsy)
Low-cardinality aggregates → Prometheus (for heatmap)

Symmetric Surprise Buckets:
- (-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0)
- Centered around zero for directional analysis

Bucket Interpretation:
- Near-Zero [-0.1, 0.1]: Calibration Zone — high-fidelity LAM
- Extreme Negative [-1.0, -0.5]: Critical Failures → Inversion
- Extreme Positive [0.5, 1.0]: Exploration Jackpots → Knowledge Mesh
"""

import logging
from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY
)
from functools import wraps
import time
import asyncio

# =============================================================================
# LOGGING — High-cardinality details go here, NOT Prometheus
# =============================================================================

logger = logging.getLogger("lam.metrics")

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

SURPRISE_BUCKETS = (
    -1.0, -0.5, -0.25, -0.1,  # Under-performance (model too optimistic)
    0.0,                       # Perfect calibration
    0.1, 0.25, 0.5, 1.0        # Over-performance (model too pessimistic)
)

# =============================================================================
# LAM SURPRISE METRICS — Low Cardinality for Heatmap
# =============================================================================

# CORRECT: Only durable labels (vertical, site_id)
# page_id is EXCLUDED to prevent cardinality explosion
lam_surprise_delta = Histogram(
    'lam_surprise_delta',
    'Delta between expected and observed reward. '
    'Symmetric buckets enable directional analysis around zero.',
    labelnames=['vertical', 'site_id'],  # page_id EXCLUDED
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

# Zone counters (low cardinality)
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
# BANDIT METRICS — Low Cardinality
# =============================================================================

# Aggregate by site_id and outcome, NOT page_id
bandit_reward_total = Counter(
    'bandit_reward_total',
    'Total rewards by outcome type',
    labelnames=['site_id', 'outcome'],  # page_id EXCLUDED
    registry=registry
)

bandit_arm_samples = Gauge(
    'bandit_arm_samples',
    'Number of samples per arm',
    labelnames=['site_id', 'arm_index'],  # arm_index is bounded (0-3)
    registry=registry
)

bandit_arm_alpha = Gauge(
    'bandit_arm_alpha',
    'Thompson Sampling alpha parameter',
    labelnames=['site_id', 'arm_index'],
    registry=registry
)

bandit_arm_beta = Gauge(
    'bandit_arm_beta',
    'Thompson Sampling beta parameter',
    labelnames=['site_id', 'arm_index'],
    registry=registry
)

# =============================================================================
# NEURAL DRIFT METRICS
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
# ROUTER METRICS
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
# LOA METRICS
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
# SURPRISE RECORDING — THE FINAL IMPLEMENTATION
# =============================================================================

def record_surprise(sample: dict, pred_reward: float, actual_reward: float):
    """
    Record a surprise delta with proper cardinality management.
    
    1. Update the Heatmap (Prometheus) — LOW cardinality
    2. Log the high-cardinality "Autopsy" data (Loki/Mongo)
    
    Args:
        sample: The sample dict with meta.category, site_id, page_id
        pred_reward: What the LAM predicted
        actual_reward: What actually happened
    """
    vertical = sample.get("meta", {}).get("category", "unknown")
    site_id = sample.get("site_id", "unknown")
    page_id = sample.get("page_id", "unknown")  # For logging only
    
    surprise = actual_reward - pred_reward
    
    # 1. Update the Heatmap (Prometheus) — LOW CARDINALITY
    lam_surprise_delta.labels(
        vertical=vertical,
        site_id=site_id
        # page_id is EXCLUDED to prevent cardinality explosion
    ).observe(surprise)
    
    # Track zone counters
    if surprise <= -0.5:
        surprise_critical_failures.labels(
            vertical=vertical,
            site_id=site_id
        ).inc()
    elif surprise >= 0.5:
        surprise_exploration_jackpots.labels(
            vertical=vertical,
            site_id=site_id
        ).inc()
    elif -0.1 <= surprise <= 0.1:
        surprise_calibrated_total.labels(
            vertical=vertical,
            site_id=site_id
        ).inc()
    
    # 2. Log the high-cardinality "Autopsy" data (Loki/Mongo)
    # This is where page_id lives for debugging
    logger.info(
        "Surprise Event",
        extra={
            "page_id": page_id,  # HIGH cardinality — logs only
            "site_id": site_id,
            "vertical": vertical,
            "actual": actual_reward,
            "predicted": pred_reward,
            "diff": surprise
        }
    )


def record_reward(site_id: str, outcome: str):
    """Record a bandit reward event (low cardinality)."""
    bandit_reward_total.labels(site_id=site_id, outcome=outcome).inc()


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
