#!/usr/bin/env python3
"""
LAM FORGE — Prometheus Metrics
==============================

Exports metrics for monitoring:
- Surprise levels
- Inversion events
- Training progress
- Container stats
- Google Ads sync status
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# =============================================================================
# SURPRISE & INVERSION METRICS
# =============================================================================

SURPRISE_MAGNITUDE = Histogram(
    'lam_surprise_magnitude',
    'Distribution of surprise magnitudes (predicted - actual)',
    ['vertical', 'site_id'],
    buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
)

INVERSIONS_TOTAL = Counter(
    'lam_inversions_total',
    'Total inversion events triggered',
    ['vertical', 'error_type']  # error_type: Structural, Behavioral
)

PENALTY_WEIGHT = Gauge(
    'lam_penalty_weight',
    'Current penalty weight for training',
    ['vertical', 'site_id']
)

VERTICAL_MATURITY = Gauge(
    'lam_vertical_maturity',
    'Vertical maturity score (0-1)',
    ['vertical']
)

# =============================================================================
# TRAINING METRICS
# =============================================================================

TRAINING_STEPS = Counter(
    'lam_training_steps_total',
    'Total training steps completed',
    ['model_version']
)

TRAINING_LOSS = Gauge(
    'lam_training_loss',
    'Current training loss',
    ['loss_type']  # total, structural, inversion
)

SAMPLES_TOTAL = Counter(
    'lam_samples_total',
    'Total samples processed',
    ['site_id', 'vertical']
)

GHOST_MEMORY_SIZE = Gauge(
    'lam_ghost_memory_size',
    'Number of ghosts in memory',
    ['site_id']
)

# =============================================================================
# MODE & CONVERGENCE METRICS
# =============================================================================

MODE = Gauge(
    'lam_mode',
    'Current operating mode (0=exploration, 1=inference)',
    ['site_id']
)

CONVERGENCE_TIME = Histogram(
    'lam_convergence_time_seconds',
    'Time to convergence',
    ['vertical'],
    buckets=[3600, 7200, 14400, 28800, 43200, 86400, 172800, 259200]  # 1h to 3d
)

SAMPLES_TO_CONVERGENCE = Histogram(
    'lam_samples_to_convergence',
    'Samples needed for convergence',
    ['vertical'],
    buckets=[100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
)

# =============================================================================
# CONTAINER METRICS
# =============================================================================

ACTIVE_CONTAINERS = Gauge(
    'lam_active_containers',
    'Number of active page containers',
    ['site_id']
)

CONTAINER_SPAWN_TIME = Histogram(
    'lam_container_spawn_seconds',
    'Container spawn time',
    ['site_id'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

CONTAINER_CACHE_HITS = Counter(
    'lam_container_cache_hits_total',
    'Container cache hits',
    ['site_id']
)

CONTAINER_CACHE_MISSES = Counter(
    'lam_container_cache_misses_total',
    'Container cache misses (cold spawns)',
    ['site_id']
)

# =============================================================================
# GOOGLE ADS METRICS
# =============================================================================

GADS_PROMOTIONS = Counter(
    'gads_promotions_total',
    'Final URL promotions to Google Ads',
    ['site_id', 'status']  # success, error, skipped
)

GADS_ADJUSTMENTS = Counter(
    'gads_conversion_adjustments_total',
    'Behavioral score uploads',
    ['site_id', 'status']
)

GADS_SITELINKS = Counter(
    'gads_sitelink_deployments_total',
    'Sitelink mutations deployed',
    ['site_id', 'status']
)

GADS_BATCH_SIZE = Histogram(
    'gads_batch_size',
    'Batch size for conversion adjustments',
    ['site_id'],
    buckets=[10, 50, 100, 250, 500, 1000, 2000]
)

GADS_API_LATENCY = Histogram(
    'gads_api_latency_seconds',
    'Google Ads API latency',
    ['operation'],  # promote, sitelink, adjustment
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# =============================================================================
# GLOBAL DRIFT METRICS
# =============================================================================

GLOBAL_DRIFT_SCORE = Gauge(
    'lam_global_drift_score',
    'Global drift score across all sites'
)

DRIFT_EVENTS = Counter(
    'lam_drift_events_total',
    'Global drift events detected',
    ['severity']  # low, medium, high
)

ENTROPY_INJECTIONS = Counter(
    'lam_entropy_injections_total',
    'Global entropy injection events',
    ['vertical']
)

# =============================================================================
# INFO METRICS
# =============================================================================

LAM_INFO = Info(
    'lam_forge',
    'LAM Forge service information'
)

# Set version info
LAM_INFO.info({
    'version': '1.0.0',
    'model_architecture': 'triplet_inversion',
    'embedding_dim': '256'
})
