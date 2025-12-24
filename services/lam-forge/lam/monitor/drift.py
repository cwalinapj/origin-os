#!/usr/bin/env python3
"""
DRIFT MONITOR — EMA-Smoothed Neural Drift Detection
====================================================

Prevents Alert Fatigue and flapping re-exploration events.

Raw drift signals are high-variance and can trigger false positives
from 15-minute traffic anomalies. By applying EMA smoothing (α=0.2),
we ensure FORCE_REEXPLORE only fires on sustained market shifts.

Why α=0.2:
- Inertia: ~10-15 high-drift events to cross 0.15 threshold
- Latency: Ignores "Friday Afternoon Spike" while reacting within 1 hour

Flow:
1. Monitor: Detects raw drift via embedding centroid distance
2. Smooth: Applies EMA (α=0.2)
3. Threshold: Smoothed Drift crosses 0.15
4. Action: Global Sync API sends FORCE_REEXPLORE
5. Bandit: Increases Thompson entropy across all containers
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass

import numpy as np
import redis.asyncio as redis

from lam.metrics import (
    global_neural_drift_magnitude,
    neural_drift_events,
    force_reexplore_triggers
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.2"))

logger = logging.getLogger("drift-monitor")


@dataclass
class DriftState:
    """Current state of drift detection."""
    raw_drift: float
    smoothed_drift: float
    samples_since_reexplore: int
    last_reexplore: Optional[datetime]
    trend_direction: str


class DriftMonitor:
    """
    EMA-smoothed neural drift monitor.
    
    Prevents alert fatigue by smoothing raw drift signals before
    exposing them to Prometheus and triggering FORCE_REEXPLORE.
    """
    
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.current_smoothed_drift = 0.0
        self.previous_smoothed_drift = 0.0
        self.samples_since_reexplore = 0
        self.last_reexplore: Optional[datetime] = None
        self.cluster_drifts: Dict[str, float] = {}
    
    def update_drift(self, raw_drift: float, cluster_id: str = "global") -> float:
        """
        Update drift with EMA smoothing.
        
        EMA Formula: S_t = α * Y_t + (1 - α) * S_{t-1}
        """
        self.previous_smoothed_drift = self.current_smoothed_drift
        
        # EMA smoothing
        self.current_smoothed_drift = (
            (self.alpha * raw_drift) + 
            ((1 - self.alpha) * self.current_smoothed_drift)
        )
        
        self.cluster_drifts[cluster_id] = self.current_smoothed_drift
        
        # Expose only smoothed value to Prometheus
        global_neural_drift_magnitude.labels(cluster_id=cluster_id).set(
            self.current_smoothed_drift
        )
        
        self.samples_since_reexplore += 1
        return self.current_smoothed_drift
    
    def get_trend_direction(self) -> str:
        """Determine if drift is rising, falling, or stable."""
        delta = self.current_smoothed_drift - self.previous_smoothed_drift
        if delta > 0.01:
            return "rising"
        elif delta < -0.01:
            return "falling"
        return "stable"
    
    def should_force_reexplore(self) -> bool:
        """Check if FORCE_REEXPLORE should be triggered."""
        if self.current_smoothed_drift <= DRIFT_THRESHOLD:
            return False
        if self.samples_since_reexplore < 10:
            return False
        if self.last_reexplore:
            elapsed = (datetime.now(timezone.utc) - self.last_reexplore).total_seconds()
            if elapsed < 1800:
                return False
        return True
    
    def trigger_reexplore(self, vertical: str = "global") -> DriftState:
        """Execute FORCE_REEXPLORE and reset counters."""
        state = self.get_state()
        
        force_reexplore_triggers.labels(vertical=vertical).inc()
        neural_drift_events.labels(vertical=vertical, severity="reexplore").inc()
        
        self.samples_since_reexplore = 0
        self.last_reexplore = datetime.now(timezone.utc)
        
        logger.warning(f"FORCE_REEXPLORE: smoothed={state.smoothed_drift:.4f}")
        return state
    
    def get_state(self) -> DriftState:
        """Get current drift state."""
        return DriftState(
            raw_drift=self.cluster_drifts.get("global", 0.0),
            smoothed_drift=self.current_smoothed_drift,
            samples_since_reexplore=self.samples_since_reexplore,
            last_reexplore=self.last_reexplore,
            trend_direction=self.get_trend_direction()
        )
