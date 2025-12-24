#!/usr/bin/env python3
"""
NEURAL MONITOR — Ensemble Drift Detection
==========================================

Detects behavioral shifts across the entire system using:
1. Macro-Variance Tracking: Aggregate σ spikes across all arms
2. Embedding Centroid Drift: Movement in converting user latent space
3. Decision Latency Shifts: Changes in time-to-action patterns

When drift is detected, triggers:
- Global Entropy Injection (inflate Thompson σ)
- Force Re-exploration for affected verticals
- Knowledge Mesh synchronization
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DRIFT_CHECK_INTERVAL = int(os.getenv("DRIFT_CHECK_INTERVAL", "300"))  # 5 minutes
CENTROID_WINDOW_DAYS = int(os.getenv("CENTROID_WINDOW_DAYS", "30"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
VARIANCE_SPIKE_THRESHOLD = float(os.getenv("VARIANCE_SPIKE_THRESHOLD", "2.0"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural-monitor")

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class DriftEvent:
    """Represents a detected drift event."""
    event_id: str
    timestamp: str
    drift_type: str  # "variance", "centroid", "latency", "composite"
    severity: str    # "low", "medium", "high", "critical"
    global_drift_score: float
    detected_shifts: List[Dict[str, Any]]
    affected_verticals: List[str]
    recommended_action: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "drift_type": self.drift_type,
            "severity": self.severity,
            "global_drift_score": self.global_drift_score,
            "detected_shifts": self.detected_shifts,
            "affected_verticals": self.affected_verticals,
            "recommended_action": self.recommended_action
        }


@dataclass
class SessionNeuralState:
    """Neural state embedding for a user session."""
    session_id: str
    site_id: str
    vertical: str
    embedding: np.ndarray
    converted: bool
    timestamp: str
    behavioral_score: float
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "site_id": self.site_id,
            "vertical": self.vertical,
            "embedding": self.embedding.tolist(),
            "converted": self.converted,
            "timestamp": self.timestamp,
            "behavioral_score": self.behavioral_score
        }


@dataclass 
class VerticalStats:
    """Aggregate statistics for a vertical."""
    vertical: str
    mean_reward: float
    std_reward: float
    sample_count: int
    converting_centroid: np.ndarray
    non_converting_centroid: np.ndarray
    decision_latency_p50: float
    decision_latency_p95: float
    last_updated: str


# =============================================================================
# MACRO-VARIANCE TRACKER
# =============================================================================

class MacroVarianceTracker:
    """
    Monitors aggregate standard deviation of reward means across all active arms.
    
    If σ across all sites spikes simultaneously without a change in traffic volume,
    the environment has shifted (e.g., competitor launched new campaign, 
    seasonal behavior change, news event affecting purchasing).
    """
    
    def __init__(self, redis_client: redis.Redis, window_size: int = 100):
        self.redis = redis_client
        self.window_size = window_size
        self.variance_history: deque = deque(maxlen=window_size)
        self.baseline_variance: Optional[float] = None
    
    async def compute_global_variance(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute variance of reward means across all active sites.
        
        Returns:
            (global_variance, per_vertical_variance)
        """
        # Get all site stats
        sites = await self._get_all_site_stats()
        
        if not sites:
            return 0.0, {}
        
        # Group by vertical
        vertical_rewards: Dict[str, List[float]] = {}
        all_rewards = []
        
        for site in sites:
            vertical = site.get("vertical", "unknown")
            mean_reward = site.get("mean_reward", 0.0)
            
            if vertical not in vertical_rewards:
                vertical_rewards[vertical] = []
            vertical_rewards[vertical].append(mean_reward)
            all_rewards.append(mean_reward)
        
        # Compute global variance
        global_variance = np.var(all_rewards) if all_rewards else 0.0
        
        # Compute per-vertical variance
        per_vertical = {
            v: np.var(rewards) for v, rewards in vertical_rewards.items()
        }
        
        return global_variance, per_vertical
    
    async def check_variance_spike(self) -> Optional[Dict[str, Any]]:
        """
        Check if current variance represents a significant spike.
        
        Returns drift info if spike detected, None otherwise.
        """
        current_variance, per_vertical = await self.compute_global_variance()
        
        # Update history
        self.variance_history.append(current_variance)
        
        # Need enough history for baseline
        if len(self.variance_history) < 10:
            return None
        
        # Compute baseline (excluding current)
        history_array = np.array(list(self.variance_history)[:-1])
        baseline_mean = np.mean(history_array)
        baseline_std = np.std(history_array) + 1e-8
        
        # Z-score of current variance
        z_score = (current_variance - baseline_mean) / baseline_std
        
        if z_score > VARIANCE_SPIKE_THRESHOLD:
            # Identify which verticals are spiking
            spiking_verticals = []
            for vertical, var in per_vertical.items():
                vertical_history = await self._get_vertical_variance_history(vertical)
                if vertical_history:
                    v_baseline = np.mean(vertical_history)
                    v_std = np.std(vertical_history) + 1e-8
                    v_zscore = (var - v_baseline) / v_std
                    if v_zscore > VARIANCE_SPIKE_THRESHOLD:
                        spiking_verticals.append({
                            "vertical": vertical,
                            "z_score": v_zscore,
                            "current_variance": var
                        })
            
            return {
                "type": "variance_spike",
                "global_z_score": z_score,
                "current_variance": current_variance,
                "baseline_mean": baseline_mean,
                "spiking_verticals": spiking_verticals
            }
        
        return None
    
    async def _get_all_site_stats(self) -> List[Dict]:
        """Get stats for all active sites."""
        sites = []
        cursor = 0
        
        while True:
            cursor, keys = await self.redis.scan(
                cursor, match="site_stats:*", count=100
            )
            
            for key in keys:
                data = await self.redis.hgetall(key)
                if data:
                    sites.append({
                        k.decode(): float(v) if k != b"vertical" else v.decode()
                        for k, v in data.items()
                    })
            
            if cursor == 0:
                break
        
        return sites
    
    async def _get_vertical_variance_history(self, vertical: str) -> List[float]:
        """Get historical variance for a vertical."""
        history = await self.redis.lrange(f"variance_history:{vertical}", 0, -1)
        return [float(v) for v in history] if history else []


# =============================================================================
# EMBEDDING CENTROID DRIFT DETECTOR
# =============================================================================

class EmbeddingCentroidDrift:
    """
    Detects drift in the latent space of converting users.
    
    Daily background task clusters SessionNeuralState embeddings.
    If the centroid of "Converting Users" moves significantly compared
    to the 30-day moving average, we flag a behavioral shift.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.centroid_history: Dict[str, deque] = {}  # vertical -> history
    
    async def compute_centroids(self, vertical: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute centroids for converting and non-converting users.
        
        Returns:
            (converting_centroid, non_converting_centroid)
        """
        # Get recent session embeddings
        sessions = await self._get_recent_sessions(vertical, days=7)
        
        if not sessions:
            return np.zeros(EMBEDDING_DIM), np.zeros(EMBEDDING_DIM)
        
        converting = [s.embedding for s in sessions if s.converted]
        non_converting = [s.embedding for s in sessions if not s.converted]
        
        converting_centroid = np.mean(converting, axis=0) if converting else np.zeros(EMBEDDING_DIM)
        non_converting_centroid = np.mean(non_converting, axis=0) if non_converting else np.zeros(EMBEDDING_DIM)
        
        return converting_centroid, non_converting_centroid
    
    async def check_centroid_drift(self, vertical: str) -> Optional[Dict[str, Any]]:
        """
        Check if the converting user centroid has drifted.
        
        Returns drift info if significant drift detected.
        """
        current_conv, current_non = await self.compute_centroids(vertical)
        
        # Get historical centroid
        historical = await self._get_historical_centroid(vertical)
        
        if historical is None:
            # Store current as baseline
            await self._store_centroid(vertical, current_conv)
            return None
        
        # Compute cosine similarity
        if np.linalg.norm(current_conv) == 0 or np.linalg.norm(historical) == 0:
            return None
        
        similarity = cosine_similarity(
            current_conv.reshape(1, -1),
            historical.reshape(1, -1)
        )[0, 0]
        
        drift_score = 1 - similarity
        
        # Store current centroid
        await self._store_centroid(vertical, current_conv)
        
        if drift_score > DRIFT_THRESHOLD:
            # Analyze which dimensions shifted most
            diff = current_conv - historical
            top_dims = np.argsort(np.abs(diff))[-10:]  # Top 10 changed dimensions
            
            return {
                "type": "centroid_drift",
                "vertical": vertical,
                "drift_score": drift_score,
                "similarity": similarity,
                "top_changed_dimensions": top_dims.tolist(),
                "dimension_deltas": diff[top_dims].tolist()
            }
        
        return None
    
    async def _get_recent_sessions(self, vertical: str, days: int = 7) -> List[SessionNeuralState]:
        """Get recent session states for a vertical."""
        sessions = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Scan session embeddings
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor, match=f"session_state:{vertical}:*", count=100
            )
            
            for key in keys:
                data = await self.redis.hgetall(key)
                if data:
                    timestamp = data.get(b"timestamp", b"").decode()
                    if timestamp and datetime.fromisoformat(timestamp.replace('Z', '+00:00')) > cutoff:
                        embedding_str = data.get(b"embedding", b"[]").decode()
                        sessions.append(SessionNeuralState(
                            session_id=data.get(b"session_id", b"").decode(),
                            site_id=data.get(b"site_id", b"").decode(),
                            vertical=vertical,
                            embedding=np.array(json.loads(embedding_str)),
                            converted=data.get(b"converted", b"0") == b"1",
                            timestamp=timestamp,
                            behavioral_score=float(data.get(b"behavioral_score", 0))
                        ))
            
            if cursor == 0:
                break
        
        return sessions
    
    async def _get_historical_centroid(self, vertical: str) -> Optional[np.ndarray]:
        """Get the 30-day moving average centroid."""
        history = await self.redis.lrange(f"centroid_history:{vertical}", 0, CENTROID_WINDOW_DAYS - 1)
        
        if not history:
            return None
        
        centroids = [np.array(json.loads(h)) for h in history]
        return np.mean(centroids, axis=0)
    
    async def _store_centroid(self, vertical: str, centroid: np.ndarray):
        """Store centroid in history."""
        await self.redis.lpush(
            f"centroid_history:{vertical}",
            json.dumps(centroid.tolist())
        )
        await self.redis.ltrim(f"centroid_history:{vertical}", 0, CENTROID_WINDOW_DAYS - 1)


# =============================================================================
# DECISION LATENCY TRACKER
# =============================================================================

class DecisionLatencyTracker:
    """
    Tracks changes in user decision-making speed.
    
    If users are making decisions faster/slower than historical baseline,
    it indicates a shift in intent or context (e.g., urgency, confusion).
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def record_decision_latency(
        self,
        site_id: str,
        session_id: str,
        latency_ms: float,
        decision_type: str  # "cta_click", "scroll_depth", "form_submit"
    ):
        """Record a decision latency event."""
        await self.redis.xadd(
            f"decision_latency:{site_id}",
            {
                "session_id": session_id,
                "latency_ms": latency_ms,
                "decision_type": decision_type,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            maxlen=10000
        )
    
    async def check_latency_shift(self, vertical: str) -> Optional[Dict[str, Any]]:
        """
        Check for significant shifts in decision latency.
        """
        # Get recent latencies
        recent = await self._get_recent_latencies(vertical, hours=24)
        historical = await self._get_historical_latencies(vertical, days=7)
        
        if len(recent) < 100 or len(historical) < 100:
            return None
        
        recent_p50 = np.percentile(recent, 50)
        recent_p95 = np.percentile(recent, 95)
        hist_p50 = np.percentile(historical, 50)
        hist_p95 = np.percentile(historical, 95)
        
        # Check for significant shift
        p50_shift = (recent_p50 - hist_p50) / (hist_p50 + 1e-8)
        p95_shift = (recent_p95 - hist_p95) / (hist_p95 + 1e-8)
        
        if abs(p50_shift) > 0.2 or abs(p95_shift) > 0.3:
            direction = "shortening" if p50_shift < 0 else "lengthening"
            return {
                "type": "latency_shift",
                "vertical": vertical,
                "direction": direction,
                "p50_shift_pct": p50_shift * 100,
                "p95_shift_pct": p95_shift * 100,
                "recent_p50": recent_p50,
                "recent_p95": recent_p95,
                "historical_p50": hist_p50,
                "historical_p95": hist_p95
            }
        
        return None
    
    async def _get_recent_latencies(self, vertical: str, hours: int = 24) -> List[float]:
        """Get recent latencies for vertical."""
        latencies = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Get sites in vertical
        sites = await self._get_sites_in_vertical(vertical)
        
        for site_id in sites:
            entries = await self.redis.xrange(
                f"decision_latency:{site_id}",
                min="-",
                max="+",
                count=1000
            )
            for _, data in entries:
                timestamp = data.get(b"timestamp", b"").decode()
                if timestamp and datetime.fromisoformat(timestamp.replace('Z', '+00:00')) > cutoff:
                    latencies.append(float(data.get(b"latency_ms", 0)))
        
        return latencies
    
    async def _get_historical_latencies(self, vertical: str, days: int = 7) -> List[float]:
        """Get historical latencies for vertical."""
        # Similar to recent but for longer window
        return await self._get_recent_latencies(vertical, hours=days * 24)
    
    async def _get_sites_in_vertical(self, vertical: str) -> List[str]:
        """Get all site IDs in a vertical."""
        sites = []
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor, match="site_config:*", count=100
            )
            for key in keys:
                v = await self.redis.hget(key, "vertical")
                if v and v.decode() == vertical:
                    site_id = key.decode().split(":")[-1]
                    sites.append(site_id)
            if cursor == 0:
                break
        return sites


# =============================================================================
# NEURAL MONITOR (ENSEMBLE)
# =============================================================================

class NeuralMonitor:
    """
    Ensemble drift detection combining all signals.
    
    Aggregates:
    - Macro-Variance spikes
    - Embedding centroid drift
    - Decision latency shifts
    
    Triggers actions when composite drift score exceeds threshold.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.variance_tracker = MacroVarianceTracker(redis_client)
        self.centroid_drift = EmbeddingCentroidDrift(redis_client)
        self.latency_tracker = DecisionLatencyTracker(redis_client)
    
    async def run_drift_check(self) -> Optional[DriftEvent]:
        """
        Run comprehensive drift detection.
        
        Returns DriftEvent if significant drift detected.
        """
        detected_shifts = []
        affected_verticals = set()
        
        # 1. Check variance spike
        variance_result = await self.variance_tracker.check_variance_spike()
        if variance_result:
            detected_shifts.append(variance_result)
            for v in variance_result.get("spiking_verticals", []):
                affected_verticals.add(v["vertical"])
        
        # 2. Check centroid drift per vertical
        verticals = await self._get_all_verticals()
        for vertical in verticals:
            centroid_result = await self.centroid_drift.check_centroid_drift(vertical)
            if centroid_result:
                detected_shifts.append(centroid_result)
                affected_verticals.add(vertical)
            
            latency_result = await self.latency_tracker.check_latency_shift(vertical)
            if latency_result:
                detected_shifts.append(latency_result)
                affected_verticals.add(vertical)
        
        if not detected_shifts:
            return None
        
        # Compute composite drift score
        global_drift_score = self._compute_composite_score(detected_shifts)
        
        # Determine severity
        severity = self._determine_severity(global_drift_score, len(affected_verticals))
        
        # Generate recommended action
        action = self._generate_action(severity, list(affected_verticals), detected_shifts)
        
        event = DriftEvent(
            event_id=f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            drift_type="composite" if len(detected_shifts) > 1 else detected_shifts[0]["type"],
            severity=severity,
            global_drift_score=global_drift_score,
            detected_shifts=detected_shifts,
            affected_verticals=list(affected_verticals),
            recommended_action=action
        )
        
        # Store event
        await self._store_drift_event(event)
        
        # Publish for other services
        await self.redis.publish("drift_events", json.dumps(event.to_dict()))
        
        return event
    
    def _compute_composite_score(self, shifts: List[Dict]) -> float:
        """Compute weighted composite drift score."""
        weights = {
            "variance_spike": 0.4,
            "centroid_drift": 0.4,
            "latency_shift": 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for shift in shifts:
            shift_type = shift.get("type", "")
            weight = weights.get(shift_type, 0.2)
            
            # Normalize individual scores to 0-1
            if shift_type == "variance_spike":
                score = min(shift.get("global_z_score", 0) / 5.0, 1.0)
            elif shift_type == "centroid_drift":
                score = min(shift.get("drift_score", 0), 1.0)
            elif shift_type == "latency_shift":
                score = min(abs(shift.get("p50_shift_pct", 0)) / 50.0, 1.0)
            else:
                score = 0.5
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_severity(self, score: float, num_verticals: int) -> str:
        """Determine severity level."""
        if score > 0.8 or num_verticals > 5:
            return "critical"
        elif score > 0.5 or num_verticals > 3:
            return "high"
        elif score > 0.3:
            return "medium"
        else:
            return "low"
    
    def _generate_action(
        self,
        severity: str,
        affected_verticals: List[str],
        shifts: List[Dict]
    ) -> Dict[str, Any]:
        """Generate recommended action based on drift analysis."""
        
        if severity in ["critical", "high"]:
            action_type = "FORCE_REEXPLORE"
        elif severity == "medium":
            action_type = "INJECT_ENTROPY"
        else:
            action_type = "MONITOR"
        
        # Determine prior bias based on latency shifts
        new_prior_bias = {}
        for shift in shifts:
            if shift.get("type") == "latency_shift":
                if shift.get("direction") == "shortening":
                    new_prior_bias["urgency"] = 0.15
                else:
                    new_prior_bias["trust_building"] = 0.1
        
        return {
            "type": action_type,
            "target_verticals": affected_verticals,
            "new_prior_bias": new_prior_bias,
            "entropy_multiplier": 2.0 if severity in ["critical", "high"] else 1.5,
            "reexplore_arm_count": 3 if severity == "critical" else 2
        }
    
    async def inject_global_entropy(self, verticals: List[str], multiplier: float = 2.0):
        """
        Inject entropy into Thompson Sampling for specified verticals.
        
        Artificially inflates σ to force exploration.
        """
        for vertical in verticals:
            sites = await self.latency_tracker._get_sites_in_vertical(vertical)
            
            for site_id in sites:
                # Get current Thompson parameters
                params = await self.redis.hgetall(f"thompson:{site_id}")
                
                if params:
                    # Inflate beta (uncertainty)
                    current_beta = float(params.get(b"beta", 1))
                    new_beta = current_beta * multiplier
                    
                    await self.redis.hset(f"thompson:{site_id}", "beta", new_beta)
                    
                    logger.info(f"Injected entropy for {site_id}: β {current_beta} -> {new_beta}")
        
        # Log entropy injection
        await self.redis.xadd(
            "entropy_injections",
            {
                "verticals": json.dumps(verticals),
                "multiplier": multiplier,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def _get_all_verticals(self) -> List[str]:
        """Get all unique verticals."""
        verticals = set()
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor, match="site_config:*", count=100
            )
            for key in keys:
                v = await self.redis.hget(key, "vertical")
                if v:
                    verticals.add(v.decode())
            if cursor == 0:
                break
        return list(verticals)
    
    async def _store_drift_event(self, event: DriftEvent):
        """Store drift event for history."""
        await self.redis.xadd(
            "drift_events_history",
            {"event": json.dumps(event.to_dict())},
            maxlen=1000
        )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Neural Monitor",
    description="Ensemble drift detection for LAM system",
    version="1.0.0"
)

redis_client: Optional[redis.Redis] = None
monitor: Optional[NeuralMonitor] = None


class DriftCheckResponse(BaseModel):
    drift_detected: bool
    event: Optional[Dict] = None


class EntropyInjectionRequest(BaseModel):
    verticals: List[str]
    multiplier: float = 2.0


@app.on_event("startup")
async def startup():
    global redis_client, monitor
    redis_client = redis.from_url(REDIS_URL)
    monitor = NeuralMonitor(redis_client)
    
    # Start background drift check loop
    asyncio.create_task(drift_check_loop())


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


async def drift_check_loop():
    """Periodic drift detection."""
    while True:
        await asyncio.sleep(DRIFT_CHECK_INTERVAL)
        try:
            event = await monitor.run_drift_check()
            if event:
                logger.info(f"Drift detected: {event.severity} - {event.drift_type}")
                
                # Auto-execute action for high severity
                if event.severity in ["critical", "high"]:
                    action = event.recommended_action
                    if action["type"] == "FORCE_REEXPLORE":
                        await monitor.inject_global_entropy(
                            action["target_verticals"],
                            action["entropy_multiplier"]
                        )
        except Exception as e:
            logger.error(f"Drift check error: {e}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "last_check": datetime.now(timezone.utc).isoformat()
    }


@app.post("/check", response_model=DriftCheckResponse)
async def trigger_drift_check():
    """Manually trigger drift detection."""
    event = await monitor.run_drift_check()
    return DriftCheckResponse(
        drift_detected=event is not None,
        event=event.to_dict() if event else None
    )


@app.post("/inject-entropy")
async def inject_entropy(request: EntropyInjectionRequest):
    """Manually inject entropy into Thompson Sampling."""
    await monitor.inject_global_entropy(request.verticals, request.multiplier)
    return {
        "status": "injected",
        "verticals": request.verticals,
        "multiplier": request.multiplier
    }


@app.get("/history")
async def get_drift_history(limit: int = 50):
    """Get recent drift events."""
    events = await redis_client.xrange(
        "drift_events_history",
        min="-",
        max="+",
        count=limit
    )
    
    return {
        "events": [
            json.loads(data[b"event"]) for _, data in events
        ]
    }


@app.post("/record-session")
async def record_session_state(
    session_id: str,
    site_id: str,
    vertical: str,
    embedding: List[float],
    converted: bool,
    behavioral_score: float
):
    """Record a session neural state for centroid tracking."""
    await redis_client.hset(
        f"session_state:{vertical}:{session_id}",
        mapping={
            "session_id": session_id,
            "site_id": site_id,
            "embedding": json.dumps(embedding),
            "converted": "1" if converted else "0",
            "behavioral_score": behavioral_score,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )
    await redis_client.expire(f"session_state:{vertical}:{session_id}", 86400 * 30)  # 30 days
    
    return {"status": "recorded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)
