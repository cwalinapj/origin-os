#!/usr/bin/env python3
"""
BANDIT ROUTER — Thompson Sampling Traffic Gateway
==================================================

Routes Google Ads traffic to page variants using Thompson Sampling.
Records behavioral events and triggers divergence checks.

Endpoints:
- GET /route/{site_id} → Redirect to winning variant
- POST /event → Record behavioral event
- GET /health → Health check
- GET /metrics → Prometheus metrics
"""

import os
import random
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional
from contextlib import asynccontextmanager

import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
import numpy as np

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DIVERGENCE_THRESHOLD = float(os.getenv("DIVERGENCE_THRESHOLD", "0.07"))
MIN_EVENTS = int(os.getenv("MIN_EVENTS", "50"))

# Page variant URLs (internal Docker DNS)
PAGE_VARIANTS = {
    "baseline": "http://page-a:80",
    "challenger": "http://page-b:80"
}


# =============================================================================
# METRICS
# =============================================================================

route_requests = Counter(
    'router_requests_total',
    'Total routing requests',
    ['site_id', 'variant']
)

route_latency = Histogram(
    'router_decision_latency_seconds',
    'Time to make routing decision',
    ['loa_level'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5, 1.0)
)

bandit_arm_samples = Gauge(
    'bandit_arm_samples',
    'Samples per arm',
    ['site_id', 'arm_index']
)

bandit_arm_alpha = Gauge(
    'bandit_arm_alpha',
    'Thompson alpha per arm',
    ['site_id', 'arm_index']
)

bandit_arm_beta = Gauge(
    'bandit_arm_beta',
    'Thompson beta per arm',
    ['site_id', 'arm_index']
)


# =============================================================================
# MODELS
# =============================================================================

class BehavioralEvent(BaseModel):
    session_id: str
    page_id: str
    event_type: str  # scroll, click, dwell, conversion
    value: float
    timestamp: Optional[str] = None


class ThompsonArm:
    """Single arm in Thompson Sampling bandit."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.samples = 0
    
    def sample(self) -> float:
        """Draw from Beta distribution."""
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, reward: float):
        """Update with observed reward (0 or 1)."""
        self.alpha += reward
        self.beta += (1 - reward)
        self.samples += 1


class AdsBanditRouter:
    """Thompson Sampling router for A/B testing."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.arms: Dict[str, Dict[str, ThompsonArm]] = {}
    
    async def load_arms(self, site_id: str):
        """Load arm priors from Redis."""
        if site_id not in self.arms:
            self.arms[site_id] = {}
            
            # Load baseline arm
            baseline = await self.redis.hgetall(f"bandit:arms:{site_id}:baseline")
            if baseline:
                self.arms[site_id]["baseline"] = ThompsonArm(
                    alpha=float(baseline.get(b"alpha", 1)),
                    beta=float(baseline.get(b"beta", 1))
                )
            else:
                self.arms[site_id]["baseline"] = ThompsonArm()
            
            # Load challenger arm
            challenger = await self.redis.hgetall(f"bandit:arms:{site_id}:challenger")
            if challenger:
                self.arms[site_id]["challenger"] = ThompsonArm(
                    alpha=float(challenger.get(b"alpha", 1)),
                    beta=float(challenger.get(b"beta", 1))
                )
            else:
                self.arms[site_id]["challenger"] = ThompsonArm()
    
    async def save_arms(self, site_id: str):
        """Persist arm priors to Redis."""
        for variant, arm in self.arms.get(site_id, {}).items():
            await self.redis.hset(
                f"bandit:arms:{site_id}:{variant}",
                mapping={
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "samples": arm.samples
                }
            )
    
    async def select_variant(self, site_id: str) -> str:
        """Select variant using Thompson Sampling."""
        await self.load_arms(site_id)
        
        arms = self.arms[site_id]
        samples = {k: v.sample() for k, v in arms.items()}
        
        # Update metrics
        for i, (variant, arm) in enumerate(arms.items()):
            bandit_arm_samples.labels(site_id=site_id, arm_index=str(i)).set(arm.samples)
            bandit_arm_alpha.labels(site_id=site_id, arm_index=str(i)).set(arm.alpha)
            bandit_arm_beta.labels(site_id=site_id, arm_index=str(i)).set(arm.beta)
        
        return max(samples, key=samples.get)
    
    async def update_arm(self, site_id: str, variant: str, reward: float):
        """Update arm with observed reward."""
        await self.load_arms(site_id)
        
        if variant in self.arms.get(site_id, {}):
            self.arms[site_id][variant].update(reward)
            await self.save_arms(site_id)


# =============================================================================
# APPLICATION
# =============================================================================

redis_client: Optional[redis.Redis] = None
mongo_client: Optional[AsyncIOMotorClient] = None
router: Optional[AdsBanditRouter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, mongo_client, router
    
    redis_client = redis.from_url(REDIS_URL)
    mongo_client = AsyncIOMotorClient(MONGO_URL)
    router = AdsBanditRouter(redis_client)
    
    yield
    
    await redis_client.close()
    mongo_client.close()


app = FastAPI(
    title="Bandit Router",
    description="Thompson Sampling traffic gateway",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/route/{site_id}")
async def route_traffic(site_id: str, request: Request):
    """Route traffic to selected variant via Thompson Sampling."""
    import time
    start = time.time()
    
    # Select variant
    variant = await router.select_variant(site_id)
    
    # Record latency
    latency = time.time() - start
    route_latency.labels(loa_level="1").observe(latency)
    route_requests.labels(site_id=site_id, variant=variant).inc()
    
    # Get target URL
    target_url = PAGE_VARIANTS.get(variant, PAGE_VARIANTS["baseline"])
    
    # Generate session ID
    session_id = f"{site_id}:{variant}:{random.randint(100000, 999999)}"
    
    # Store session mapping
    await redis_client.setex(
        f"session:{session_id}",
        3600,  # 1 hour TTL
        variant
    )
    
    # Redirect with session cookie
    response = RedirectResponse(url=target_url, status_code=302)
    response.set_cookie("session_id", session_id, max_age=3600)
    
    return response


@app.post("/event")
async def record_event(event: BehavioralEvent):
    """Record behavioral event from Neural Collector."""
    # Store in MongoDB
    db = mongo_client.origin_os
    await db.events.insert_one({
        "session_id": event.session_id,
        "page_id": event.page_id,
        "event_type": event.event_type,
        "value": event.value,
        "timestamp": event.timestamp or datetime.now(timezone.utc).isoformat()
    })
    
    # Check for conversion event
    if event.event_type == "conversion":
        # Get variant from session
        variant = await redis_client.get(f"session:{event.session_id}")
        if variant:
            variant = variant.decode()
            site_id = event.page_id.split(":")[0]
            
            # Update bandit arm
            await router.update_arm(site_id, variant, event.value)
    
    return {"status": "recorded"}


@app.get("/stats/{site_id}")
async def get_stats(site_id: str):
    """Get bandit statistics for a site."""
    await router.load_arms(site_id)
    
    arms = router.arms.get(site_id, {})
    stats = {}
    
    for variant, arm in arms.items():
        stats[variant] = {
            "alpha": arm.alpha,
            "beta": arm.beta,
            "samples": arm.samples,
            "mean": arm.alpha / (arm.alpha + arm.beta)
        }
    
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8024)
