#!/usr/bin/env python3
"""
LAM FORGE — Inference Mode Container Spawner
=============================================

The "Final 1000" System

When a site hits 1000 samples and the Divergence Rule confirms convergence,
the router transitions from Thompson Sampling to Inference Mode:

1. Instead of sampling from uncertainty, it queries the LAM directly
2. The LAM returns a JSON Structure Definition
3. The Docker MCP spawns the container instantly
4. The generated page is served to the user

This module handles the container spawning and lifecycle management.
"""

import os
import json
import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

import docker
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CONTAINER_POOL_SIZE = int(os.getenv("CONTAINER_POOL_SIZE", "10"))
CONTAINER_TTL_SECONDS = int(os.getenv("CONTAINER_TTL_SECONDS", "300"))
MAX_CONCURRENT_SPAWNS = int(os.getenv("MAX_CONCURRENT_SPAWNS", "5"))
BASE_IMAGE = os.getenv("LAM_BASE_IMAGE", "origin-os/page-generator:latest")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference-spawner")

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class StructureDefinition:
    """JSON Structure Definition from LAM inference."""
    structure_hash: str
    layout: Dict[str, Any]          # Component positions, sizes
    copy: Dict[str, str]            # Text content
    styles: Dict[str, Any]          # CSS/styling
    assets: List[str]               # Image/media URLs
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "structure_hash": self.structure_hash,
            "layout": self.layout,
            "copy": self.copy,
            "styles": self.styles,
            "assets": self.assets,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StructureDefinition":
        return cls(**data)
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of structure."""
        content = json.dumps({
            "layout": self.layout,
            "copy": self.copy,
            "styles": self.styles
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ContainerInstance:
    """A spawned container serving a specific structure."""
    container_id: str
    structure_hash: str
    site_id: str
    port: int
    created_at: str
    last_request: str
    request_count: int = 0
    status: str = "running"  # running, stopping, stopped


class InferenceRequest(BaseModel):
    """Request to generate a page via inference."""
    site_id: str
    user_agent: str
    gclid: Optional[str] = None
    session_id: Optional[str] = None
    geo: Optional[Dict[str, str]] = None
    device_type: Optional[str] = None


class SpawnResponse(BaseModel):
    """Response with container URL for serving."""
    container_url: str
    structure_hash: str
    cache_hit: bool
    spawn_latency_ms: float


# =============================================================================
# CONTAINER POOL MANAGER
# =============================================================================

class ContainerPoolManager:
    """
    Manages a pool of pre-warmed containers for instant page generation.
    
    Architecture:
    - Hot Pool: Pre-spawned containers ready for immediate use
    - Warm Pool: Containers with cached structures (LRU eviction)
    - Cold Spawn: On-demand container creation for new structures
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.docker_client = docker.from_env()
        self.spawn_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SPAWNS)
        self.active_containers: Dict[str, ContainerInstance] = {}
        self._port_counter = 9000
    
    async def get_or_spawn(
        self,
        site_id: str,
        structure: StructureDefinition
    ) -> ContainerInstance:
        """
        Get an existing container for this structure or spawn a new one.
        
        1. Check if structure is already running (cache hit)
        2. Check if we have a hot container to assign
        3. Spawn new container (cold start)
        """
        structure_hash = structure.structure_hash or structure.compute_hash()
        
        # 1. Check cache
        cache_key = f"container:{site_id}:{structure_hash}"
        cached = await self.redis.hgetall(cache_key)
        
        if cached and cached.get(b"status") == b"running":
            container_id = cached[b"container_id"].decode()
            if container_id in self.active_containers:
                container = self.active_containers[container_id]
                container.request_count += 1
                container.last_request = datetime.now(timezone.utc).isoformat()
                await self._update_container_stats(container)
                return container
        
        # 2. Spawn new container
        return await self._spawn_container(site_id, structure)
    
    async def _spawn_container(
        self,
        site_id: str,
        structure: StructureDefinition
    ) -> ContainerInstance:
        """Spawn a new container for the given structure."""
        async with self.spawn_semaphore:
            port = self._get_next_port()
            structure_hash = structure.structure_hash or structure.compute_hash()
            
            # Prepare environment
            env = {
                "SITE_ID": site_id,
                "STRUCTURE_HASH": structure_hash,
                "STRUCTURE_JSON": json.dumps(structure.to_dict()),
                "PORT": str(port)
            }
            
            # Spawn container
            try:
                container = self.docker_client.containers.run(
                    image=BASE_IMAGE,
                    environment=env,
                    ports={f"{port}/tcp": port},
                    detach=True,
                    remove=True,
                    name=f"lam-page-{site_id}-{structure_hash[:8]}",
                    labels={
                        "origin-os.service": "lam-page",
                        "origin-os.site-id": site_id,
                        "origin-os.structure-hash": structure_hash
                    }
                )
                
                instance = ContainerInstance(
                    container_id=container.id,
                    structure_hash=structure_hash,
                    site_id=site_id,
                    port=port,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    last_request=datetime.now(timezone.utc).isoformat(),
                    request_count=1,
                    status="running"
                )
                
                self.active_containers[container.id] = instance
                await self._cache_container(instance)
                
                logger.info(f"Spawned container {container.id[:12]} for {site_id}/{structure_hash[:8]}")
                return instance
                
            except docker.errors.DockerException as e:
                logger.error(f"Failed to spawn container: {e}")
                raise HTTPException(status_code=500, detail=f"Container spawn failed: {e}")
    
    async def _cache_container(self, instance: ContainerInstance):
        """Cache container info in Redis."""
        cache_key = f"container:{instance.site_id}:{instance.structure_hash}"
        await self.redis.hset(cache_key, mapping={
            "container_id": instance.container_id,
            "port": instance.port,
            "status": instance.status,
            "created_at": instance.created_at,
            "request_count": instance.request_count
        })
        await self.redis.expire(cache_key, CONTAINER_TTL_SECONDS)
    
    async def _update_container_stats(self, instance: ContainerInstance):
        """Update container stats in Redis."""
        cache_key = f"container:{instance.site_id}:{instance.structure_hash}"
        await self.redis.hset(cache_key, mapping={
            "request_count": instance.request_count,
            "last_request": instance.last_request
        })
        await self.redis.expire(cache_key, CONTAINER_TTL_SECONDS)
    
    def _get_next_port(self) -> int:
        """Get next available port."""
        self._port_counter += 1
        if self._port_counter > 9999:
            self._port_counter = 9000
        return self._port_counter
    
    async def cleanup_idle_containers(self):
        """Remove containers that haven't been used recently."""
        now = datetime.now(timezone.utc)
        to_remove = []
        
        for container_id, instance in self.active_containers.items():
            last_request = datetime.fromisoformat(instance.last_request.replace('Z', '+00:00'))
            idle_seconds = (now - last_request).total_seconds()
            
            if idle_seconds > CONTAINER_TTL_SECONDS:
                to_remove.append(container_id)
        
        for container_id in to_remove:
            await self._stop_container(container_id)
    
    async def _stop_container(self, container_id: str):
        """Stop and remove a container."""
        try:
            container = self.docker_client.containers.get(container_id)
            container.stop(timeout=5)
            logger.info(f"Stopped container {container_id[:12]}")
        except docker.errors.NotFound:
            pass
        except Exception as e:
            logger.error(f"Error stopping container {container_id[:12]}: {e}")
        
        if container_id in self.active_containers:
            instance = self.active_containers.pop(container_id)
            cache_key = f"container:{instance.site_id}:{instance.structure_hash}"
            await self.redis.delete(cache_key)


# =============================================================================
# LAM INFERENCE CLIENT
# =============================================================================

class LAMInferenceClient:
    """
    Client for querying the LAM model in inference mode.
    
    The LAM takes user context and returns a StructureDefinition
    optimized for that specific user.
    """
    
    def __init__(self, redis_client: redis.Redis, lam_url: str = "http://lam-forge:8050"):
        self.redis = redis_client
        self.lam_url = lam_url
    
    async def get_structure_for_user(
        self,
        site_id: str,
        user_context: Dict[str, Any]
    ) -> StructureDefinition:
        """
        Query the LAM for the optimal structure given user context.
        
        In inference mode, the LAM uses the learned embeddings to
        predict the best structure without exploration.
        """
        # Check if site is in inference mode
        mode = await self.redis.hget(f"site_mode:{site_id}", "mode")
        
        if mode != b"inference":
            # Still in exploration mode - use Thompson Sampling
            return await self._get_exploration_structure(site_id)
        
        # Get winning structure from convergence
        winning_hash = await self.redis.hget(f"site_mode:{site_id}", "winning_structure")
        
        if winning_hash:
            # Load cached winning structure
            structure_data = await self.redis.get(f"structure:{site_id}:{winning_hash.decode()}")
            if structure_data:
                return StructureDefinition.from_dict(json.loads(structure_data))
        
        # Fallback: Query LAM directly
        return await self._query_lam(site_id, user_context)
    
    async def _get_exploration_structure(self, site_id: str) -> StructureDefinition:
        """Get structure via Thompson Sampling (exploration mode)."""
        # This would integrate with the existing LAM router
        # For now, return a default structure
        return StructureDefinition(
            structure_hash="default",
            layout={"type": "default"},
            copy={"headline": "Welcome"},
            styles={},
            assets=[],
            metadata={"mode": "exploration"}
        )
    
    async def _query_lam(
        self,
        site_id: str,
        user_context: Dict[str, Any]
    ) -> StructureDefinition:
        """Query the LAM model directly for structure inference."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.lam_url}/infer",
                json={
                    "site_id": site_id,
                    "user_context": user_context
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return StructureDefinition.from_dict(data["structure"])
                else:
                    raise HTTPException(
                        status_code=response.status,
                        detail="LAM inference failed"
                    )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="LAM Inference Spawner",
    description="Container spawner for inference mode page generation",
    version="1.0.0"
)

# Global instances
redis_client: Optional[redis.Redis] = None
pool_manager: Optional[ContainerPoolManager] = None
lam_client: Optional[LAMInferenceClient] = None


@app.on_event("startup")
async def startup():
    global redis_client, pool_manager, lam_client
    redis_client = redis.from_url(REDIS_URL)
    pool_manager = ContainerPoolManager(redis_client)
    lam_client = LAMInferenceClient(redis_client)
    
    # Start cleanup task
    asyncio.create_task(cleanup_loop())


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


async def cleanup_loop():
    """Periodic cleanup of idle containers."""
    while True:
        await asyncio.sleep(60)
        if pool_manager:
            await pool_manager.cleanup_idle_containers()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_containers": len(pool_manager.active_containers) if pool_manager else 0,
        "pool_size": CONTAINER_POOL_SIZE
    }


@app.post("/generate", response_model=SpawnResponse)
async def generate_page(request: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Generate a page for the given user context.
    
    1. Query LAM for optimal structure
    2. Get or spawn container
    3. Return container URL
    """
    import time
    start = time.time()
    
    # Build user context
    user_context = {
        "user_agent": request.user_agent,
        "gclid": request.gclid,
        "session_id": request.session_id,
        "geo": request.geo,
        "device_type": request.device_type
    }
    
    # Get structure from LAM
    structure = await lam_client.get_structure_for_user(
        request.site_id,
        user_context
    )
    
    # Check cache first
    cache_key = f"container:{request.site_id}:{structure.structure_hash}"
    cached = await redis_client.exists(cache_key)
    
    # Get or spawn container
    container = await pool_manager.get_or_spawn(request.site_id, structure)
    
    latency_ms = (time.time() - start) * 1000
    
    # Log for analytics
    background_tasks.add_task(
        log_generation,
        request.site_id,
        structure.structure_hash,
        cached,
        latency_ms
    )
    
    return SpawnResponse(
        container_url=f"http://localhost:{container.port}",
        structure_hash=structure.structure_hash,
        cache_hit=bool(cached),
        spawn_latency_ms=latency_ms
    )


async def log_generation(site_id: str, structure_hash: str, cache_hit: bool, latency_ms: float):
    """Log generation event for analytics."""
    await redis_client.xadd(
        f"generations:{site_id}",
        {
            "structure_hash": structure_hash,
            "cache_hit": str(cache_hit),
            "latency_ms": str(latency_ms),
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        maxlen=10000
    )


@app.get("/containers")
async def list_containers():
    """List all active containers."""
    return {
        "containers": [
            {
                "container_id": c.container_id[:12],
                "site_id": c.site_id,
                "structure_hash": c.structure_hash,
                "port": c.port,
                "request_count": c.request_count,
                "status": c.status
            }
            for c in pool_manager.active_containers.values()
        ]
    }


@app.delete("/containers/{container_id}")
async def stop_container(container_id: str):
    """Stop a specific container."""
    # Find full container ID
    full_id = None
    for cid in pool_manager.active_containers:
        if cid.startswith(container_id):
            full_id = cid
            break
    
    if not full_id:
        raise HTTPException(status_code=404, detail="Container not found")
    
    await pool_manager._stop_container(full_id)
    return {"status": "stopped", "container_id": container_id}


@app.post("/transition/{site_id}")
async def transition_to_inference(site_id: str, winning_structure_hash: str):
    """
    Transition a site from exploration to inference mode.
    
    Called by the Neural Monitor when Divergence Rule confirms convergence.
    """
    await redis_client.hset(f"site_mode:{site_id}", mapping={
        "mode": "inference",
        "winning_structure": winning_structure_hash,
        "transition_time": datetime.now(timezone.utc).isoformat()
    })
    
    logger.info(f"Site {site_id} transitioned to inference mode with structure {winning_structure_hash}")
    
    return {
        "status": "transitioned",
        "site_id": site_id,
        "mode": "inference",
        "winning_structure": winning_structure_hash
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)
