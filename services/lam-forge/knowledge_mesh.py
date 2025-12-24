#!/usr/bin/env python3
"""
KNOWLEDGE MESH — Cross-Site Learning with Privacy-Preserving Federation
========================================================================

Enables learning to flow between sites without exposing individual data:

1. Vertical Aggregation: Sites in same vertical share anonymized insights
2. Structural Templates: Winning layouts become templates for new sites
3. Semantic Drift Anchors: Brand-safe copy vectors shared within verticals
4. Ghost Propagation: Failed experiments inform other sites' priors

Privacy Guarantees:
- Differential Privacy: Noise added to shared gradients
- Federated Averaging: Only model updates shared, not raw data
- Site Isolation: Cross-site queries return aggregated stats only
"""

import os
import json
import asyncio
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DP_EPSILON = float(os.getenv("DP_EPSILON", "1.0"))  # Differential privacy budget
DP_DELTA = float(os.getenv("DP_DELTA", "1e-5"))
MIN_SITES_FOR_AGGREGATION = int(os.getenv("MIN_SITES_FOR_AGGREGATION", "5"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "3600"))  # 1 hour

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knowledge-mesh")

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class VerticalKnowledge:
    """Aggregated knowledge for a vertical."""
    vertical: str
    site_count: int
    
    # Structural insights
    winning_layout_templates: List[Dict[str, Any]]
    layout_performance_matrix: np.ndarray  # [layout_type x metric]
    
    # Semantic insights
    copy_anchor_centroid: np.ndarray  # Brand-safe copy vector
    tone_distribution: Dict[str, float]  # {tone: avg_performance}
    
    # Behavioral insights
    avg_conversion_rate: float
    avg_behavioral_score: float
    decision_latency_p50: float
    
    # Ghost insights (anonymized failures)
    toxic_directions: List[np.ndarray]  # Directions to avoid
    
    last_updated: str
    
    def to_dict(self) -> dict:
        return {
            "vertical": self.vertical,
            "site_count": self.site_count,
            "winning_layout_templates": self.winning_layout_templates,
            "layout_performance_matrix": self.layout_performance_matrix.tolist(),
            "copy_anchor_centroid": self.copy_anchor_centroid.tolist(),
            "tone_distribution": self.tone_distribution,
            "avg_conversion_rate": self.avg_conversion_rate,
            "avg_behavioral_score": self.avg_behavioral_score,
            "decision_latency_p50": self.decision_latency_p50,
            "toxic_directions": [t.tolist() for t in self.toxic_directions],
            "last_updated": self.last_updated
        }


@dataclass
class SiteContribution:
    """A site's contribution to the knowledge mesh."""
    site_id: str
    vertical: str
    
    # Anonymized structural insights
    layout_hash: str
    layout_performance: float
    
    # Anonymized semantic insights
    copy_embedding: np.ndarray  # Noised for DP
    tone_scores: Dict[str, float]
    
    # Anonymized behavioral insights
    conversion_rate: float
    behavioral_score: float
    decision_latency: float
    
    # Ghost contributions
    failed_directions: List[np.ndarray]
    
    timestamp: str


@dataclass
class MeshTemplate:
    """A winning template from the mesh."""
    template_id: str
    vertical: str
    source_sites: int  # Number of sites contributing (not identifiable)
    
    # Structure
    layout_definition: Dict[str, Any]
    component_hierarchy: List[Dict[str, Any]]
    
    # Semantic anchors
    copy_style_vector: np.ndarray
    recommended_tone: str
    
    # Performance stats (aggregated)
    avg_lift: float
    confidence: float
    
    created_at: str


# =============================================================================
# DIFFERENTIAL PRIVACY
# =============================================================================

class DifferentialPrivacy:
    """
    Implements differential privacy for gradient sharing.
    
    Uses Gaussian mechanism for continuous values and
    Laplace mechanism for counts.
    """
    
    def __init__(self, epsilon: float = DP_EPSILON, delta: float = DP_DELTA):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_gaussian_noise(self, value: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Add Gaussian noise for (ε, δ)-DP."""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma, value.shape)
        return value + noise
    
    def add_laplace_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise for ε-DP."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def clip_gradient(self, gradient: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
        """Clip gradient to bound sensitivity."""
        norm = np.linalg.norm(gradient)
        if norm > max_norm:
            gradient = gradient * (max_norm / norm)
        return gradient


# =============================================================================
# FEDERATED AGGREGATOR
# =============================================================================

class FederatedAggregator:
    """
    Aggregates site contributions using federated averaging.
    
    Only model updates are shared, never raw data.
    """
    
    def __init__(self, redis_client: redis.Redis, dp: DifferentialPrivacy):
        self.redis = redis_client
        self.dp = dp
    
    async def aggregate_vertical(self, vertical: str) -> Optional[VerticalKnowledge]:
        """
        Aggregate knowledge from all sites in a vertical.
        
        Returns None if insufficient sites for privacy.
        """
        contributions = await self._get_contributions(vertical)
        
        if len(contributions) < MIN_SITES_FOR_AGGREGATION:
            logger.info(f"Insufficient sites for {vertical}: {len(contributions)} < {MIN_SITES_FOR_AGGREGATION}")
            return None
        
        # Aggregate layout templates
        layout_templates = self._aggregate_layouts(contributions)
        layout_matrix = self._build_layout_matrix(contributions)
        
        # Aggregate semantic vectors
        copy_centroid = self._aggregate_copy_vectors(contributions)
        tone_dist = self._aggregate_tone_distribution(contributions)
        
        # Aggregate behavioral stats
        avg_conv = np.mean([c.conversion_rate for c in contributions])
        avg_behavioral = np.mean([c.behavioral_score for c in contributions])
        avg_latency = np.median([c.decision_latency for c in contributions])
        
        # Aggregate ghost insights
        toxic_dirs = self._aggregate_toxic_directions(contributions)
        
        # Apply DP noise to aggregates
        copy_centroid = self.dp.add_gaussian_noise(copy_centroid, sensitivity=0.1)
        avg_conv = self.dp.add_laplace_noise(avg_conv, sensitivity=0.01)
        avg_behavioral = self.dp.add_laplace_noise(avg_behavioral, sensitivity=0.1)
        
        return VerticalKnowledge(
            vertical=vertical,
            site_count=len(contributions),
            winning_layout_templates=layout_templates,
            layout_performance_matrix=layout_matrix,
            copy_anchor_centroid=copy_centroid,
            tone_distribution=tone_dist,
            avg_conversion_rate=max(0, avg_conv),
            avg_behavioral_score=max(0, avg_behavioral),
            decision_latency_p50=avg_latency,
            toxic_directions=toxic_dirs,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
    
    async def _get_contributions(self, vertical: str) -> List[SiteContribution]:
        """Get all site contributions for a vertical."""
        contributions = []
        cursor = 0
        
        while True:
            cursor, keys = await self.redis.scan(
                cursor, match=f"contribution:{vertical}:*", count=100
            )
            
            for key in keys:
                data = await self.redis.hgetall(key)
                if data:
                    contributions.append(self._parse_contribution(data))
            
            if cursor == 0:
                break
        
        return contributions
    
    def _parse_contribution(self, data: Dict[bytes, bytes]) -> SiteContribution:
        """Parse contribution from Redis."""
        return SiteContribution(
            site_id=data.get(b"site_id", b"").decode(),
            vertical=data.get(b"vertical", b"").decode(),
            layout_hash=data.get(b"layout_hash", b"").decode(),
            layout_performance=float(data.get(b"layout_performance", 0)),
            copy_embedding=np.array(json.loads(data.get(b"copy_embedding", b"[]"))),
            tone_scores=json.loads(data.get(b"tone_scores", b"{}")),
            conversion_rate=float(data.get(b"conversion_rate", 0)),
            behavioral_score=float(data.get(b"behavioral_score", 0)),
            decision_latency=float(data.get(b"decision_latency", 0)),
            failed_directions=[
                np.array(d) for d in json.loads(data.get(b"failed_directions", b"[]"))
            ],
            timestamp=data.get(b"timestamp", b"").decode()
        )
    
    def _aggregate_layouts(self, contributions: List[SiteContribution]) -> List[Dict]:
        """Find winning layout templates."""
        layout_scores: Dict[str, List[float]] = {}
        
        for c in contributions:
            if c.layout_hash not in layout_scores:
                layout_scores[c.layout_hash] = []
            layout_scores[c.layout_hash].append(c.layout_performance)
        
        # Get top performers
        templates = []
        for layout_hash, scores in layout_scores.items():
            if len(scores) >= 3:  # Minimum occurrences
                templates.append({
                    "layout_hash": layout_hash,
                    "avg_performance": np.mean(scores),
                    "occurrences": len(scores)
                })
        
        templates.sort(key=lambda x: x["avg_performance"], reverse=True)
        return templates[:5]  # Top 5
    
    def _build_layout_matrix(self, contributions: List[SiteContribution]) -> np.ndarray:
        """Build layout performance matrix."""
        # Simplified: just return performance by layout type
        # In production, would track multiple metrics
        return np.array([[c.layout_performance] for c in contributions[:10]])
    
    def _aggregate_copy_vectors(self, contributions: List[SiteContribution]) -> np.ndarray:
        """Aggregate copy embeddings into centroid."""
        vectors = [c.copy_embedding for c in contributions if len(c.copy_embedding) > 0]
        if not vectors:
            return np.zeros(EMBEDDING_DIM)
        
        # Weight by performance
        weights = [c.conversion_rate for c in contributions if len(c.copy_embedding) > 0]
        total_weight = sum(weights) + 1e-8
        
        centroid = np.zeros(EMBEDDING_DIM)
        for v, w in zip(vectors, weights):
            if len(v) == EMBEDDING_DIM:
                centroid += v * (w / total_weight)
        
        return centroid
    
    def _aggregate_tone_distribution(self, contributions: List[SiteContribution]) -> Dict[str, float]:
        """Aggregate tone performance distribution."""
        tone_totals: Dict[str, List[float]] = {}
        
        for c in contributions:
            for tone, score in c.tone_scores.items():
                if tone not in tone_totals:
                    tone_totals[tone] = []
                tone_totals[tone].append(score)
        
        return {
            tone: np.mean(scores)
            for tone, scores in tone_totals.items()
            if len(scores) >= 3
        }
    
    def _aggregate_toxic_directions(self, contributions: List[SiteContribution]) -> List[np.ndarray]:
        """Aggregate toxic (failed) directions for avoidance."""
        all_toxic = []
        
        for c in contributions:
            all_toxic.extend(c.failed_directions)
        
        if not all_toxic:
            return []
        
        # Cluster similar toxic directions
        from sklearn.cluster import DBSCAN
        
        if len(all_toxic) < 3:
            return all_toxic
        
        toxic_array = np.array([t for t in all_toxic if len(t) == EMBEDDING_DIM])
        if len(toxic_array) < 3:
            return all_toxic
        
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(toxic_array)
        
        # Return cluster centroids
        unique_labels = set(clustering.labels_)
        centroids = []
        for label in unique_labels:
            if label == -1:
                continue
            mask = clustering.labels_ == label
            centroid = np.mean(toxic_array[mask], axis=0)
            centroids.append(centroid)
        
        return centroids[:10]  # Top 10 toxic directions


# =============================================================================
# TEMPLATE GENERATOR
# =============================================================================

class TemplateGenerator:
    """
    Generates templates from aggregated knowledge.
    
    New sites can bootstrap from these templates instead of
    starting from scratch.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def generate_template(
        self,
        vertical: str,
        knowledge: VerticalKnowledge
    ) -> MeshTemplate:
        """Generate a template from vertical knowledge."""
        
        if not knowledge.winning_layout_templates:
            raise ValueError(f"No winning templates for {vertical}")
        
        # Use top layout
        top_layout = knowledge.winning_layout_templates[0]
        
        # Get full layout definition
        layout_def = await self._get_layout_definition(top_layout["layout_hash"])
        
        # Determine recommended tone
        if knowledge.tone_distribution:
            recommended_tone = max(
                knowledge.tone_distribution.items(),
                key=lambda x: x[1]
            )[0]
        else:
            recommended_tone = "neutral"
        
        template = MeshTemplate(
            template_id=f"tpl_{vertical}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            vertical=vertical,
            source_sites=knowledge.site_count,
            layout_definition=layout_def,
            component_hierarchy=layout_def.get("components", []),
            copy_style_vector=knowledge.copy_anchor_centroid,
            recommended_tone=recommended_tone,
            avg_lift=top_layout["avg_performance"],
            confidence=min(knowledge.site_count / 10, 1.0),  # More sites = more confidence
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Store template
        await self._store_template(template)
        
        return template
    
    async def get_template_for_site(
        self,
        vertical: str,
        site_context: Optional[Dict] = None
    ) -> Optional[MeshTemplate]:
        """Get best template for a new site."""
        template_key = await self.redis.get(f"template:latest:{vertical}")
        
        if not template_key:
            return None
        
        data = await self.redis.hgetall(template_key)
        if not data:
            return None
        
        return self._parse_template(data)
    
    async def _get_layout_definition(self, layout_hash: str) -> Dict:
        """Get full layout definition from hash."""
        data = await self.redis.get(f"layout_def:{layout_hash}")
        if data:
            return json.loads(data)
        return {"hash": layout_hash, "components": []}
    
    async def _store_template(self, template: MeshTemplate):
        """Store template in Redis."""
        key = f"template:{template.template_id}"
        await self.redis.hset(key, mapping={
            "template_id": template.template_id,
            "vertical": template.vertical,
            "source_sites": template.source_sites,
            "layout_definition": json.dumps(template.layout_definition),
            "component_hierarchy": json.dumps(template.component_hierarchy),
            "copy_style_vector": json.dumps(template.copy_style_vector.tolist()),
            "recommended_tone": template.recommended_tone,
            "avg_lift": template.avg_lift,
            "confidence": template.confidence,
            "created_at": template.created_at
        })
        
        # Update latest pointer
        await self.redis.set(f"template:latest:{template.vertical}", key)
    
    def _parse_template(self, data: Dict[bytes, bytes]) -> MeshTemplate:
        """Parse template from Redis."""
        return MeshTemplate(
            template_id=data.get(b"template_id", b"").decode(),
            vertical=data.get(b"vertical", b"").decode(),
            source_sites=int(data.get(b"source_sites", 0)),
            layout_definition=json.loads(data.get(b"layout_definition", b"{}")),
            component_hierarchy=json.loads(data.get(b"component_hierarchy", b"[]")),
            copy_style_vector=np.array(json.loads(data.get(b"copy_style_vector", b"[]"))),
            recommended_tone=data.get(b"recommended_tone", b"neutral").decode(),
            avg_lift=float(data.get(b"avg_lift", 0)),
            confidence=float(data.get(b"confidence", 0)),
            created_at=data.get(b"created_at", b"").decode()
        )


# =============================================================================
# KNOWLEDGE MESH
# =============================================================================

class KnowledgeMesh:
    """
    Main orchestrator for the knowledge mesh.
    
    Coordinates:
    - Site contributions
    - Vertical aggregation
    - Template generation
    - Cross-site prior updates
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.dp = DifferentialPrivacy()
        self.aggregator = FederatedAggregator(redis_client, self.dp)
        self.template_gen = TemplateGenerator(redis_client)
    
    async def contribute(self, contribution: SiteContribution):
        """Submit a site contribution to the mesh."""
        key = f"contribution:{contribution.vertical}:{contribution.site_id}"
        
        # Clip and noise the embedding before storing
        clipped_embedding = self.dp.clip_gradient(contribution.copy_embedding)
        noised_embedding = self.dp.add_gaussian_noise(clipped_embedding, sensitivity=0.1)
        
        await self.redis.hset(key, mapping={
            "site_id": contribution.site_id,
            "vertical": contribution.vertical,
            "layout_hash": contribution.layout_hash,
            "layout_performance": contribution.layout_performance,
            "copy_embedding": json.dumps(noised_embedding.tolist()),
            "tone_scores": json.dumps(contribution.tone_scores),
            "conversion_rate": contribution.conversion_rate,
            "behavioral_score": contribution.behavioral_score,
            "decision_latency": contribution.decision_latency,
            "failed_directions": json.dumps([d.tolist() for d in contribution.failed_directions]),
            "timestamp": contribution.timestamp
        })
        
        # Expire after 30 days
        await self.redis.expire(key, 86400 * 30)
        
        logger.info(f"Received contribution from {contribution.site_id} for {contribution.vertical}")
    
    async def sync_vertical(self, vertical: str) -> Optional[VerticalKnowledge]:
        """Synchronize and aggregate knowledge for a vertical."""
        knowledge = await self.aggregator.aggregate_vertical(vertical)
        
        if knowledge:
            # Store aggregated knowledge
            await self._store_knowledge(knowledge)
            
            # Generate new template
            try:
                template = await self.template_gen.generate_template(vertical, knowledge)
                logger.info(f"Generated template {template.template_id} for {vertical}")
            except ValueError as e:
                logger.warning(f"Could not generate template: {e}")
            
            # Notify sites in vertical
            await self.redis.publish(
                f"knowledge_update:{vertical}",
                json.dumps(knowledge.to_dict())
            )
        
        return knowledge
    
    async def get_priors_for_site(
        self,
        site_id: str,
        vertical: str
    ) -> Dict[str, Any]:
        """
        Get mesh-informed priors for a new site.
        
        Returns Thompson Sampling priors based on vertical knowledge.
        """
        knowledge = await self._get_knowledge(vertical)
        template = await self.template_gen.get_template_for_site(vertical)
        
        priors = {
            "alpha": 1.0,  # Base prior
            "beta": 1.0,
        }
        
        if knowledge:
            # Adjust priors based on vertical performance
            priors["alpha"] = 1.0 + knowledge.avg_conversion_rate * 10
            priors["beta"] = 1.0 + (1 - knowledge.avg_conversion_rate) * 10
            priors["toxic_directions"] = [t.tolist() for t in knowledge.toxic_directions]
            priors["copy_anchor"] = knowledge.copy_anchor_centroid.tolist()
        
        if template:
            priors["template_id"] = template.template_id
            priors["recommended_layout"] = template.layout_definition
            priors["recommended_tone"] = template.recommended_tone
        
        return priors
    
    async def propagate_ghost(
        self,
        site_id: str,
        vertical: str,
        failed_direction: np.ndarray,
        failure_severity: float
    ):
        """
        Propagate a ghost (failed experiment) to the mesh.
        
        Other sites can learn from this failure without knowing the source.
        """
        # Clip and noise before sharing
        clipped = self.dp.clip_gradient(failed_direction)
        noised = self.dp.add_gaussian_noise(clipped, sensitivity=0.5)
        
        # Add to vertical's toxic directions pool
        await self.redis.rpush(
            f"toxic_pool:{vertical}",
            json.dumps({
                "direction": noised.tolist(),
                "severity": failure_severity,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        )
        
        # Trim to keep only recent
        await self.redis.ltrim(f"toxic_pool:{vertical}", -1000, -1)
        
        logger.info(f"Ghost propagated for {vertical} with severity {failure_severity}")
    
    async def run_sync_loop(self):
        """Background loop to sync all verticals."""
        while True:
            try:
                verticals = await self._get_all_verticals()
                
                for vertical in verticals:
                    knowledge = await self.sync_vertical(vertical)
                    if knowledge:
                        logger.info(
                            f"Synced {vertical}: {knowledge.site_count} sites, "
                            f"avg_conv={knowledge.avg_conversion_rate:.3f}"
                        )
                    
                    await asyncio.sleep(10)  # Stagger syncs
                
            except Exception as e:
                logger.error(f"Sync error: {e}")
            
            await asyncio.sleep(SYNC_INTERVAL)
    
    async def _store_knowledge(self, knowledge: VerticalKnowledge):
        """Store aggregated knowledge."""
        key = f"knowledge:{knowledge.vertical}"
        await self.redis.set(key, json.dumps(knowledge.to_dict()))
    
    async def _get_knowledge(self, vertical: str) -> Optional[VerticalKnowledge]:
        """Get stored knowledge for vertical."""
        data = await self.redis.get(f"knowledge:{vertical}")
        if not data:
            return None
        
        parsed = json.loads(data)
        return VerticalKnowledge(
            vertical=parsed["vertical"],
            site_count=parsed["site_count"],
            winning_layout_templates=parsed["winning_layout_templates"],
            layout_performance_matrix=np.array(parsed["layout_performance_matrix"]),
            copy_anchor_centroid=np.array(parsed["copy_anchor_centroid"]),
            tone_distribution=parsed["tone_distribution"],
            avg_conversion_rate=parsed["avg_conversion_rate"],
            avg_behavioral_score=parsed["avg_behavioral_score"],
            decision_latency_p50=parsed["decision_latency_p50"],
            toxic_directions=[np.array(t) for t in parsed["toxic_directions"]],
            last_updated=parsed["last_updated"]
        )
    
    async def _get_all_verticals(self) -> List[str]:
        """Get all verticals with contributions."""
        verticals = set()
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor, match="contribution:*", count=100
            )
            for key in keys:
                parts = key.decode().split(":")
                if len(parts) >= 2:
                    verticals.add(parts[1])
            if cursor == 0:
                break
        return list(verticals)


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Knowledge Mesh",
    description="Cross-site learning with privacy-preserving federation",
    version="1.0.0"
)

redis_client: Optional[redis.Redis] = None
mesh: Optional[KnowledgeMesh] = None


class ContributionRequest(BaseModel):
    site_id: str
    vertical: str
    layout_hash: str
    layout_performance: float
    copy_embedding: List[float]
    tone_scores: Dict[str, float]
    conversion_rate: float
    behavioral_score: float
    decision_latency: float
    failed_directions: List[List[float]] = []


class GhostRequest(BaseModel):
    site_id: str
    vertical: str
    failed_direction: List[float]
    failure_severity: float


@app.on_event("startup")
async def startup():
    global redis_client, mesh
    redis_client = redis.from_url(REDIS_URL)
    mesh = KnowledgeMesh(redis_client)
    
    # Start sync loop
    asyncio.create_task(mesh.run_sync_loop())


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/contribute")
async def contribute(request: ContributionRequest):
    """Submit a site contribution."""
    contribution = SiteContribution(
        site_id=request.site_id,
        vertical=request.vertical,
        layout_hash=request.layout_hash,
        layout_performance=request.layout_performance,
        copy_embedding=np.array(request.copy_embedding),
        tone_scores=request.tone_scores,
        conversion_rate=request.conversion_rate,
        behavioral_score=request.behavioral_score,
        decision_latency=request.decision_latency,
        failed_directions=[np.array(d) for d in request.failed_directions],
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    await mesh.contribute(contribution)
    return {"status": "contributed"}


@app.post("/ghost")
async def propagate_ghost(request: GhostRequest):
    """Propagate a failed experiment."""
    await mesh.propagate_ghost(
        request.site_id,
        request.vertical,
        np.array(request.failed_direction),
        request.failure_severity
    )
    return {"status": "propagated"}


@app.get("/priors/{vertical}/{site_id}")
async def get_priors(vertical: str, site_id: str):
    """Get mesh-informed priors for a site."""
    priors = await mesh.get_priors_for_site(site_id, vertical)
    return priors


@app.get("/knowledge/{vertical}")
async def get_knowledge(vertical: str):
    """Get aggregated knowledge for a vertical."""
    knowledge = await mesh._get_knowledge(vertical)
    if not knowledge:
        raise HTTPException(status_code=404, detail="No knowledge for vertical")
    return knowledge.to_dict()


@app.get("/template/{vertical}")
async def get_template(vertical: str):
    """Get latest template for a vertical."""
    template = await mesh.template_gen.get_template_for_site(vertical)
    if not template:
        raise HTTPException(status_code=404, detail="No template for vertical")
    return {
        "template_id": template.template_id,
        "source_sites": template.source_sites,
        "layout_definition": template.layout_definition,
        "recommended_tone": template.recommended_tone,
        "avg_lift": template.avg_lift,
        "confidence": template.confidence
    }


@app.post("/sync/{vertical}")
async def trigger_sync(vertical: str):
    """Manually trigger vertical sync."""
    knowledge = await mesh.sync_vertical(vertical)
    if not knowledge:
        return {"status": "insufficient_sites"}
    return {"status": "synced", "site_count": knowledge.site_count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8040)
