#!/usr/bin/env python3
"""
PER-VERTICAL LAM HEADS — The "Specialist" Architecture
=======================================================

Solves the "Averaging Paradox" where a single global policy would try to
find a "universal marketing truth" that doesn't exist.

Architecture:
- Shared Backbone: Universal patterns (F-pattern reading, contrast → CTA clicks)
- Vertical Heads: Social-cognitive norms specific to each domain

Knowledge Transfer Without Contamination:
- Shared Encoder: Distills "Visual Grammar" and "Behavioral Physics"
- Vertical Heads: Translate grammar into specific mutation vectors

Gradient Isolation:
- When B2B fails: gradients → B2B head + backbone (E-commerce frozen)
- When E-commerce wins: gradients → E-commerce head + backbone (B2B frozen)

Scaling Path:
- Multi-Head (3-10 verticals): Current implementation
- LoRA Adaptation (100+ verticals): Future implementation
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MODEL_DIR = os.getenv("MODEL_DIR", "/models/lam")

# Architecture config
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "512"))
NUM_ATTENTION_HEADS = int(os.getenv("NUM_ATTENTION_HEADS", "8"))
NUM_BACKBONE_LAYERS = int(os.getenv("NUM_BACKBONE_LAYERS", "6"))

# Default verticals
DEFAULT_VERTICALS = ["ecommerce", "b2b", "saas", "media", "finance", "healthcare"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vertical-heads")

# =============================================================================
# VERTICAL PROFILES
# =============================================================================

class VerticalProfile:
    """
    Defines the social-cognitive norms for a vertical.
    
    Each vertical has different:
    - Risk tolerance (how aggressive mutations can be)
    - Trust requirements (how much brand consistency matters)
    - Conversion timelines (immediate vs. nurturing)
    - Copy tone preferences
    """
    
    PROFILES = {
        "ecommerce": {
            "risk_tolerance": 0.8,       # High: Flash sales, urgency works
            "trust_requirement": 0.3,     # Low: Transactional relationship
            "conversion_timeline": 0.1,   # Immediate: Buy now
            "preferred_tones": ["urgent", "promotional", "exciting"],
            "forbidden_tones": ["academic", "formal"],
            "mutation_aggressiveness": 0.9,
            "description": "Fast, transactional, impulse-driven"
        },
        "b2b": {
            "risk_tolerance": 0.2,        # Low: Trust is everything
            "trust_requirement": 0.9,     # High: Long-term relationships
            "conversion_timeline": 0.9,   # Long: Nurturing required
            "preferred_tones": ["professional", "trustworthy", "authoritative"],
            "forbidden_tones": ["urgent", "promotional", "casual"],
            "mutation_aggressiveness": 0.3,
            "description": "Trust-focused, long sales cycle, high stakes"
        },
        "saas": {
            "risk_tolerance": 0.5,        # Medium: Feature-focused
            "trust_requirement": 0.6,     # Medium-high: Security matters
            "conversion_timeline": 0.5,   # Medium: Free trial → Paid
            "preferred_tones": ["professional", "helpful", "innovative"],
            "forbidden_tones": ["pushy", "desperate"],
            "mutation_aggressiveness": 0.5,
            "description": "Feature-driven, trial-focused, technical"
        },
        "media": {
            "risk_tolerance": 0.7,        # High: Attention is currency
            "trust_requirement": 0.4,     # Medium-low: Entertainment
            "conversion_timeline": 0.2,   # Fast: Subscribe now
            "preferred_tones": ["engaging", "exciting", "casual"],
            "forbidden_tones": ["boring", "corporate"],
            "mutation_aggressiveness": 0.7,
            "description": "Attention-grabbing, engagement-focused"
        },
        "finance": {
            "risk_tolerance": 0.1,        # Very low: Regulatory constraints
            "trust_requirement": 0.95,    # Very high: Money is at stake
            "conversion_timeline": 0.8,   # Long: Due diligence
            "preferred_tones": ["trustworthy", "professional", "reassuring"],
            "forbidden_tones": ["urgent", "promotional", "casual"],
            "mutation_aggressiveness": 0.2,
            "description": "Highly regulated, trust-critical, conservative"
        },
        "healthcare": {
            "risk_tolerance": 0.1,        # Very low: Health is sensitive
            "trust_requirement": 0.95,    # Very high: Life at stake
            "conversion_timeline": 0.7,   # Long: Research, consult
            "preferred_tones": ["compassionate", "professional", "reassuring"],
            "forbidden_tones": ["urgent", "promotional", "alarming"],
            "mutation_aggressiveness": 0.15,
            "description": "Highly sensitive, trust-critical, empathetic"
        }
    }
    
    @classmethod
    def get(cls, vertical: str) -> Dict[str, Any]:
        """Get profile for a vertical."""
        return cls.PROFILES.get(vertical, cls.PROFILES["saas"])  # Default to SaaS
    
    @classmethod
    def get_mutation_bounds(cls, vertical: str) -> Tuple[float, float]:
        """Get mutation vector bounds for a vertical."""
        profile = cls.get(vertical)
        aggressiveness = profile["mutation_aggressiveness"]
        
        # Higher aggressiveness = wider mutation bounds
        max_magnitude = aggressiveness * 1.5
        return (-max_magnitude, max_magnitude)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class SharedBackbone(nn.Module):
    """
    Shared encoder that learns universal patterns.
    
    Universal patterns include:
    - Visual grammar (F-pattern, contrast, hierarchy)
    - Behavioral physics (scroll dynamics, dwell patterns)
    - Intent signals (mouse movement, click patterns)
    """
    
    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_ATTENTION_HEADS,
        num_layers: int = NUM_BACKBONE_LAYERS,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Behavioral physics encoder
        self.behavioral_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode page state into universal representation.
        
        Args:
            x: Page state tensor [batch, seq_len, input_dim]
        
        Returns:
            Universal features [batch, hidden_dim]
        """
        # Project to hidden dim
        x = self.input_projection(x)
        
        # Transformer encoding
        x = self.encoder(x)
        
        # Pool to single vector (mean pooling)
        x = x.mean(dim=1)
        
        # Behavioral encoding
        x = self.behavioral_encoder(x)
        
        return self.layer_norm(x)


class VerticalHead(nn.Module):
    """
    Specialized head for a specific vertical.
    
    Translates universal features into vertical-specific:
    - Mutation vectors
    - Reward predictions
    - Confidence scores
    """
    
    def __init__(
        self,
        vertical: str,
        hidden_dim: int = HIDDEN_DIM,
        mutation_dim: int = EMBEDDING_DIM,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vertical = vertical
        self.profile = VerticalProfile.get(vertical)
        
        # Mutation direction predictor
        self.mutation_head = nn.Sequential(
            nn.Linear(hidden_dim + mutation_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, mutation_dim)
        )
        
        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + mutation_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Vertical-specific scaling based on profile
        self.aggressiveness_scale = self.profile["mutation_aggressiveness"]
    
    def forward(
        self,
        backbone_features: torch.Tensor,
        mutation_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate vertical-specific predictions.
        
        Returns:
            (predicted_mutation, predicted_reward, confidence)
        """
        # Combine backbone features with mutation
        combined = torch.cat([backbone_features, mutation_vector], dim=-1)
        
        # Predict optimal mutation direction
        pred_mutation = self.mutation_head(combined)
        
        # Scale mutation by vertical aggressiveness
        pred_mutation = pred_mutation * self.aggressiveness_scale
        
        # Predict reward
        pred_reward = self.reward_head(combined)
        
        # Estimate confidence
        confidence = self.confidence_head(backbone_features)
        
        return pred_mutation, pred_reward, confidence


class MultiHeadLAM(nn.Module):
    """
    Multi-Head Large Action Model with per-vertical specialization.
    
    Architecture:
    - Shared backbone: Universal pattern learning
    - Vertical heads: Domain-specific policy heads
    
    Gradient isolation ensures cross-vertical contamination is prevented.
    """
    
    def __init__(
        self,
        verticals: List[str] = DEFAULT_VERTICALS,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM
    ):
        super().__init__()
        
        self.verticals = verticals
        self.hidden_dim = hidden_dim
        
        # Shared backbone
        self.backbone = SharedBackbone(input_dim, hidden_dim)
        
        # Per-vertical heads
        self.heads = nn.ModuleDict({
            v: VerticalHead(v, hidden_dim, input_dim)
            for v in verticals
        })
        
        # Vertical embedding (for future LoRA scaling)
        self.vertical_embeddings = nn.Embedding(len(verticals), hidden_dim)
        self.vertical_to_idx = {v: i for i, v in enumerate(verticals)}
    
    def forward(
        self,
        x: torch.Tensor,
        mutation_vector: torch.Tensor,
        vertical: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through vertical-specific head.
        
        Args:
            x: Page state [batch, seq_len, input_dim]
            mutation_vector: Proposed mutation [batch, mutation_dim]
            vertical: Target vertical
        
        Returns:
            (predicted_mutation, predicted_reward, confidence)
        """
        # Shared backbone encoding
        backbone_features = self.backbone(x)
        
        # Add vertical embedding (conditioning)
        if vertical in self.vertical_to_idx:
            v_idx = torch.tensor([self.vertical_to_idx[vertical]] * x.size(0))
            v_emb = self.vertical_embeddings(v_idx.to(x.device))
            backbone_features = backbone_features + v_emb * 0.1
        
        # Route to vertical-specific head
        if vertical not in self.heads:
            logger.warning(f"Unknown vertical {vertical}, using default")
            vertical = "saas"
        
        head = self.heads[vertical]
        return head(backbone_features, mutation_vector)
    
    def get_active_parameters(self, vertical: str) -> List[nn.Parameter]:
        """Get parameters that should be updated for a vertical."""
        params = list(self.backbone.parameters())
        if vertical in self.heads:
            params.extend(self.heads[vertical].parameters())
        return params
    
    def freeze_other_heads(self, active_vertical: str):
        """Freeze all heads except the active one."""
        for v, head in self.heads.items():
            for param in head.parameters():
                param.requires_grad = (v == active_vertical)
    
    def unfreeze_all_heads(self):
        """Unfreeze all heads."""
        for head in self.heads.values():
            for param in head.parameters():
                param.requires_grad = True


# =============================================================================
# VERTICAL-AWARE TRAINER
# =============================================================================

class VerticalAwareTrainer:
    """
    Trainer with selective gradient updates per vertical.
    
    When a tombstone is recorded for B2B:
    - Gradients → B2B head + backbone
    - E-commerce, SaaS heads remain FROZEN
    """
    
    def __init__(
        self,
        model: MultiHeadLAM,
        redis_client: redis.Redis,
        mongo_client: AsyncIOMotorClient,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.redis = redis_client
        self.mongo = mongo_client
        self.db = mongo_client.origin_os
        
        # Per-vertical optimizers
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._setup_optimizers()
        
        # Metrics tracking
        self.vertical_metrics: Dict[str, Dict[str, List[float]]] = {
            v: {"loss": [], "accuracy": [], "confidence": []}
            for v in model.verticals
        }
    
    def _setup_optimizers(self):
        """Create per-vertical optimizers."""
        # Shared optimizer for backbone (always updates)
        backbone_params = list(self.model.backbone.parameters())
        
        for vertical in self.model.verticals:
            head_params = list(self.model.heads[vertical].parameters())
            
            # Each vertical has its own optimizer for its head + shared backbone
            self.optimizers[vertical] = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": 1e-4},
                    {"params": head_params, "lr": 5e-4}  # Higher LR for heads
                ],
                weight_decay=0.01
            )
    
    def train_step(
        self,
        batch: List[Dict[str, Any]],
        shaped_reward_fn
    ) -> Dict[str, float]:
        """
        Training step with selective gradient updates.
        
        Each sample only updates its vertical's head + backbone.
        """
        self.model.train()
        
        # Group by vertical
        vertical_batches: Dict[str, List[Dict]] = {}
        for sample in batch:
            vertical = sample.get("meta", {}).get("category", "saas")
            if vertical not in vertical_batches:
                vertical_batches[vertical] = []
            vertical_batches[vertical].append(sample)
        
        total_loss = 0.0
        sample_count = 0
        
        for vertical, v_batch in vertical_batches.items():
            if vertical not in self.optimizers:
                continue
            
            # Freeze other heads
            self.model.freeze_other_heads(vertical)
            
            optimizer = self.optimizers[vertical]
            optimizer.zero_grad()
            
            v_loss = 0.0
            
            for sample in v_batch:
                # Featurize
                x = self._featurize(sample)
                mutation = torch.tensor(
                    sample.get("mutation_vector", [0] * EMBEDDING_DIM),
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                # Forward pass
                pred_mutation, pred_reward, confidence = self.model(
                    x, mutation, vertical
                )
                
                # Get shaped reward target
                target_reward = shaped_reward_fn(sample)
                target_tensor = torch.tensor(
                    [[target_reward]], dtype=torch.float32
                ).to(self.device)
                
                # Loss: MSE on reward + confidence calibration
                reward_loss = F.mse_loss(pred_reward, target_tensor)
                
                # Confidence should be low when wrong, high when right
                error = (pred_reward - target_tensor).abs()
                conf_target = torch.exp(-error * 2)  # Exponential decay
                conf_loss = F.mse_loss(confidence, conf_target)
                
                loss = reward_loss + 0.1 * conf_loss
                v_loss += loss
                sample_count += 1
            
            # Backward
            if len(v_batch) > 0:
                v_loss = v_loss / len(v_batch)
                v_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_active_parameters(vertical), 1.0
                )
                
                optimizer.step()
                total_loss += v_loss.item() * len(v_batch)
                
                # Track metrics
                self.vertical_metrics[vertical]["loss"].append(v_loss.item())
        
        # Unfreeze all heads
        self.model.unfreeze_all_heads()
        
        return {
            "total_loss": total_loss / max(sample_count, 1),
            "samples_processed": sample_count,
            "verticals_updated": list(vertical_batches.keys())
        }
    
    def _featurize(self, sample: Dict) -> torch.Tensor:
        """Convert sample to tensor."""
        base_state = sample.get("base_state", [0] * EMBEDDING_DIM)
        # Reshape to [batch, seq_len, dim]
        x = torch.tensor(base_state, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        return x.to(self.device)
    
    async def get_vertical_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics per vertical."""
        metrics = {}
        for vertical, data in self.vertical_metrics.items():
            metrics[vertical] = {
                "avg_loss": np.mean(data["loss"][-100:]) if data["loss"] else 0,
                "avg_accuracy": np.mean(data["accuracy"][-100:]) if data["accuracy"] else 0,
                "sample_count": len(data["loss"])
            }
        return metrics
    
    async def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_states": {
                v: opt.state_dict() for v, opt in self.optimizers.items()
            },
            "vertical_metrics": self.vertical_metrics
        }, path)


# =============================================================================
# CROSS-DOMAIN TRANSFER TRACKER
# =============================================================================

class CrossDomainTracker:
    """
    Tracks knowledge transfer between verticals through shared backbone.
    
    When e-commerce discovers optimal mobile navigation:
    1. Backbone updates its "mobile nav" representation
    2. B2B benefits from this on next inference
    3. We track this "Shared Wisdom Gain"
    """
    
    def __init__(self, model: MultiHeadLAM, redis_client: redis.Redis):
        self.model = model
        self.redis = redis_client
        
        # Track backbone changes per vertical
        self.backbone_snapshots: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def snapshot_backbone(self, vertical: str):
        """Take snapshot of backbone state after vertical update."""
        self.backbone_snapshots[vertical] = {
            name: param.clone().detach()
            for name, param in self.model.backbone.named_parameters()
        }
    
    def compute_transfer_gain(
        self,
        source_vertical: str,
        target_vertical: str
    ) -> float:
        """
        Compute knowledge transfer from source to target.
        
        Measures how much the backbone changed from source's update
        that would benefit target.
        """
        if source_vertical not in self.backbone_snapshots:
            return 0.0
        
        source_snapshot = self.backbone_snapshots[source_vertical]
        
        total_change = 0.0
        param_count = 0
        
        for name, current_param in self.model.backbone.named_parameters():
            if name in source_snapshot:
                diff = (current_param - source_snapshot[name]).abs().mean()
                total_change += diff.item()
                param_count += 1
        
        return total_change / max(param_count, 1)
    
    async def log_transfer_event(
        self,
        source_vertical: str,
        pattern_type: str,
        description: str
    ):
        """Log a cross-domain transfer event."""
        await self.redis.xadd(
            "transfer_events",
            {
                "source": source_vertical,
                "pattern": pattern_type,
                "description": description,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            maxlen=1000
        )


# =============================================================================
# LORA ADAPTER (FUTURE SCALING)
# =============================================================================

class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation for scaling to 100+ verticals.
    
    Instead of full heads, we inject small learnable matrices
    into the backbone based on vertical embedding.
    
    Parameters:
    - Original: O(verticals × head_params)
    - LoRA: O(verticals × rank × 2)
    
    For 100 verticals with rank=8:
    - Multi-Head: 100 × 1M = 100M params
    - LoRA: 100 × 8 × 2 × 512 = 0.8M params
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta."""
        return self.lora_B(self.lora_A(x)) * self.scaling


class LoRABackbone(nn.Module):
    """
    Backbone with LoRA adapters for each vertical.
    
    This is the future scaling path for 100+ verticals.
    """
    
    def __init__(
        self,
        base_backbone: SharedBackbone,
        num_verticals: int = 100,
        lora_rank: int = 8
    ):
        super().__init__()
        
        self.backbone = base_backbone
        self.lora_rank = lora_rank
        
        # LoRA adapters per vertical
        self.lora_adapters = nn.ModuleList([
            LoRAAdapter(HIDDEN_DIM, HIDDEN_DIM, rank=lora_rank)
            for _ in range(num_verticals)
        ])
        
        # Vertical embedding for interpolation
        self.vertical_embeddings = nn.Embedding(num_verticals, HIDDEN_DIM)
    
    def forward(
        self,
        x: torch.Tensor,
        vertical_idx: int
    ) -> torch.Tensor:
        """Forward with vertical-specific LoRA adaptation."""
        # Base backbone features
        features = self.backbone(x)
        
        # Apply LoRA adapter for this vertical
        if vertical_idx < len(self.lora_adapters):
            lora_delta = self.lora_adapters[vertical_idx](features)
            features = features + lora_delta
        
        return features


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(
    title="Per-Vertical LAM Heads",
    description="Specialist architecture with multi-head LAM",
    version="1.0.0"
)

redis_client: Optional[redis.Redis] = None
mongo_client: Optional[AsyncIOMotorClient] = None
model: Optional[MultiHeadLAM] = None
trainer: Optional[VerticalAwareTrainer] = None


class PredictRequest(BaseModel):
    page_state: List[float]
    mutation_vector: List[float]
    vertical: str


class TrainBatchRequest(BaseModel):
    samples: List[Dict[str, Any]]


class VerticalProfileResponse(BaseModel):
    vertical: str
    profile: Dict[str, Any]


@app.on_event("startup")
async def startup():
    global redis_client, mongo_client, model, trainer
    redis_client = redis.from_url(REDIS_URL)
    mongo_client = AsyncIOMotorClient(MONGO_URL)
    
    model = MultiHeadLAM(DEFAULT_VERTICALS)
    trainer = VerticalAwareTrainer(model, redis_client, mongo_client)


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()
    if mongo_client:
        mongo_client.close()


@app.get("/health")
async def health():
    return {"status": "healthy", "verticals": model.verticals if model else []}


@app.post("/predict")
async def predict(request: PredictRequest):
    """Get prediction from vertical-specific head."""
    x = torch.tensor(request.page_state, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
    
    mutation = torch.tensor(request.mutation_vector, dtype=torch.float32)
    mutation = mutation.unsqueeze(0)  # [1, dim]
    
    model.eval()
    with torch.no_grad():
        pred_mutation, pred_reward, confidence = model(
            x, mutation, request.vertical
        )
    
    return {
        "predicted_mutation": pred_mutation.squeeze().tolist(),
        "predicted_reward": pred_reward.item(),
        "confidence": confidence.item(),
        "vertical": request.vertical
    }


@app.get("/profile/{vertical}")
async def get_profile(vertical: str) -> VerticalProfileResponse:
    """Get vertical profile."""
    profile = VerticalProfile.get(vertical)
    return VerticalProfileResponse(vertical=vertical, profile=profile)


@app.get("/profiles")
async def get_all_profiles():
    """Get all vertical profiles."""
    return {"profiles": VerticalProfile.PROFILES}


@app.get("/metrics")
async def get_metrics():
    """Get training metrics per vertical."""
    if trainer:
        return await trainer.get_vertical_metrics()
    return {}


@app.post("/add-vertical/{vertical}")
async def add_vertical(vertical: str):
    """Add a new vertical head."""
    if vertical in model.verticals:
        raise HTTPException(status_code=400, detail="Vertical already exists")
    
    # Add new head
    model.verticals.append(vertical)
    model.heads[vertical] = VerticalHead(vertical, model.hidden_dim, EMBEDDING_DIM)
    model.vertical_to_idx[vertical] = len(model.verticals) - 1
    
    # Setup optimizer
    trainer._setup_optimizers()
    
    return {"status": "added", "vertical": vertical}


@app.get("/architecture")
async def get_architecture():
    """Get model architecture info."""
    return {
        "type": "MultiHeadLAM",
        "backbone_params": sum(p.numel() for p in model.backbone.parameters()),
        "heads": {
            v: sum(p.numel() for p in h.parameters())
            for v, h in model.heads.items()
        },
        "total_params": sum(p.numel() for p in model.parameters()),
        "verticals": model.verticals
    }


@app.get("/scaling-comparison")
async def scaling_comparison():
    """Compare Multi-Head vs LoRA scaling."""
    current_verticals = len(model.verticals)
    head_params = sum(
        sum(p.numel() for p in h.parameters())
        for h in model.heads.values()
    )
    
    # LoRA projection
    lora_rank = 8
    lora_params_per_vertical = lora_rank * 2 * HIDDEN_DIM
    
    return {
        "current": {
            "architecture": "Multi-Head",
            "verticals": current_verticals,
            "head_params": head_params,
            "best_for": "3-10 distinct domains",
            "vertical_isolation": "Absolute"
        },
        "future": {
            "architecture": "LoRA",
            "projected_verticals": 100,
            "params_per_vertical": lora_params_per_vertical,
            "total_params": lora_params_per_vertical * 100,
            "best_for": "100+ niche sub-verticals",
            "vertical_isolation": "Relative (Interpolated)"
        },
        "scaling_factor": head_params / (lora_params_per_vertical * current_verticals)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8110)
