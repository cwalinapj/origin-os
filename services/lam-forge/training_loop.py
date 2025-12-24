#!/usr/bin/env python3
"""
LAM FORGE — Training Loop
=========================

Implements the Triplet Loss architecture for LAM training:
- Anchor: Current page state (S_base)
- Positive: Mutation direction that resulted in a win
- Negative: Mutation direction that resulted in an Inversion (Confident Failure)

Integrates:
- Failure Inversion packets
- Ghost Memory from tombstoned variants
- Global Knowledge Mesh synchronization
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import redis.asyncio as redis

from inversion_math import (
    InversionMath,
    InversionPacket,
    TripletInversionLoss,
    GhostMemory
)

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/lam"))
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "1000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))
STRUCTURAL_DIMS = int(os.getenv("STRUCTURAL_DIMS", "64"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lam-forge-training")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class HTMLEncoder(nn.Module):
    """
    Encodes HTML/page state into a latent representation.
    
    Uses a transformer-based architecture to understand
    both structural (DOM) and semantic (content) features.
    """
    
    def __init__(self, vocab_size: int = 50000, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len] token IDs
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.transpose(1, 2)  # [batch, embed_dim, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch, embed_dim]
        return x


class RewardHead(nn.Module):
    """
    Predicts reward given page embedding and mutation vector.
    
    reward = f(page_state, mutation_direction)
    """
    
    def __init__(self, page_dim: int = 256, mutation_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(page_dim + mutation_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, page_features: torch.Tensor, mutation_vector: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([page_features, mutation_vector], dim=-1)
        return self.fc(combined).squeeze(-1)


class LAMModel(nn.Module):
    """
    Large Action Model for page optimization.
    
    Components:
    - Encoder: Understands page structure and content
    - Reward Head: Predicts expected reward for mutations
    - Mutation Generator: Proposes new mutations (future)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}
        
        self.encoder = HTMLEncoder(
            vocab_size=config.get("vocab_size", 50000),
            embed_dim=config.get("embed_dim", EMBEDDING_DIM)
        )
        self.reward_head = RewardHead(
            page_dim=config.get("embed_dim", EMBEDDING_DIM),
            mutation_dim=config.get("mutation_dim", EMBEDDING_DIM)
        )
    
    def forward(
        self,
        page_tokens: torch.Tensor,
        mutation_vector: torch.Tensor
    ) -> torch.Tensor:
        """Predict reward for mutation on page."""
        page_features = self.encoder(page_tokens)
        reward = self.reward_head(page_features, mutation_vector)
        return reward
    
    def encode_page(self, page_tokens: torch.Tensor) -> torch.Tensor:
        """Get page embedding without mutation."""
        return self.encoder(page_tokens)


# =============================================================================
# TRAINING DATASET
# =============================================================================

@dataclass
class TrainingSample:
    """A single training sample with positive and negative mutations."""
    page_tokens: torch.Tensor
    positive_mutation: torch.Tensor
    negative_mutation: torch.Tensor
    positive_reward: float
    negative_reward: float
    penalty_weight: float
    site_id: str
    vertical: str


class LAMDataset(Dataset):
    """Dataset of training samples from inversions and ghost memory."""
    
    def __init__(self, samples: List[TrainingSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "page_tokens": sample.page_tokens,
            "positive_mutation": sample.positive_mutation,
            "negative_mutation": sample.negative_mutation,
            "positive_reward": torch.tensor(sample.positive_reward, dtype=torch.float32),
            "negative_reward": torch.tensor(sample.negative_reward, dtype=torch.float32),
            "penalty_weight": torch.tensor(sample.penalty_weight, dtype=torch.float32)
        }


# =============================================================================
# TRAINING LOOP
# =============================================================================

class LAMTrainer:
    """
    Trainer for the LAM model.
    
    Implements:
    - Triplet loss with inversion penalty
    - Ghost memory integration
    - Global knowledge mesh sync
    """
    
    def __init__(
        self,
        model: LAMModel,
        redis_client: redis.Redis,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.redis = redis_client
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = TripletInversionLoss(margin=0.5)
        self.inversion_math = InversionMath()
        self.ghost_memory = GhostMemory()
        
        self.step_count = 0
        self.metrics_history: List[Dict] = []
    
    async def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        page_tokens = batch["page_tokens"].to(self.device)
        pos_mut = batch["positive_mutation"].to(self.device)
        neg_mut = batch["negative_mutation"].to(self.device)
        pos_actual = batch["positive_reward"].to(self.device)
        neg_actual = batch["negative_reward"].to(self.device)
        penalty = batch["penalty_weight"].to(self.device)
        
        # Forward pass
        page_features = self.model.encode_page(page_tokens)
        pos_pred = self.model.reward_head(page_features, pos_mut)
        neg_pred = self.model.reward_head(page_features, neg_mut)
        
        # Compute loss
        loss, metrics = self.loss_fn(
            pos_pred, neg_pred,
            pos_actual, neg_actual,
            penalty
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.step_count += 1
        self.metrics_history.append(metrics)
        
        # Log to Redis
        await self.redis.xadd(
            "lam_training_metrics",
            {
                "step": self.step_count,
                "loss": metrics["total"],
                "structural_loss": metrics["structural"],
                "inversion_loss": metrics["inversion"],
                "avg_penalty": metrics["avg_penalty"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            maxlen=10000
        )
        
        return metrics
    
    async def process_inversion_queue(self) -> int:
        """Process pending inversion packets from Redis."""
        processed = 0
        
        while True:
            # Pop from inversion queue
            result = await self.redis.blpop("inversion_queue", timeout=1)
            if not result:
                break
            
            _, data = result
            packet = InversionPacket.from_dict(json.loads(data))
            
            # Create training sample from inversion
            sample = await self._inversion_to_sample(packet)
            if sample:
                # Add to training batch
                await self.redis.rpush(
                    "training_samples",
                    json.dumps(self._sample_to_dict(sample))
                )
                processed += 1
        
        return processed
    
    async def _inversion_to_sample(self, packet: InversionPacket) -> Optional[TrainingSample]:
        """Convert inversion packet to training sample."""
        # Get a positive example from ghost memory
        positive_ghosts = self.ghost_memory.get_positive_ghosts(packet.site_id)
        if not positive_ghosts:
            return None
        
        # Use best positive ghost
        best_positive = max(positive_ghosts, key=lambda g: g["final_reward"])
        
        # Get site vertical for metadata
        vertical = await self.redis.hget(f"site:{packet.site_id}", "vertical")
        vertical = vertical.decode() if vertical else "unknown"
        
        return TrainingSample(
            page_tokens=packet.base_state,
            positive_mutation=torch.tensor(best_positive["mutation_vector"]),
            negative_mutation=packet.failed_mutation,
            positive_reward=best_positive["final_reward"],
            negative_reward=packet.actual_reward,
            penalty_weight=packet.penalty_weight,
            site_id=packet.site_id,
            vertical=vertical
        )
    
    def _sample_to_dict(self, sample: TrainingSample) -> dict:
        """Convert sample to JSON-serializable dict."""
        return {
            "page_tokens": sample.page_tokens.tolist(),
            "positive_mutation": sample.positive_mutation.tolist(),
            "negative_mutation": sample.negative_mutation.tolist(),
            "positive_reward": sample.positive_reward,
            "negative_reward": sample.negative_reward,
            "penalty_weight": sample.penalty_weight,
            "site_id": sample.site_id,
            "vertical": sample.vertical
        }
    
    async def run_training_loop(self):
        """Main training loop."""
        logger.info(f"Starting training loop on {self.device}")
        
        while True:
            # 1. Process new inversions
            inversions_processed = await self.process_inversion_queue()
            if inversions_processed > 0:
                logger.info(f"Processed {inversions_processed} inversions")
            
            # 2. Check if we have enough samples for a batch
            sample_count = await self.redis.llen("training_samples")
            
            if sample_count >= BATCH_SIZE:
                # Load batch
                batch_data = []
                for _ in range(BATCH_SIZE):
                    data = await self.redis.lpop("training_samples")
                    if data:
                        batch_data.append(json.loads(data))
                
                if batch_data:
                    # Convert to tensors
                    batch = self._collate_batch(batch_data)
                    
                    # Train step
                    metrics = await self.train_step(batch)
                    
                    logger.info(
                        f"Step {self.step_count}: "
                        f"loss={metrics['total']:.4f}, "
                        f"struct={metrics['structural']:.4f}, "
                        f"inv={metrics['inversion']:.4f}"
                    )
                    
                    # Checkpoint
                    if self.step_count % CHECKPOINT_INTERVAL == 0:
                        await self.save_checkpoint()
            
            # Small sleep to prevent busy loop
            await asyncio.sleep(0.1)
    
    def _collate_batch(self, batch_data: List[dict]) -> Dict[str, torch.Tensor]:
        """Collate batch data into tensors."""
        return {
            "page_tokens": torch.stack([
                torch.tensor(d["page_tokens"], dtype=torch.long)
                for d in batch_data
            ]),
            "positive_mutation": torch.stack([
                torch.tensor(d["positive_mutation"], dtype=torch.float32)
                for d in batch_data
            ]),
            "negative_mutation": torch.stack([
                torch.tensor(d["negative_mutation"], dtype=torch.float32)
                for d in batch_data
            ]),
            "positive_reward": torch.tensor(
                [d["positive_reward"] for d in batch_data],
                dtype=torch.float32
            ),
            "negative_reward": torch.tensor(
                [d["negative_reward"] for d in batch_data],
                dtype=torch.float32
            ),
            "penalty_weight": torch.tensor(
                [d["penalty_weight"] for d in batch_data],
                dtype=torch.float32
            )
        }
    
    async def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_path = MODEL_DIR / f"checkpoint_{self.step_count}.pt"
        torch.save({
            "step": self.step_count,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history[-1000:]  # Last 1000 steps
        }, checkpoint_path)
        
        # Update latest link
        latest_path = MODEL_DIR / "latest.pt"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_path.name)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Notify via Redis
        await self.redis.publish("lam_checkpoints", json.dumps({
            "step": self.step_count,
            "path": str(checkpoint_path),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
    
    async def load_checkpoint(self, path: Optional[Path] = None):
        """Load model checkpoint."""
        if path is None:
            path = MODEL_DIR / "latest.pt"
        
        if not path.exists():
            logger.info("No checkpoint found, starting fresh")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step"]
        self.metrics_history = checkpoint.get("metrics_history", [])
        
        logger.info(f"Loaded checkpoint from step {self.step_count}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    # Initialize
    redis_client = redis.from_url(REDIS_URL)
    model = LAMModel()
    trainer = LAMTrainer(model, redis_client)
    
    # Load existing checkpoint
    await trainer.load_checkpoint()
    
    # Start training loop
    await trainer.run_training_loop()


if __name__ == "__main__":
    asyncio.run(main())
