#!/usr/bin/env python3
"""
LAM FORGE — Inversion Vector Mathematics
=========================================

The core insight: When the model is confidently wrong, the error vector
contains more information than a typical gradient update. We extract
and amplify this signal.

Key Equations:

1. Surprise Vector (δ_surprise):
   δ = R_predicted - R_actual
   
2. Surprise Significance (σ_sig):
   σ_sig = δ / σ_baseline(vertical)
   
   When σ_sig > 2.0, we have a statistically significant surprise.

3. Inversion Direction:
   - Structural: V_inv = -V_failed (geometric flip)
   - Behavioral: V_inv = project(V_anchor - V_failed, S_safe)
   
4. Penalty Weight Decay:
   W(t) = W_max · e^(-λt)
   
   Where t = samples_in_vertical / 1000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np
from datetime import datetime, timezone
import json


@dataclass
class InversionPacket:
    """Captures a confident failure for LAM training."""
    site_id: str
    base_state: torch.Tensor        # S_base: HTML embedding
    failed_mutation: torch.Tensor   # V_failed: The direction we went
    predicted_reward: float         # What we expected
    actual_reward: float            # What we got
    surprise_magnitude: float       # |predicted - actual|
    error_type: str                 # "Structural" or "Behavioral"
    penalty_weight: float           # Amplification factor
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "base_state": self.base_state.tolist() if isinstance(self.base_state, torch.Tensor) else self.base_state,
            "failed_mutation": self.failed_mutation.tolist() if isinstance(self.failed_mutation, torch.Tensor) else self.failed_mutation,
            "predicted_reward": self.predicted_reward,
            "actual_reward": self.actual_reward,
            "surprise_magnitude": self.surprise_magnitude,
            "error_type": self.error_type,
            "penalty_weight": self.penalty_weight,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "InversionPacket":
        return cls(
            site_id=data["site_id"],
            base_state=torch.tensor(data["base_state"]),
            failed_mutation=torch.tensor(data["failed_mutation"]),
            predicted_reward=data["predicted_reward"],
            actual_reward=data["actual_reward"],
            surprise_magnitude=data["surprise_magnitude"],
            error_type=data["error_type"],
            penalty_weight=data["penalty_weight"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {})
        )


class InversionMath:
    """
    Mathematical framework for Failure Inversion.
    
    This class implements the core algorithms for detecting confident failures
    and computing the inversion vectors used to correct the LAM's behavior.
    """
    
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.margin = config.get("triplet_margin", 0.5)
        self.w_max = config.get("max_penalty_weight", 5.0)
        self.lambda_decay = config.get("penalty_decay_rate", 0.1)
        self.surprise_threshold = config.get("surprise_threshold", 2.0)
        self.structural_dims = config.get("structural_dims", 64)  # First N dims = layout
    
    def compute_surprise(
        self,
        predicted: float,
        actual: float,
        baseline_variance: float
    ) -> Tuple[float, bool]:
        """
        Compute surprise magnitude and whether it triggers inversion.
        
        Args:
            predicted: The model's predicted reward
            actual: The actual observed reward
            baseline_variance: Historical variance for this vertical
        
        Returns:
            (surprise_significance, requires_inversion)
        """
        delta = predicted - actual
        
        # Normalize by baseline variance for this vertical
        # A 5% miss in fintech is different from 5% in e-commerce
        sigma_sig = abs(delta) / (baseline_variance + 1e-8)
        
        requires_inversion = sigma_sig > self.surprise_threshold
        
        return sigma_sig, requires_inversion
    
    def compute_structural_inversion(
        self,
        failed_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Structural Inversion: Direct geometric flip.
        
        If moving the CTA right by 50px failed, we penalize
        rightward CTA movement in the embedding space.
        
        V_inv = -V_failed
        """
        return -1.0 * failed_vector
    
    def compute_behavioral_inversion(
        self,
        failed_vector: torch.Tensor,
        brand_anchor: torch.Tensor,
        safe_subspace: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Behavioral Inversion: Project toward safety.
        
        We don't just flip "aggressive" to "passive" — we project
        back toward the brand's verified semantic history.
        
        V_inv = 0.5 * (V_anchor - V_failed)
        
        Then optionally project onto the safe subspace to ensure we stay
        within brand-approved semantic territory.
        
        Args:
            failed_vector: The mutation that failed
            brand_anchor: The brand's baseline semantic vector
            safe_subspace: Principal components of safe copy (optional)
        """
        # Direction toward safety
        retreat_vector = brand_anchor - failed_vector
        
        # Scale to half-step (don't overcorrect)
        retreat_vector = 0.5 * retreat_vector
        
        if safe_subspace is not None:
            # Project onto safe subspace (brand-approved semantics)
            # safe_subspace is [n_components, embedding_dim]
            projection = torch.mm(
                torch.mm(retreat_vector.unsqueeze(0), safe_subspace.T),
                safe_subspace
            ).squeeze(0)
            return projection
        
        return retreat_vector
    
    def compute_penalty_weight(
        self,
        vertical_maturity: float,  # 0.0 (new) to 1.0 (mature)
        surprise_magnitude: float
    ) -> float:
        """
        Penalty weight with decay for mature verticals.
        
        W(t) = W_max · e^(-λ · maturity) · min(surprise / threshold, 2.0)
        
        - New verticals get full penalty (learning is expensive)
        - Mature verticals get decayed penalty (model is calibrated)
        - Higher surprise = higher penalty (up to 2x cap)
        """
        maturity_decay = np.exp(-self.lambda_decay * vertical_maturity * 10)
        surprise_multiplier = min(surprise_magnitude / self.surprise_threshold, 2.0)
        
        return self.w_max * maturity_decay * surprise_multiplier
    
    def detect_error_type(self, mutation_vector: torch.Tensor) -> str:
        """
        Auto-detect error type from mutation characteristics.
        
        Structural mutations have high variance in positional dimensions.
        Behavioral mutations have high variance in semantic dimensions.
        """
        positional_variance = mutation_vector[:self.structural_dims].var().item()
        semantic_variance = mutation_vector[self.structural_dims:].var().item()
        
        return "Structural" if positional_variance > semantic_variance else "Behavioral"
    
    def build_inversion_packet(
        self,
        site_id: str,
        base_html_embedding: torch.Tensor,
        mutation_vector: torch.Tensor,
        predicted_reward: float,
        actual_reward: float,
        baseline_variance: float,
        vertical_maturity: float,
        brand_anchor: Optional[torch.Tensor] = None,
        safe_subspace: Optional[torch.Tensor] = None,
        error_type: str = "auto",
        metadata: Optional[Dict] = None
    ) -> Optional[InversionPacket]:
        """
        Build a complete inversion packet for LAM training.
        
        Returns None if the surprise doesn't meet the threshold.
        """
        surprise_mag, requires_inv = self.compute_surprise(
            predicted_reward, actual_reward, baseline_variance
        )
        
        if not requires_inv:
            return None
        
        # Auto-detect error type if needed
        if error_type == "auto":
            error_type = self.detect_error_type(mutation_vector)
        
        # Compute penalty
        penalty = self.compute_penalty_weight(vertical_maturity, surprise_mag)
        
        return InversionPacket(
            site_id=site_id,
            base_state=base_html_embedding,
            failed_mutation=mutation_vector,
            predicted_reward=predicted_reward,
            actual_reward=actual_reward,
            surprise_magnitude=surprise_mag,
            error_type=error_type,
            penalty_weight=penalty,
            metadata=metadata or {}
        )


class TripletInversionLoss(nn.Module):
    """
    Custom loss function combining:
    1. Standard reward prediction (MSE)
    2. Triplet margin loss (separate winners from failures)
    3. Inversion penalty (amplify confident failure signal)
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        pos_pred: torch.Tensor,      # Predicted reward for winning mutation
        neg_pred: torch.Tensor,      # Predicted reward for failed mutation
        pos_actual: torch.Tensor,    # Actual reward for winner
        neg_actual: torch.Tensor,    # Actual reward for failure
        penalty_weights: torch.Tensor # Per-sample penalty weights
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        L_total = L_structural + W · L_inversion
        
        Where:
        - L_structural = MSE(pos_pred, pos_actual)
        - L_inversion = ReLU(neg_pred - neg_actual + margin)
        
        The inversion loss penalizes the model for predicting high rewards
        on mutations that actually failed. The margin ensures the model
        learns to separate winners from losers by at least `margin` units.
        """
        # Standard prediction loss on winners
        structural_loss = F.mse_loss(pos_pred, pos_actual, reduction='none')
        
        # Inversion loss: Penalize overconfidence on failures
        # If neg_pred > neg_actual + margin, we were overconfident
        inversion_loss = F.relu(neg_pred - neg_actual + self.margin)
        
        # Weight inversion loss by penalty (higher for confident failures)
        weighted_inversion = inversion_loss * penalty_weights
        
        # Combine
        total_loss = structural_loss.mean() + weighted_inversion.mean()
        
        metrics = {
            "structural": structural_loss.mean().item(),
            "inversion": weighted_inversion.mean().item(),
            "avg_penalty": penalty_weights.mean().item(),
            "total": total_loss.item()
        }
        
        return total_loss, metrics


class GhostMemory:
    """
    Ghost Memory: Retain knowledge from tombstoned variants.
    
    When a variant is killed (tombstoned), we don't discard its learnings.
    Instead, we extract the "ghost" — the directional knowledge of what
    worked and what didn't — and preserve it for the LAM.
    """
    
    def __init__(self, max_ghosts: int = 1000):
        self.max_ghosts = max_ghosts
        self.ghosts: list = []
    
    def add_ghost(
        self,
        variant_id: str,
        site_id: str,
        mutation_vector: torch.Tensor,
        final_reward: float,
        samples_collected: int,
        death_reason: str  # "underperform", "converged", "timeout"
    ):
        """Add a ghost from a tombstoned variant."""
        ghost = {
            "variant_id": variant_id,
            "site_id": site_id,
            "mutation_vector": mutation_vector.tolist(),
            "final_reward": final_reward,
            "samples_collected": samples_collected,
            "death_reason": death_reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.ghosts.append(ghost)
        
        # Prune old ghosts if needed
        if len(self.ghosts) > self.max_ghosts:
            self.ghosts = self.ghosts[-self.max_ghosts:]
    
    def get_ghosts_for_site(self, site_id: str) -> list:
        """Get all ghosts for a specific site."""
        return [g for g in self.ghosts if g["site_id"] == site_id]
    
    def get_positive_ghosts(self, site_id: str, reward_threshold: float = 0.6) -> list:
        """Get ghosts that performed well (for positive examples)."""
        return [
            g for g in self.get_ghosts_for_site(site_id)
            if g["final_reward"] >= reward_threshold
        ]
    
    def get_negative_ghosts(self, site_id: str, reward_threshold: float = 0.3) -> list:
        """Get ghosts that performed poorly (for negative examples)."""
        return [
            g for g in self.get_ghosts_for_site(site_id)
            if g["final_reward"] <= reward_threshold
        ]
    
    def to_training_batch(self, site_id: str) -> Optional[Dict]:
        """
        Convert ghosts into a training batch for triplet loss.
        
        Returns pairs of (positive_ghost, negative_ghost) for contrastive learning.
        """
        positives = self.get_positive_ghosts(site_id)
        negatives = self.get_negative_ghosts(site_id)
        
        if not positives or not negatives:
            return None
        
        # Create pairs
        pairs = []
        for pos in positives:
            for neg in negatives:
                pairs.append({
                    "positive_vector": torch.tensor(pos["mutation_vector"]),
                    "negative_vector": torch.tensor(neg["mutation_vector"]),
                    "positive_reward": pos["final_reward"],
                    "negative_reward": neg["final_reward"]
                })
        
        return {
            "site_id": site_id,
            "pairs": pairs,
            "num_positives": len(positives),
            "num_negatives": len(negatives)
        }


if __name__ == "__main__":
    # Test the inversion math
    math = InversionMath()
    
    # Simulate a confident failure
    predicted = 0.75  # We expected 75% conversion lift
    actual = 0.20     # We got 20%
    baseline_var = 0.15
    
    surprise, needs_inversion = math.compute_surprise(predicted, actual, baseline_var)
    print(f"Surprise: {surprise:.2f}, Needs Inversion: {needs_inversion}")
    
    # Compute penalty
    penalty = math.compute_penalty_weight(vertical_maturity=0.3, surprise_magnitude=surprise)
    print(f"Penalty Weight: {penalty:.2f}")
    
    # Test structural inversion
    failed_vec = torch.randn(128)
    inverted = math.compute_structural_inversion(failed_vec)
    print(f"Structural Inversion: dot product = {torch.dot(failed_vec, inverted).item():.2f}")
