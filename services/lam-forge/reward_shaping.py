#!/usr/bin/env python3
"""
MULTI-OBJECTIVE REWARD SHAPING — Pareto-Optimal Evolution
==========================================================

Prevents the "Paperclip Maximizer" problem by teaching the LAM that
QUALITY of victory matters as much as the victory itself.

Reward Vector Components:
- Conversion (0.5): Primary engine, but not sole dictator
- Engagement (0.3): Rewards "Near Misses" and micro-wins
- Bounce Penalty (-0.2): Proxy for Google Ads Quality Score
- Brand Integrity (-0.1): Enforcement distance from brand anchor

Dynamic Weight Adaptation:
- Stage 1 (High-Signal): Prioritize conversion learning (0.7)
- Stage 2 (Failures): Balance conversion with bounce penalty
- Stage 3 (Marginal): Standard balanced weights
- Stage 4 (Conflicting): Prioritize brand safety over risky conversion

This creates "Elegant Mutations" — the subtle intersection where
high engagement meets sustainable bounce rate.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Default reward weights
DEFAULT_WEIGHTS = {
    "conversion": 0.5,
    "engagement": 0.3,
    "bounce": -0.2,
    "brand_integrity": -0.1
}

# Pareto constraints
MAX_BOUNCE_RATE = 0.7  # Above this, mutation is rejected
MIN_BRAND_SIMILARITY = 0.6  # Below this, mutation is rejected
MAX_DARK_PATTERN_SCORE = 0.3  # Above this, mutation is flagged

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reward-shaping")

# =============================================================================
# ENUMS & DATA MODELS
# =============================================================================

class RewardObjective(Enum):
    CONVERSION = "conversion"
    ENGAGEMENT = "engagement"
    BOUNCE = "bounce"
    BRAND_INTEGRITY = "brand_integrity"


class DarkPatternType(Enum):
    HIDDEN_CLOSE = "hidden_close"
    FAKE_COUNTDOWN = "fake_countdown"
    MISLEADING_COPY = "misleading_copy"
    FORCED_ACTION = "forced_action"
    CONFIRM_SHAMING = "confirm_shaming"
    BAIT_SWITCH = "bait_switch"


@dataclass
class RewardVector:
    """Multi-dimensional reward representation."""
    conversion_delta: float
    engagement_delta: float  # Mean behavioral score change
    bounce_delta: float
    brand_drift: float  # Semantic distance from brand anchor
    
    # Computed
    shaped_reward: float = 0.0
    pareto_optimal: bool = True
    dark_pattern_flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "conversion_delta": self.conversion_delta,
            "engagement_delta": self.engagement_delta,
            "bounce_delta": self.bounce_delta,
            "brand_drift": self.brand_drift,
            "shaped_reward": self.shaped_reward,
            "pareto_optimal": self.pareto_optimal,
            "dark_pattern_flags": self.dark_pattern_flags
        }
    
    def to_radar_data(self) -> Dict[str, float]:
        """Convert to radar chart format (0-1 scale)."""
        return {
            "Conversion": max(0, min(1, (self.conversion_delta + 0.5) / 1.0)),
            "Engagement": max(0, min(1, (self.engagement_delta + 0.5) / 1.0)),
            "Low Bounce": max(0, min(1, 1 - self.bounce_delta)),
            "Brand Fit": max(0, min(1, 1 - self.brand_drift))
        }


@dataclass
class DynamicWeights:
    """Stage-adaptive reward weights."""
    conversion: float
    engagement: float
    bounce: float
    brand_integrity: float
    stage: int
    
    def to_dict(self) -> dict:
        return {
            "conversion": self.conversion,
            "engagement": self.engagement,
            "bounce": self.bounce,
            "brand_integrity": self.brand_integrity,
            "stage": self.stage
        }


@dataclass
class ParetoFrontier:
    """Represents the Pareto-optimal solutions."""
    solutions: List[RewardVector]
    dominated_count: int
    frontier_size: int
    
    def to_dict(self) -> dict:
        return {
            "solutions": [s.to_dict() for s in self.solutions],
            "dominated_count": self.dominated_count,
            "frontier_size": self.frontier_size
        }


# =============================================================================
# DYNAMIC WEIGHT CALCULATOR
# =============================================================================

class DynamicWeightCalculator:
    """
    Adjusts reward weights based on curriculum stage.
    
    Early stages: Higher conversion weight for signal discovery
    Late stages: Tighter brand constraints for sustainable optimization
    """
    
    # Stage-specific weight configurations
    STAGE_WEIGHTS = {
        1: {  # High-Signal Wins: Prioritize conversion learning
            "conversion": 0.7,
            "engagement": 0.2,
            "bounce": -0.1,
            "brand_integrity": 0.0  # Relax brand constraints initially
        },
        2: {  # Clear Failures: Learn what NOT to do
            "conversion": 0.5,
            "engagement": 0.3,
            "bounce": -0.2,
            "brand_integrity": -0.1
        },
        3: {  # Marginal Cases: Standard balanced optimization
            "conversion": 0.5,
            "engagement": 0.3,
            "bounce": -0.2,
            "brand_integrity": -0.1
        },
        4: {  # Conflicting Signals: Prioritize brand safety
            "conversion": 0.3,
            "engagement": 0.4,
            "bounce": -0.1,
            "brand_integrity": -0.2  # Tighten brand constraints
        }
    }
    
    def __init__(self):
        self.vertical_adjustments: Dict[str, Dict[str, float]] = {}
    
    def get_weights(self, stage: int, vertical: Optional[str] = None) -> DynamicWeights:
        """Get weights for a specific stage and optional vertical."""
        base_weights = self.STAGE_WEIGHTS.get(stage, self.STAGE_WEIGHTS[3])
        
        # Apply vertical-specific adjustments if available
        if vertical and vertical in self.vertical_adjustments:
            adjustments = self.vertical_adjustments[vertical]
            weights = {
                k: base_weights[k] + adjustments.get(k, 0)
                for k in base_weights
            }
        else:
            weights = base_weights.copy()
        
        return DynamicWeights(
            conversion=weights["conversion"],
            engagement=weights["engagement"],
            bounce=weights["bounce"],
            brand_integrity=weights["brand_integrity"],
            stage=stage
        )
    
    def set_vertical_adjustment(
        self,
        vertical: str,
        adjustments: Dict[str, float]
    ):
        """Set vertical-specific weight adjustments."""
        self.vertical_adjustments[vertical] = adjustments
    
    def compute_maturity_modifier(
        self,
        sample_count: int,
        conversion_rate: float
    ) -> Dict[str, float]:
        """
        Compute weight modifiers based on site maturity.
        
        New sites: More exploration (higher engagement weight)
        Mature sites: More exploitation (higher conversion weight)
        """
        maturity = min(sample_count / 1000, 1.0)
        
        # As site matures, shift from exploration to exploitation
        conversion_boost = maturity * 0.1
        engagement_reduction = maturity * 0.05
        
        return {
            "conversion": conversion_boost,
            "engagement": -engagement_reduction,
            "bounce": 0,
            "brand_integrity": -maturity * 0.05  # Tighter constraints as we mature
        }


# =============================================================================
# DARK PATTERN DETECTOR
# =============================================================================

class DarkPatternDetector:
    """
    Detects potential dark patterns in mutations.
    
    Dark patterns destroy:
    1. Google Ads Account Health
    2. Brand Equity
    3. Long-term customer relationships
    """
    
    # Keywords associated with dark patterns
    URGENCY_KEYWORDS = [
        "hurry", "limited", "only", "last chance", "expires",
        "running out", "don't miss", "act now", "today only"
    ]
    
    SHAME_KEYWORDS = [
        "no thanks, i don't want", "i'll stay", "i prefer to",
        "no, i don't like", "i hate saving"
    ]
    
    FAKE_SCARCITY_PATTERNS = [
        r"\d+ (people|users|customers) (viewing|watching|buying)",
        r"only \d+ left",
        r"\d+:\d+:\d+ remaining"
    ]
    
    def __init__(self):
        import re
        self.scarcity_patterns = [re.compile(p) for p in self.FAKE_SCARCITY_PATTERNS]
    
    def detect(self, mutation: Dict[str, Any]) -> Tuple[float, List[DarkPatternType]]:
        """
        Detect dark patterns in a mutation.
        
        Returns:
            (dark_pattern_score, list of detected patterns)
        """
        score = 0.0
        detected = []
        
        copy_changes = mutation.get("copy_changes", [])
        style_changes = mutation.get("style_changes", [])
        layout_changes = mutation.get("layout_changes", [])
        
        # Check copy for dark patterns
        for change in copy_changes:
            new_text = change.get("new_text", "").lower()
            
            # Excessive urgency
            urgency_count = sum(1 for kw in self.URGENCY_KEYWORDS if kw in new_text)
            if urgency_count >= 3:
                score += 0.2
                detected.append(DarkPatternType.FAKE_COUNTDOWN)
            
            # Confirm shaming
            if any(kw in new_text for kw in self.SHAME_KEYWORDS):
                score += 0.3
                detected.append(DarkPatternType.CONFIRM_SHAMING)
            
            # Fake scarcity
            for pattern in self.scarcity_patterns:
                if pattern.search(new_text):
                    score += 0.2
                    detected.append(DarkPatternType.MISLEADING_COPY)
                    break
        
        # Check styles for hidden elements
        for change in style_changes:
            prop = change.get("property", "")
            new_val = change.get("new_value", "")
            
            # Hidden close button
            if prop == "opacity" and float(new_val or 1) < 0.3:
                score += 0.4
                detected.append(DarkPatternType.HIDDEN_CLOSE)
            
            # Tiny clickable areas
            if prop == "font-size" and self._parse_size(new_val) < 8:
                score += 0.2
                detected.append(DarkPatternType.HIDDEN_CLOSE)
        
        # Check layout for forced actions
        for change in layout_changes:
            element_id = change.get("element_id", "")
            new_pos = change.get("new_pos", {})
            
            # Close button moved off-screen
            if "close" in element_id.lower() or "dismiss" in element_id.lower():
                x = new_pos.get("x", 0)
                y = new_pos.get("y", 0)
                if x < -100 or y < -100 or x > 2000 or y > 1200:
                    score += 0.5
                    detected.append(DarkPatternType.HIDDEN_CLOSE)
        
        return min(score, 1.0), detected
    
    def _parse_size(self, size_str: str) -> float:
        """Parse size string to numeric value."""
        import re
        match = re.search(r"(\d+)", str(size_str))
        return float(match.group(1)) if match else 12


# =============================================================================
# PARETO OPTIMIZER
# =============================================================================

class ParetoOptimizer:
    """
    Implements Pareto-optimal selection for multi-objective optimization.
    
    A solution is Pareto-optimal if no other solution is better
    in ALL objectives simultaneously.
    """
    
    def __init__(self):
        self.objectives = [
            RewardObjective.CONVERSION,
            RewardObjective.ENGAGEMENT,
            RewardObjective.BOUNCE,
            RewardObjective.BRAND_INTEGRITY
        ]
    
    def is_dominated(self, a: RewardVector, b: RewardVector) -> bool:
        """
        Check if solution 'a' is dominated by solution 'b'.
        
        'a' is dominated if 'b' is better or equal in all objectives
        and strictly better in at least one.
        """
        # Get objective values (higher is better, except bounce and brand_drift)
        a_vals = [
            a.conversion_delta,
            a.engagement_delta,
            -a.bounce_delta,  # Negate: lower bounce is better
            -a.brand_drift    # Negate: lower drift is better
        ]
        b_vals = [
            b.conversion_delta,
            b.engagement_delta,
            -b.bounce_delta,
            -b.brand_drift
        ]
        
        # Check if b dominates a
        all_better_or_equal = all(bv >= av for av, bv in zip(a_vals, b_vals))
        at_least_one_better = any(bv > av for av, bv in zip(a_vals, b_vals))
        
        return all_better_or_equal and at_least_one_better
    
    def compute_frontier(self, solutions: List[RewardVector]) -> ParetoFrontier:
        """
        Compute the Pareto frontier from a set of solutions.
        
        Returns only non-dominated solutions.
        """
        frontier = []
        dominated_count = 0
        
        for i, sol_a in enumerate(solutions):
            is_on_frontier = True
            
            for j, sol_b in enumerate(solutions):
                if i != j and self.is_dominated(sol_a, sol_b):
                    is_on_frontier = False
                    dominated_count += 1
                    break
            
            if is_on_frontier:
                sol_a.pareto_optimal = True
                frontier.append(sol_a)
            else:
                sol_a.pareto_optimal = False
        
        return ParetoFrontier(
            solutions=frontier,
            dominated_count=dominated_count,
            frontier_size=len(frontier)
        )
    
    def select_best(
        self,
        frontier: ParetoFrontier,
        weights: DynamicWeights
    ) -> RewardVector:
        """
        Select the best solution from the Pareto frontier
        using the current weight configuration.
        """
        best_solution = None
        best_score = float("-inf")
        
        for solution in frontier.solutions:
            score = self._compute_weighted_score(solution, weights)
            if score > best_score:
                best_score = score
                best_solution = solution
        
        return best_solution
    
    def _compute_weighted_score(
        self,
        solution: RewardVector,
        weights: DynamicWeights
    ) -> float:
        """Compute weighted score for a solution."""
        return (
            weights.conversion * solution.conversion_delta +
            weights.engagement * solution.engagement_delta +
            weights.bounce * max(solution.bounce_delta, 0) +
            weights.brand_integrity * solution.brand_drift
        )


# =============================================================================
# REWARD SHAPER
# =============================================================================

class RewardShaper:
    """
    Main reward shaping engine.
    
    Combines:
    - Dynamic weight calculation
    - Dark pattern detection
    - Pareto optimization
    - Constraint enforcement
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.weight_calculator = DynamicWeightCalculator()
        self.dark_pattern_detector = DarkPatternDetector()
        self.pareto_optimizer = ParetoOptimizer()
    
    def shape_reward(
        self,
        sample: Dict[str, Any],
        stage: int = 2,
        vertical: Optional[str] = None
    ) -> RewardVector:
        """
        Shape the reward for a sample based on multiple objectives.
        
        Returns a RewardVector with shaped reward and Pareto status.
        """
        # Extract outcome metrics
        outcome = sample.get("outcome_after", {})
        conversion_delta = outcome.get("conversion_delta", 0)
        engagement_delta = outcome.get("mean_score_delta", 0)
        bounce_delta = outcome.get("bounce_delta", 0)
        
        # Get brand drift from enforcement distance
        enforcement = sample.get("enforcement_distance", {})
        brand_drift = enforcement.get("semantic_drift", 0)
        
        # Get dynamic weights
        weights = self.weight_calculator.get_weights(stage, vertical)
        
        # Compute shaped reward
        shaped = (
            weights.conversion * conversion_delta +
            weights.engagement * engagement_delta +
            weights.bounce * max(bounce_delta, 0) +
            weights.brand_integrity * brand_drift
        )
        
        # Detect dark patterns if mutation data available
        dark_score, dark_patterns = 0.0, []
        if "mutation" in sample:
            dark_score, dark_patterns = self.dark_pattern_detector.detect(
                sample["mutation"]
            )
        
        # Check Pareto constraints
        pareto_optimal = self._check_constraints(
            bounce_delta, brand_drift, dark_score
        )
        
        # Penalize dark patterns
        if dark_score > 0:
            shaped -= dark_score * 0.5  # Heavy penalty
        
        return RewardVector(
            conversion_delta=conversion_delta,
            engagement_delta=engagement_delta,
            bounce_delta=bounce_delta,
            brand_drift=brand_drift,
            shaped_reward=shaped,
            pareto_optimal=pareto_optimal,
            dark_pattern_flags=[p.value for p in dark_patterns]
        )
    
    def _check_constraints(
        self,
        bounce_delta: float,
        brand_drift: float,
        dark_score: float
    ) -> bool:
        """Check if solution meets Pareto constraints."""
        # Reject if bounce too high
        if bounce_delta > MAX_BOUNCE_RATE:
            return False
        
        # Reject if too far from brand
        if brand_drift > (1 - MIN_BRAND_SIMILARITY):
            return False
        
        # Reject if dark patterns detected
        if dark_score > MAX_DARK_PATTERN_SCORE:
            return False
        
        return True
    
    async def compute_batch_rewards(
        self,
        samples: List[Dict],
        stage: int,
        vertical: Optional[str] = None
    ) -> Tuple[List[RewardVector], ParetoFrontier]:
        """
        Compute rewards for a batch and find Pareto frontier.
        """
        rewards = [
            self.shape_reward(s, stage, vertical)
            for s in samples
        ]
        
        frontier = self.pareto_optimizer.compute_frontier(rewards)
        
        return rewards, frontier
    
    async def get_optimal_direction(
        self,
        candidates: List[Dict],
        stage: int,
        vertical: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        From a set of mutation candidates, find the Pareto-optimal one.
        """
        rewards, frontier = await self.compute_batch_rewards(
            candidates, stage, vertical
        )
        
        if not frontier.solutions:
            return {"status": "no_valid_solutions"}
        
        weights = self.weight_calculator.get_weights(stage, vertical)
        best = self.pareto_optimizer.select_best(frontier, weights)
        
        # Find corresponding candidate
        best_idx = rewards.index(best)
        
        return {
            "status": "success",
            "selected_candidate": candidates[best_idx],
            "reward_vector": best.to_dict(),
            "radar_data": best.to_radar_data(),
            "frontier_size": frontier.frontier_size,
            "weights_used": weights.to_dict()
        }
    
    async def log_reward_event(
        self,
        sample_id: str,
        reward: RewardVector,
        stage: int
    ):
        """Log reward shaping event for analysis."""
        await self.redis.xadd(
            "reward_events",
            {
                "sample_id": sample_id,
                "stage": stage,
                "shaped_reward": reward.shaped_reward,
                "conversion": reward.conversion_delta,
                "engagement": reward.engagement_delta,
                "bounce": reward.bounce_delta,
                "brand_drift": reward.brand_drift,
                "pareto_optimal": str(reward.pareto_optimal),
                "dark_patterns": json.dumps(reward.dark_pattern_flags),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            maxlen=10000
        )


# =============================================================================
# RADAR CHART GENERATOR
# =============================================================================

class RadarChartGenerator:
    """
    Generates radar chart data for visualizing multi-objective trade-offs.
    
    Used in the Offline Evaluation Harness to detect degenerate models.
    """
    
    AXES = ["Conversion", "Engagement", "Low Bounce", "Brand Fit"]
    
    def generate_chart_data(
        self,
        rewards: List[RewardVector]
    ) -> Dict[str, Any]:
        """Generate radar chart data for a set of rewards."""
        if not rewards:
            return {"axes": self.AXES, "data": []}
        
        # Compute average values per axis
        avg_values = {axis: 0.0 for axis in self.AXES}
        
        for reward in rewards:
            radar = reward.to_radar_data()
            for axis in self.AXES:
                avg_values[axis] += radar[axis]
        
        for axis in self.AXES:
            avg_values[axis] /= len(rewards)
        
        # Compute variance for each axis
        variance = {axis: 0.0 for axis in self.AXES}
        for reward in rewards:
            radar = reward.to_radar_data()
            for axis in self.AXES:
                variance[axis] += (radar[axis] - avg_values[axis]) ** 2
        
        for axis in self.AXES:
            variance[axis] = (variance[axis] / len(rewards)) ** 0.5
        
        return {
            "axes": self.AXES,
            "average": avg_values,
            "variance": variance,
            "sample_count": len(rewards)
        }
    
    def detect_degeneracy(
        self,
        rewards: List[RewardVector],
        threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Detect if the model is becoming degenerate.
        
        A model is degenerate if it optimizes one axis at the
        total expense of another.
        """
        chart_data = self.generate_chart_data(rewards)
        avg = chart_data["average"]
        
        # Check for extreme imbalance
        max_axis = max(avg.values())
        min_axis = min(avg.values())
        
        is_degenerate = (max_axis - min_axis) > 0.6
        
        weak_axes = [axis for axis, val in avg.items() if val < threshold]
        strong_axes = [axis for axis, val in avg.items() if val > 0.8]
        
        return {
            "is_degenerate": is_degenerate,
            "balance_score": 1 - (max_axis - min_axis),
            "weak_axes": weak_axes,
            "strong_axes": strong_axes,
            "recommendation": self._get_recommendation(weak_axes, strong_axes)
        }
    
    def _get_recommendation(
        self,
        weak_axes: List[str],
        strong_axes: List[str]
    ) -> str:
        """Generate recommendation based on axis analysis."""
        if not weak_axes:
            return "Model is well-balanced across all objectives."
        
        if "Brand Fit" in weak_axes:
            return "Increase brand_integrity weight to prevent semantic drift."
        
        if "Low Bounce" in weak_axes:
            return "Increase bounce penalty to improve page relevance."
        
        if "Engagement" in weak_axes:
            return "Increase engagement weight to capture micro-wins."
        
        return f"Rebalance weights: weak in {weak_axes}, strong in {strong_axes}"


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Multi-Objective Reward Shaping",
    description="Pareto-optimal reward shaping for LAM",
    version="1.0.0"
)

redis_client: Optional[redis.Redis] = None
shaper: Optional[RewardShaper] = None
radar_gen: Optional[RadarChartGenerator] = None


class ShapeRewardRequest(BaseModel):
    outcome_after: Dict[str, float]
    enforcement_distance: Dict[str, float] = {}
    mutation: Dict[str, Any] = {}
    stage: int = 2
    vertical: Optional[str] = None


class OptimalDirectionRequest(BaseModel):
    candidates: List[Dict[str, Any]]
    stage: int = 2
    vertical: Optional[str] = None


@app.on_event("startup")
async def startup():
    global redis_client, shaper, radar_gen
    redis_client = redis.from_url(REDIS_URL)
    shaper = RewardShaper(redis_client)
    radar_gen = RadarChartGenerator()


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/shape")
async def shape_reward(request: ShapeRewardRequest):
    """Shape reward for a single sample."""
    sample = {
        "outcome_after": request.outcome_after,
        "enforcement_distance": request.enforcement_distance,
        "mutation": request.mutation
    }
    
    reward = shaper.shape_reward(sample, request.stage, request.vertical)
    
    return {
        "reward_vector": reward.to_dict(),
        "radar_data": reward.to_radar_data()
    }


@app.post("/optimal")
async def get_optimal_direction(request: OptimalDirectionRequest):
    """Find Pareto-optimal mutation from candidates."""
    result = await shaper.get_optimal_direction(
        request.candidates,
        request.stage,
        request.vertical
    )
    return result


@app.get("/weights/{stage}")
async def get_weights(stage: int, vertical: Optional[str] = None):
    """Get dynamic weights for a stage."""
    weights = shaper.weight_calculator.get_weights(stage, vertical)
    return weights.to_dict()


@app.post("/weights/{vertical}")
async def set_vertical_weights(vertical: str, adjustments: Dict[str, float]):
    """Set vertical-specific weight adjustments."""
    shaper.weight_calculator.set_vertical_adjustment(vertical, adjustments)
    return {"status": "updated", "vertical": vertical}


@app.post("/analyze-batch")
async def analyze_batch(samples: List[Dict], stage: int = 2):
    """Analyze a batch of samples for degeneracy."""
    rewards = [shaper.shape_reward(s, stage) for s in samples]
    
    chart_data = radar_gen.generate_chart_data(rewards)
    degeneracy = radar_gen.detect_degeneracy(rewards)
    
    return {
        "radar_chart": chart_data,
        "degeneracy_analysis": degeneracy,
        "pareto_optimal_count": sum(1 for r in rewards if r.pareto_optimal),
        "total_samples": len(rewards)
    }


@app.get("/dark-patterns")
async def get_dark_pattern_stats():
    """Get dark pattern detection statistics."""
    events = await redis_client.xrange("reward_events", count=1000)
    
    pattern_counts = {}
    total = 0
    
    for _, data in events:
        patterns = json.loads(data.get(b"dark_patterns", b"[]"))
        for p in patterns:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
            total += 1
    
    return {
        "total_detected": total,
        "by_type": pattern_counts
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
