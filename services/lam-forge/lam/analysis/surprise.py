#!/usr/bin/env python3
"""
SURPRISE ANALYSIS — Counterfactual Autopsy Tools
=================================================

Tools for debugging high-surprise events and diagnosing LAM failures.

When Prometheus fires HighModelSurprise alert:
1. Query outliers in MongoDB
2. Compare with LAM mutation intent
3. Check Container Diff for LLM errors
4. Assess heatmap shape for maturity

Heatmap Shapes:
- Tight Cluster @ 0.0: High Calibration → Promote LoA
- Bimodal (Two Peaks): Hidden Variable → Segment vertical
- Wide/Flat: High Noise → Increase Stage 1 training
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np


class HeatmapShape(Enum):
    """Classification of surprise distribution shapes."""
    TIGHT_CLUSTER = "tight_cluster"      # High calibration
    BIMODAL = "bimodal"                  # Hidden variable
    WIDE_FLAT = "wide_flat"              # High noise / underfitting
    LEFT_SKEWED = "left_skewed"          # Systematic overestimation
    RIGHT_SKEWED = "right_skewed"        # Systematic underestimation


@dataclass
class SurpriseOutlier:
    """A high-surprise session for autopsy."""
    session_id: str
    site_id: str
    vertical: str
    timestamp: datetime
    predicted_reward: float
    actual_reward: float
    surprise: float
    mutation_intent: str
    tombstone_id: Optional[str]
    behavior: Dict[str, Any]


@dataclass
class AutopsyResult:
    """Result of a counterfactual autopsy."""
    session_id: str
    diagnosis: str
    root_cause: str
    recommended_action: str
    container_diff_issues: List[str]


class SurpriseAnalyzer:
    """
    Analyzer for high-surprise events.
    
    Performs counterfactual autopsies when the model's predictions
    diverge significantly from reality.
    """
    
    def __init__(self, mongo_client: AsyncIOMotorClient):
        self.db = mongo_client.origin_os
    
    async def query_outliers(
        self,
        site_id: str,
        surprise_threshold: float = 0.5,
        limit: int = 5
    ) -> List[SurpriseOutlier]:
        """
        Query high-surprise sessions from MongoDB.
        
        MongoDB Query for High-Surprise Sessions:
        db.sessions.find({
            "site_id": "site_123",
            "outcome.conversion": false,
            "behavior.cta_intent_score": { "$lt": 0.1 }
        }).sort({"timestamp": -1}).limit(5)
        """
        cursor = self.db.sessions.find({
            "site_id": site_id,
            "$or": [
                {"surprise_delta": {"$lte": -surprise_threshold}},
                {"surprise_delta": {"$gte": surprise_threshold}}
            ]
        }).sort("timestamp", -1).limit(limit)
        
        outliers = []
        async for doc in cursor:
            outliers.append(SurpriseOutlier(
                session_id=str(doc["_id"]),
                site_id=doc["site_id"],
                vertical=doc.get("vertical", "unknown"),
                timestamp=doc["timestamp"],
                predicted_reward=doc.get("predicted_reward", 0),
                actual_reward=doc.get("actual_reward", 0),
                surprise=doc.get("surprise_delta", 0),
                mutation_intent=doc.get("mutation_intent", ""),
                tombstone_id=doc.get("tombstone_id"),
                behavior=doc.get("behavior", {})
            ))
        
        return outliers
    
    async def get_tombstone(self, tombstone_id: str) -> Optional[Dict]:
        """Get tombstone record for autopsy."""
        return await self.db.tombstones.find_one({"_id": tombstone_id})
    
    async def get_container_diff(self, tombstone_id: str) -> Optional[Dict]:
        """Get container diff associated with a tombstone."""
        tombstone = await self.get_tombstone(tombstone_id)
        if not tombstone:
            return None
        
        container_id = tombstone.get("container_id")
        return await self.db.container_diffs.find_one({"container_id": container_id})
    
    async def perform_autopsy(self, outlier: SurpriseOutlier) -> AutopsyResult:
        """
        Perform counterfactual autopsy on a high-surprise event.
        
        Step 1: Get mutation intent from tombstone
        Step 2: Get container diff to check LLM execution
        Step 3: Diagnose root cause
        """
        issues = []
        diagnosis = "Unknown"
        root_cause = "Unknown"
        action = "Manual review required"
        
        # Get associated records
        tombstone = None
        container_diff = None
        
        if outlier.tombstone_id:
            tombstone = await self.get_tombstone(outlier.tombstone_id)
            container_diff = await self.get_container_diff(outlier.tombstone_id)
        
        # Analyze container diff for issues
        if container_diff:
            # Check for CTA displacement
            if container_diff.get("cta_position_changed"):
                old_pos = container_diff.get("cta_old_position", {})
                new_pos = container_diff.get("cta_new_position", {})
                if new_pos.get("y", 0) > old_pos.get("y", 0) + 200:
                    issues.append("CTA pushed below fold")
            
            # Check for layout shift
            if container_diff.get("layout_shift", 0) > 0.15:
                issues.append(f"High CLS: {container_diff['layout_shift']:.2f}")
            
            # Check for semantic drift
            if container_diff.get("semantic_drift", 0) > 0.3:
                issues.append(f"High semantic drift: {container_diff['semantic_drift']:.2f}")
        
        # Determine diagnosis based on surprise direction
        if outlier.surprise < -0.5:
            # Critical Failure: Model was too optimistic
            diagnosis = "Critical Failure"
            
            if issues:
                root_cause = f"LLM execution error: {', '.join(issues)}"
                action = "Revert mutation and add to failure inversion training"
            else:
                root_cause = "Model overestimated mutation effectiveness"
                action = "Increase penalty weight for this mutation direction"
        
        elif outlier.surprise > 0.5:
            # Exploration Jackpot: Model was too pessimistic
            diagnosis = "Exploration Jackpot"
            root_cause = "Model underestimated mutation effectiveness"
            action = "Add to Knowledge Mesh for cross-site distillation"
        
        return AutopsyResult(
            session_id=outlier.session_id,
            diagnosis=diagnosis,
            root_cause=root_cause,
            recommended_action=action,
            container_diff_issues=issues
        )
    
    async def analyze_distribution(
        self,
        vertical: str,
        window_hours: int = 24
    ) -> Tuple[HeatmapShape, Dict[str, float]]:
        """
        Analyze the surprise distribution shape for a vertical.
        
        Returns the shape classification and statistics.
        """
        since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        
        cursor = self.db.sessions.find({
            "vertical": vertical,
            "timestamp": {"$gte": since},
            "surprise_delta": {"$exists": True}
        })
        
        surprises = []
        async for doc in cursor:
            surprises.append(doc["surprise_delta"])
        
        if len(surprises) < 10:
            return HeatmapShape.WIDE_FLAT, {"samples": len(surprises)}
        
        surprises = np.array(surprises)
        
        # Calculate statistics
        mean = float(np.mean(surprises))
        std = float(np.std(surprises))
        skewness = float(self._calculate_skewness(surprises))
        kurtosis = float(self._calculate_kurtosis(surprises))
        
        # Count in zones
        calibrated = np.sum(np.abs(surprises) <= 0.1) / len(surprises)
        critical = np.sum(surprises <= -0.5) / len(surprises)
        jackpot = np.sum(surprises >= 0.5) / len(surprises)
        
        stats = {
            "samples": len(surprises),
            "mean": mean,
            "std": std,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "calibrated_ratio": calibrated,
            "critical_ratio": critical,
            "jackpot_ratio": jackpot
        }
        
        # Classify shape
        shape = self._classify_shape(stats)
        
        return shape, stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return (np.sum((data - mean) ** 3) / n) / (std ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of distribution."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return (np.sum((data - mean) ** 4) / n) / (std ** 4) - 3
    
    def _classify_shape(self, stats: Dict[str, float]) -> HeatmapShape:
        """Classify distribution shape based on statistics."""
        # Tight cluster: low std, high calibrated ratio
        if stats["std"] < 0.15 and stats["calibrated_ratio"] > 0.6:
            return HeatmapShape.TIGHT_CLUSTER
        
        # Bimodal: both critical and jackpot ratios are high
        if stats["critical_ratio"] > 0.15 and stats["jackpot_ratio"] > 0.15:
            return HeatmapShape.BIMODAL
        
        # Skewed distributions
        if stats["skewness"] < -0.5:
            return HeatmapShape.LEFT_SKEWED
        if stats["skewness"] > 0.5:
            return HeatmapShape.RIGHT_SKEWED
        
        # Wide/flat: high std, low calibrated ratio
        if stats["std"] > 0.3 or stats["calibrated_ratio"] < 0.3:
            return HeatmapShape.WIDE_FLAT
        
        return HeatmapShape.TIGHT_CLUSTER
    
    def get_recommendation(self, shape: HeatmapShape) -> str:
        """Get recommended action based on distribution shape."""
        recommendations = {
            HeatmapShape.TIGHT_CLUSTER: 
                "High calibration. Consider promoting LoA level.",
            HeatmapShape.BIMODAL: 
                "Hidden variable detected. Segment vertical further "
                "(e.g., Mobile vs. Desktop, New vs. Returning).",
            HeatmapShape.WIDE_FLAT: 
                "High noise / underfitting. Increase Curriculum Stage 1 "
                "training weight and gather more samples.",
            HeatmapShape.LEFT_SKEWED:
                "Systematic overestimation. Model is too optimistic. "
                "Increase failure inversion training.",
            HeatmapShape.RIGHT_SKEWED:
                "Systematic underestimation. Model is too conservative. "
                "Reduce exploration penalty."
        }
        return recommendations.get(shape, "Manual review required.")


# =============================================================================
# MATURITY CHECK FOR FINAL 1000
# =============================================================================

async def check_maturity_for_promotion(
    mongo_client: AsyncIOMotorClient,
    vertical: str
) -> Tuple[bool, str]:
    """
    Check if vertical is mature enough for LoA 4 promotion.
    
    Requirements:
    1. >= 1000 samples
    2. Tight cluster distribution (std < 0.15)
    3. >= 60% in calibration zone
    """
    analyzer = SurpriseAnalyzer(mongo_client)
    
    shape, stats = await analyzer.analyze_distribution(vertical, window_hours=168)  # 7 days
    
    # Check sample count
    if stats["samples"] < 1000:
        return False, f"Insufficient samples: {stats['samples']}/1000"
    
    # Check distribution shape
    if shape == HeatmapShape.WIDE_FLAT:
        return False, f"Wide distribution (std={stats['std']:.2f}). Model underfitting."
    
    if shape == HeatmapShape.BIMODAL:
        return False, "Bimodal distribution. Hidden variable needs segmentation."
    
    # Check calibration ratio
    if stats["calibrated_ratio"] < 0.5:
        return False, f"Low calibration: {stats['calibrated_ratio']:.1%} in [-0.1, 0.1]"
    
    # Check for systematic bias
    if shape in [HeatmapShape.LEFT_SKEWED, HeatmapShape.RIGHT_SKEWED]:
        return False, f"Systematic bias detected: {shape.value}"
    
    return True, f"Ready for LoA 4. Calibration: {stats['calibrated_ratio']:.1%}"
