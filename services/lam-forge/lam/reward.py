#!/usr/bin/env python3
"""
MULTI-OBJECTIVE REWARD SHAPING
==============================

Prevent dark-pattern wins & brand damage.
This is where you encode VALUES, not just conversions.

Components:
- Engagement (0.3): Near-miss value
- Conversion (0.5): Primary objective
- Bounce (-0.2): Quality Score proxy
- Brand Drift (-0.1): Semantic anchor enforcement
"""


def shaped_reward(sample):
    """
    Compute multi-objective shaped reward.
    
    Prevents the LAM from optimizing conversions at the expense
    of brand integrity and user experience.
    """
    outcome = sample["outcome_after"]
    enforcement = sample["enforcement_distance"]
    
    reward = 0.0
    
    # Engagement — rewards "near misses"
    reward += 0.3 * outcome.get("mean_score_delta", 0.0)
    
    # Conversion — primary objective (but not sole dictator)
    reward += 0.5 * outcome.get("conversion_delta", 0.0)
    
    # Bounce penalty — proxy for Google Ads Quality Score
    reward -= 0.2 * max(outcome.get("bounce_delta", 0.0), 0)
    
    # Brand / semantic drift penalty — enforcement distance
    reward -= 0.1 * enforcement.get("semantic_drift", 0.0)
    
    return reward
