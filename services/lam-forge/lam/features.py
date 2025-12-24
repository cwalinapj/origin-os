#!/usr/bin/env python3
"""
FEATURE EXTRACTION
==================

Converts raw samples into model-ready tensors.

Feature groups:
- Context: Time, traffic entropy
- Behavior: Score, bounce, scroll, dwell
- Lineage: Ghost weight, parent lifetime
- Enforcement: DOM complexity, semantic drift, visual shift
"""

import torch


def featurize(sample):
    """
    Convert sample to feature tensor.
    
    Returns:
        torch.Tensor of shape [11] (expandable to 128 via model projection)
    """
    ctx = sample["context_features"]
    beh = sample["behavior_before"]
    lin = sample["lineage"]
    enf = sample["enforcement_distance"]
    
    features = [
        # Context features
        ctx.get("hour_of_day", 0) / 24.0,
        ctx.get("traffic_entropy", 0.0),
        
        # Behavioral features (pre-mutation)
        beh.get("mean_score", 0.0),
        beh.get("bounce_rate", 0.0),
        beh.get("avg_scroll", 0.0),
        beh.get("avg_dwell_time", 0.0) / 60.0,
        
        # Lineage features
        lin.get("ghost_weight", 0.0),
        lin.get("parent_lifetime_events", 0) / 1000.0,
        
        # Enforcement features
        enf.get("dom_complexity", 0.0),
        enf.get("semantic_drift", 0.0),
        enf.get("visual_shift", 0.0),
    ]
    
    return torch.tensor(features, dtype=torch.float32)
