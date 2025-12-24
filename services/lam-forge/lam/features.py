#!/usr/bin/env python3
"""
FEATURE EXTRACTION
==================

Converts raw samples into model-ready tensors.
"""

import torch
import numpy as np


def featurize(sample):
    """
    Convert sample to feature tensor.
    
    Extracts:
    - Base state embedding
    - Mutation vector
    - Context signals
    
    Returns:
        torch.Tensor of shape [128]
    """
    # Get base state (page embedding)
    base_state = sample.get("base_state", [0] * 64)
    
    # Get mutation vector
    mutation = sample.get("mutation_vector", [0] * 32)
    
    # Get context signals
    context = sample.get("context", {})
    context_features = [
        context.get("time_of_day", 0.5),
        context.get("day_of_week", 0.5),
        context.get("device_mobile", 0.0),
        context.get("traffic_source_paid", 0.0),
        # Pad to 32
    ]
    context_features = context_features + [0] * (32 - len(context_features))
    
    # Concatenate all features
    features = base_state[:64] + mutation[:32] + context_features[:32]
    
    # Ensure 128 dimensions
    features = features[:128] + [0] * max(0, 128 - len(features))
    
    return torch.tensor(features, dtype=torch.float32)
