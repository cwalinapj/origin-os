#!/usr/bin/env python3
"""
PER-VERTICAL LAM HEADS
======================

Ecommerce ≠ B2B ≠ SaaS

This prevents cross-domain poisoning.

Architecture:
- Shared Encoder: Universal patterns (visual grammar, behavioral physics)
- Per-Vertical Heads: Domain-specific policies

The encoder learns what's universal.
The heads learn what's vertical-specific.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Shared encoder for universal pattern learning.
    
    Learns:
    - Visual grammar (F-pattern, contrast, hierarchy)
    - Behavioral physics (scroll, dwell, intent)
    """
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
    
    def forward(self, x):
        return self.net(x)


class PolicyHead(nn.Module):
    """
    Vertical-specific policy head.
    
    Outputs:
    - direction: 4D mutation direction vector (tanh bounded)
    - value: Predicted reward scalar
    """
    
    def __init__(self):
        super().__init__()
        self.direction = nn.Linear(256, 4)
        self.value = nn.Linear(256, 1)
    
    def forward(self, h):
        return torch.tanh(self.direction(h)), self.value(h)


class LAM(nn.Module):
    """
    Large Action Model with per-vertical specialization.
    
    Shared encoder + vertical-specific policy heads.
    Prevents cross-domain poisoning while enabling knowledge transfer.
    """
    
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.heads = nn.ModuleDict({
            "ecommerce": PolicyHead(),
            "b2b": PolicyHead(),
            "saas": PolicyHead(),
            "default": PolicyHead()
        })
    
    def forward(self, x, vertical):
        """
        Forward pass through vertical-specific head.
        
        Args:
            x: Input features [batch, 128]
            vertical: Vertical name (ecommerce, b2b, saas)
        
        Returns:
            (direction, value) tuple
        """
        h = self.encoder(x)
        head = self.heads.get(vertical, self.heads["default"])
        return head(h)
