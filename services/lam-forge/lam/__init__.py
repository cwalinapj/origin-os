"""
LAM — Large Action Model
========================

Core components:
- model: Per-vertical LAM architecture
- reward: Multi-objective reward shaping
- features: Feature extraction

Training:
- train.curriculum: Difficulty bucketing
- train.train: Curriculum-based training

Evaluation:
- eval.replay: Counterfactual replay harness
"""

from lam.model import LAM, Encoder, PolicyHead
from lam.reward import shaped_reward
from lam.features import featurize

__all__ = [
    "LAM",
    "Encoder", 
    "PolicyHead",
    "shaped_reward",
    "featurize"
]
