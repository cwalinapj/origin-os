"""
LAM Forge — Large Action Model Training & Inference System
==========================================================

Components:
- inversion_math: Failure Inversion mathematics
- inference_spawner: Container spawner for inference mode
- training_loop: PyTorch training with Ghost Memory
- gads_deployment: Google Ads API integration
- metrics: Prometheus metrics

Usage:
    from lam_forge import InversionMath, LAMModel, LAMTrainer
"""

from .inversion_math import (
    InversionMath,
    InversionPacket,
    TripletInversionLoss,
    GhostMemory
)
from .training_loop import (
    LAMModel,
    LAMTrainer,
    HTMLEncoder,
    RewardHead
)
from .inference_spawner import (
    ContainerPoolManager,
    LAMInferenceClient,
    StructureDefinition
)
from .gads_deployment import (
    GoogleAdsDeploymentService,
    SitelinkMutation,
    BehavioralScore
)

__version__ = "1.0.0"
__all__ = [
    # Inversion Math
    "InversionMath",
    "InversionPacket", 
    "TripletInversionLoss",
    "GhostMemory",
    # Model
    "LAMModel",
    "LAMTrainer",
    "HTMLEncoder",
    "RewardHead",
    # Inference
    "ContainerPoolManager",
    "LAMInferenceClient",
    "StructureDefinition",
    # Google Ads
    "GoogleAdsDeploymentService",
    "SitelinkMutation",
    "BehavioralScore",
]
