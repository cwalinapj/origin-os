"""
LAM Forge — Large Action Model Training & Inference System
==========================================================

Complete system for training, evaluating, and deploying LAMs:

Core Components:
- inversion_math: Failure Inversion mathematics
- training_loop: PyTorch training with Ghost Memory
- inference_spawner: Container spawner for inference mode

Monitoring & Safety:
- neural_monitor: Ensemble drift detection
- knowledge_mesh: Cross-site federated learning
- diff_enforcer: Brand safety guardrails
- evaluation_harness: Calibration gatekeeper

Integration:
- gads_deployment: Google Ads API integration
- metrics: Prometheus exporters
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

__version__ = "1.1.0"
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
