"""LAM Analysis Module — Surprise diagnostics and maturity checks."""
from lam.analysis.surprise import (
    SurpriseAnalyzer,
    HeatmapShape,
    SurpriseOutlier,
    AutopsyResult,
    check_maturity_for_promotion
)

__all__ = [
    "SurpriseAnalyzer",
    "HeatmapShape",
    "SurpriseOutlier",
    "AutopsyResult",
    "check_maturity_for_promotion"
]
