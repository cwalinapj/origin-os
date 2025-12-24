#!/usr/bin/env python3
"""
CURRICULUM LEARNING — Noise → Signal → Edge Cases
==================================================

LAM must learn easy truths before subtle ones.

Stage 1: Strong signal (|Δ| >= 0.15) — Clear wins/losses
Stage 2: Medium signal (|Δ| >= 0.08) — Moderate effects
Stage 3: Weak signal (|Δ| >= 0.03) — Subtle patterns
Stage 4: Noisy/ambiguous (|Δ| < 0.03) — Edge cases
"""


def difficulty_bucket(sample):
    """
    Classify sample into difficulty bucket based on signal strength.
    
    Higher delta = clearer signal = easier to learn.
    """
    delta = abs(sample["outcome_after"]["mean_score_delta"])
    
    if delta >= 0.15:
        return 1  # strong signal
    if delta >= 0.08:
        return 2  # medium signal
    if delta >= 0.03:
        return 3  # weak signal
    return 4  # noisy / ambiguous
