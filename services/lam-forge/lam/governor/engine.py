#!/usr/bin/env python3
"""
GOVERNOR — Level of Autonomy (LoA) Controller
==============================================

The switching logic that determines the autonomy level for each container.
Prevents handing keys to the LAM before it proves it can outperform
the ensemble of external LLMs.

LoA Scale:
- Level 1 (Shadow): LLMs generate; LAM predicts in silence
- Level 2 (Advisory): LLMs generate; LAM vetoes/re-ranks
- Level 3 (Co-Pilot): LAM generates vector; LLM executes code
- Level 4 (Governor): LAM generates code directly; LLM audits only

Thresholds:
- LoA 1: < 100 samples
- LoA 2: > 100 samples + 70% accuracy
- LoA 3: > 500 samples + 80% accuracy
- LoA 4: > 1000 samples + 85% accuracy
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import IntEnum

import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("governor")


class LoA(IntEnum):
    """Level of Autonomy"""
    SHADOW = 1      # LLM dominant, LAM silent
    ADVISORY = 2    # LLM dominant, LAM vetoes
    COPILOT = 3     # LAM vector, LLM code
    GOVERNOR = 4    # LAM native, LLM audit only


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def get_offline_accuracy(vertical: str) -> float:
    """Get offline replay accuracy for this vertical."""
    r = redis.from_url(REDIS_URL)
    try:
        val = await r.get(f"loa_accuracy:{vertical}")
        return float(val) if val else 0.0
    finally:
        await r.close()


async def get_vertical_sample_count(vertical: str) -> int:
    """Get cumulative sample count for this vertical."""
    r = redis.from_url(REDIS_URL)
    try:
        val = await r.get(f"loa_samples:{vertical}")
        return int(val) if val else 0
    finally:
        await r.close()


# =============================================================================
# GOVERNOR ENGINE
# =============================================================================

class Governor:
    """
    The Governor sits between the Router and Docker MCP.
    Intercepts regeneration requests and decides which "Brain" to use.
    """
    
    def __init__(self, vertical: str):
        self.vertical = vertical
    
    async def determine_loa(self) -> int:
        """
        Determine the Level of Autonomy for this vertical.
        
        Based on:
        - Cumulative sample volume
        - Offline replay accuracy
        """
        accuracy = await get_offline_accuracy(self.vertical)
        sample_count = await get_vertical_sample_count(self.vertical)
        
        if sample_count > 1000 and accuracy > 0.85:
            return 4  # FULL LAM AUTHORITY
        elif sample_count > 500 and accuracy > 0.80:
            return 3  # LAM GUIDED
        elif sample_count > 100 and accuracy > 0.70:
            return 2  # LAM ADVISORY
        return 1      # LLM DOMINANT
    
    async def get_generation_strategy(self) -> str:
        """Get the generation strategy based on current LoA."""
        loa = await self.determine_loa()
        
        if loa == 4:
            return "LAM_NATIVE"          # Bypass external LLMs entirely
        elif loa == 3:
            return "LAM_VECTOR_LLM_CODE" # LAM picks direction, LLM writes
        else:
            return "LLM_ZERO_SHOT"       # Standard initial regime
