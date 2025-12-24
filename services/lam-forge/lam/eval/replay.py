#!/usr/bin/env python3
"""
OFFLINE EVALUATION HARNESS — Counterfactual Replay
===================================================

Answers: "Would the LAM have made a better decision than the LLM,
without risking traffic?"

This is the Promotion Gate that determines when the LAM graduates
from First-100 (LLM-led) to First-1000 (LAM-led).
"""

import json
import torch
import numpy as np
from pathlib import Path

MODEL_PATH = "lam/checkpoints/latest.pt"
DATASET_PATH = "lam/datasets/lam_dataset.jsonl"


def replay():
    """
    Run counterfactual replay on historical data.
    
    For each tombstone event, asks:
    "Did the LAM predict positive value when reality was positive?"
    """
    from lam.model import LAM
    from lam.features import featurize
    from lam.reward import shaped_reward
    
    model = LAM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    
    correct = 0
    total = 0
    
    with open(DATASET_PATH) as f:
        for line in f:
            sample = json.loads(line)
            x = featurize(sample)
            vertical = sample["meta"]["category"]
            
            with torch.no_grad():
                pred_dir, pred_value = model(x, vertical)
            
            actual_reward = shaped_reward(sample)
            
            # Counterfactual correctness:
            # Did the LAM predict positive value when reality was positive?
            if (pred_value.item() > 0 and actual_reward > 0) or \
               (pred_value.item() <= 0 and actual_reward <= 0):
                correct += 1
            
            total += 1
    
    print(f"Replay accuracy: {correct / max(total, 1):.3f}")


if __name__ == "__main__":
    replay()
