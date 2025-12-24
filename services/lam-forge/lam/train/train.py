#!/usr/bin/env python3
"""
CURRICULUM TRAINING — Stage-Gated Learning
===========================================

Trains the LAM through progressive difficulty stages.
Easy truths first, subtle patterns last.
"""

import json
import torch
from lam.model import LAM
from lam.features import featurize
from lam.reward import shaped_reward
from lam.train.curriculum import difficulty_bucket

EPOCHS_PER_STAGE = 3


def train(dataset_path):
    """
    Train LAM with curriculum learning.
    
    Progresses through 4 stages of increasing difficulty.
    """
    model = LAM()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f]
    
    for stage in [1, 2, 3, 4]:
        stage_data = [s for s in dataset if difficulty_bucket(s) == stage]
        
        for epoch in range(EPOCHS_PER_STAGE):
            for sample in stage_data:
                x = featurize(sample)
                vertical = sample["meta"]["category"]
                
                pred_dir, pred_val = model(x, vertical)
                
                reward = shaped_reward(sample)
                reward_tensor = torch.tensor([reward], dtype=torch.float32)
                
                loss = torch.nn.functional.mse_loss(pred_val, reward_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print(f"Finished curriculum stage {stage}")
    
    torch.save(model.state_dict(), "lam/checkpoints/latest.pt")


if __name__ == "__main__":
    train("lam/datasets/lam_dataset.jsonl")
