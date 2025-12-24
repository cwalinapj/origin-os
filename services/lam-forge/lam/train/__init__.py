"""LAM Training Module"""
from lam.train.curriculum import difficulty_bucket
from lam.train.train import train

__all__ = ["difficulty_bucket", "train"]
