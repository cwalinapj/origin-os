#!/usr/bin/env python3
"""
OFFLINE EVALUATION HARNESS — The Calibration Gatekeeper
========================================================

Before the LAM controls live containers, it must pass this Calibration Gate.

Uses Counterfactual Replay to answer:
"If the LAM were in control of the previous 500 tombstone events,
how many times would its predicted mutation direction have aligned
with the actual behavioral winner?"

Metrics:
1. Directional Accuracy: Did it pick the right vector?
2. Calibration Error: Was the magnitude realistic?
3. Sensitivity Analysis: Are predictions stable?

Promotion Gate:
- Accuracy > 75%
- Mean Absolute Error < 0.15
- Sensitivity Score < 2.0 (no wild spikes)

Only then does the LAM graduate from First-100 (LLM-led) to First-1000 (LAM-led).
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/lam"))

# Promotion thresholds
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.75"))
MAX_CALIBRATION_ERROR = float(os.getenv("MAX_CALIBRATION_ERROR", "0.15"))
MAX_SENSITIVITY = float(os.getenv("MAX_SENSITIVITY", "2.0"))
MIN_SAMPLES = int(os.getenv("MIN_EVAL_SAMPLES", "500"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval-harness")

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class EvaluationSample:
    """A single evaluation sample from historical data."""
    sample_id: str
    site_id: str
    vertical: str
    
    # Input features
    base_state: np.ndarray
    mutation_vector: np.ndarray
    context: Dict[str, Any]
    
    # Ground truth outcome
    actual_delta: float  # The real behavioral score change
    was_winner: bool     # Did this mutation win?
    
    # Metadata
    timestamp: str
    tombstone_reason: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of a single prediction evaluation."""
    sample_id: str
    predicted_value: float
    actual_delta: float
    directionally_correct: bool
    calibration_error: float
    
    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "predicted_value": self.predicted_value,
            "actual_delta": self.actual_delta,
            "directionally_correct": self.directionally_correct,
            "calibration_error": self.calibration_error
        }


@dataclass
class HarnessReport:
    """Complete evaluation harness report."""
    run_id: str
    model_version: str
    timestamp: str
    
    # Sample stats
    total_samples: int
    samples_by_vertical: Dict[str, int]
    
    # Core metrics
    directional_accuracy: float
    mean_calibration_error: float
    calibration_error_std: float
    
    # Sensitivity metrics
    sensitivity_score: float
    sensitive_dimensions: List[int]
    
    # Per-vertical breakdown
    accuracy_by_vertical: Dict[str, float]
    calibration_by_vertical: Dict[str, float]
    
    # Promotion decision
    passed: bool
    failure_reasons: List[str]
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "total_samples": self.total_samples,
            "samples_by_vertical": self.samples_by_vertical,
            "directional_accuracy": self.directional_accuracy,
            "mean_calibration_error": self.mean_calibration_error,
            "calibration_error_std": self.calibration_error_std,
            "sensitivity_score": self.sensitivity_score,
            "sensitive_dimensions": self.sensitive_dimensions,
            "accuracy_by_vertical": self.accuracy_by_vertical,
            "calibration_by_vertical": self.calibration_by_vertical,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons
        }


# =============================================================================
# MODEL LOADER
# =============================================================================

class LAMModelLoader:
    """Loads and manages LAM model for evaluation."""
    
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.model_version = None
    
    def load_latest(self):
        """Load the latest model checkpoint."""
        latest_path = self.model_dir / "latest.pt"
        
        if not latest_path.exists():
            raise FileNotFoundError(f"No model found at {latest_path}")
        
        # Import the model class
        from training_loop import LAMModel
        
        checkpoint = torch.load(latest_path, map_location="cpu")
        self.model = LAMModel()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model_version = f"step_{checkpoint.get('step', 'unknown')}"
        
        logger.info(f"Loaded model version: {self.model_version}")
        return self.model
    
    def predict(self, base_state: torch.Tensor, mutation_vector: torch.Tensor) -> float:
        """Make a prediction for a mutation."""
        with torch.no_grad():
            # Encode page state
            page_features = self.model.encode_page(base_state.unsqueeze(0))
            # Predict reward
            pred = self.model.reward_head(page_features, mutation_vector.unsqueeze(0))
            return pred.item()


# =============================================================================
# DATA LOADER
# =============================================================================

class EvaluationDataLoader:
    """Loads evaluation samples from MongoDB and Redis."""
    
    def __init__(self, mongo_client: AsyncIOMotorClient, redis_client: redis.Redis):
        self.mongo = mongo_client
        self.redis = redis_client
        self.db = mongo_client.origin_os
    
    async def load_samples(self, limit: int = 1000) -> List[EvaluationSample]:
        """Load samples from tombstone records and session history."""
        samples = []
        
        # Load from MongoDB tombstones
        tombstones = await self.db.tombstones.find({
            "outcome": {"$exists": True}
        }).sort("timestamp", -1).limit(limit).to_list(length=limit)
        
        for ts in tombstones:
            try:
                sample = EvaluationSample(
                    sample_id=str(ts["_id"]),
                    site_id=ts.get("site_id", ""),
                    vertical=ts.get("vertical", "unknown"),
                    base_state=np.array(ts.get("base_state_embedding", [])),
                    mutation_vector=np.array(ts.get("mutation_vector", [])),
                    context=ts.get("context", {}),
                    actual_delta=ts.get("outcome", {}).get("mean_score_delta", 0),
                    was_winner=ts.get("outcome", {}).get("was_winner", False),
                    timestamp=ts.get("timestamp", ""),
                    tombstone_reason=ts.get("reason")
                )
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to parse tombstone: {e}")
        
        # Also load from Redis behavioral scores
        redis_samples = await self._load_from_redis(limit - len(samples))
        samples.extend(redis_samples)
        
        logger.info(f"Loaded {len(samples)} evaluation samples")
        return samples
    
    async def _load_from_redis(self, limit: int) -> List[EvaluationSample]:
        """Load additional samples from Redis."""
        samples = []
        
        cursor = 0
        while len(samples) < limit:
            cursor, keys = await self.redis.scan(
                cursor, match="eval_sample:*", count=100
            )
            
            for key in keys:
                if len(samples) >= limit:
                    break
                    
                data = await self.redis.hgetall(key)
                if data:
                    try:
                        samples.append(EvaluationSample(
                            sample_id=key.decode().split(":")[-1],
                            site_id=data.get(b"site_id", b"").decode(),
                            vertical=data.get(b"vertical", b"unknown").decode(),
                            base_state=np.array(json.loads(data.get(b"base_state", b"[]"))),
                            mutation_vector=np.array(json.loads(data.get(b"mutation_vector", b"[]"))),
                            context=json.loads(data.get(b"context", b"{}")),
                            actual_delta=float(data.get(b"actual_delta", 0)),
                            was_winner=data.get(b"was_winner", b"0") == b"1",
                            timestamp=data.get(b"timestamp", b"").decode()
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to parse redis sample: {e}")
            
            if cursor == 0:
                break
        
        return samples


# =============================================================================
# EVALUATOR
# =============================================================================

class CounterfactualEvaluator:
    """
    Runs counterfactual replay to evaluate LAM predictions.
    
    Core question: "If the LAM were in control, would it have made
    the same decisions as the winning mutations?"
    """
    
    def __init__(self, model_loader: LAMModelLoader):
        self.model_loader = model_loader
    
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single sample."""
        # Convert to tensors
        base_state = torch.tensor(sample.base_state, dtype=torch.float32)
        mutation_vector = torch.tensor(sample.mutation_vector, dtype=torch.float32)
        
        # Get model prediction
        pred_value = self.model_loader.predict(base_state, mutation_vector)
        
        # Directional accuracy: Did the model predict positive/negative correctly?
        directionally_correct = (pred_value > 0) == (sample.actual_delta > 0)
        
        # Calibration error: How far was the predicted magnitude from reality?
        calibration_error = abs(pred_value - sample.actual_delta)
        
        return EvaluationResult(
            sample_id=sample.sample_id,
            predicted_value=pred_value,
            actual_delta=sample.actual_delta,
            directionally_correct=directionally_correct,
            calibration_error=calibration_error
        )
    
    def run_sensitivity_analysis(
        self,
        sample: EvaluationSample,
        perturbation_size: float = 0.1,
        num_dimensions: int = 10
    ) -> Tuple[float, List[int]]:
        """
        Run sensitivity analysis on a sample.
        
        Question: "If I perturb dimension D by 0.1, does the prediction
        change reasonably or spike to infinity?"
        
        Returns:
            (max_sensitivity, list of sensitive dimensions)
        """
        base_state = torch.tensor(sample.base_state, dtype=torch.float32)
        mutation_vector = torch.tensor(sample.mutation_vector, dtype=torch.float32)
        
        # Get baseline prediction
        baseline = self.model_loader.predict(base_state, mutation_vector)
        
        sensitivities = []
        sensitive_dims = []
        
        # Test each dimension
        for dim in range(min(len(mutation_vector), 50)):  # Test first 50 dims
            # Positive perturbation
            perturbed = mutation_vector.clone()
            perturbed[dim] += perturbation_size
            pred_pos = self.model_loader.predict(base_state, perturbed)
            
            # Negative perturbation
            perturbed = mutation_vector.clone()
            perturbed[dim] -= perturbation_size
            pred_neg = self.model_loader.predict(base_state, perturbed)
            
            # Sensitivity = max change from baseline
            sensitivity = max(abs(pred_pos - baseline), abs(pred_neg - baseline))
            sensitivities.append(sensitivity)
            
            # Flag dimensions with extreme sensitivity
            if sensitivity > MAX_SENSITIVITY:
                sensitive_dims.append(dim)
        
        max_sensitivity = max(sensitivities) if sensitivities else 0.0
        return max_sensitivity, sensitive_dims


# =============================================================================
# HARNESS RUNNER
# =============================================================================

class EvaluationHarness:
    """
    Complete evaluation harness that runs all checks
    and produces a Go/No-Go decision.
    """
    
    def __init__(
        self,
        model_loader: LAMModelLoader,
        data_loader: EvaluationDataLoader,
        redis_client: redis.Redis
    ):
        self.model_loader = model_loader
        self.data_loader = data_loader
        self.evaluator = CounterfactualEvaluator(model_loader)
        self.redis = redis_client
    
    async def run_full_evaluation(self) -> HarnessReport:
        """Run complete evaluation and produce report."""
        run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Load model
        try:
            self.model_loader.load_latest()
        except FileNotFoundError as e:
            return self._create_failure_report(run_id, [str(e)])
        
        # Load samples
        samples = await self.data_loader.load_samples(limit=MIN_SAMPLES * 2)
        
        if len(samples) < MIN_SAMPLES:
            return self._create_failure_report(
                run_id,
                [f"Insufficient samples: {len(samples)} < {MIN_SAMPLES}"]
            )
        
        # Run evaluation
        results: List[EvaluationResult] = []
        results_by_vertical: Dict[str, List[EvaluationResult]] = {}
        
        for sample in samples:
            result = self.evaluator.evaluate_sample(sample)
            results.append(result)
            
            if sample.vertical not in results_by_vertical:
                results_by_vertical[sample.vertical] = []
            results_by_vertical[sample.vertical].append(result)
        
        # Run sensitivity analysis on a subset
        sensitivity_samples = samples[:50]  # Test first 50
        max_sensitivity = 0.0
        all_sensitive_dims = set()
        
        for sample in sensitivity_samples:
            sens, dims = self.evaluator.run_sensitivity_analysis(sample)
            max_sensitivity = max(max_sensitivity, sens)
            all_sensitive_dims.update(dims)
        
        # Compute metrics
        directional_accuracy = np.mean([r.directionally_correct for r in results])
        calibration_errors = [r.calibration_error for r in results]
        mean_calibration_error = np.mean(calibration_errors)
        calibration_error_std = np.std(calibration_errors)
        
        # Per-vertical breakdown
        accuracy_by_vertical = {}
        calibration_by_vertical = {}
        samples_by_vertical = {}
        
        for vertical, v_results in results_by_vertical.items():
            accuracy_by_vertical[vertical] = np.mean([r.directionally_correct for r in v_results])
            calibration_by_vertical[vertical] = np.mean([r.calibration_error for r in v_results])
            samples_by_vertical[vertical] = len(v_results)
        
        # Determine pass/fail
        failure_reasons = []
        
        if directional_accuracy < MIN_ACCURACY:
            failure_reasons.append(
                f"Accuracy {directional_accuracy:.2%} < {MIN_ACCURACY:.2%}"
            )
        
        if mean_calibration_error > MAX_CALIBRATION_ERROR:
            failure_reasons.append(
                f"Calibration error {mean_calibration_error:.4f} > {MAX_CALIBRATION_ERROR}"
            )
        
        if max_sensitivity > MAX_SENSITIVITY:
            failure_reasons.append(
                f"Sensitivity {max_sensitivity:.2f} > {MAX_SENSITIVITY} (dims: {list(all_sensitive_dims)[:5]})"
            )
        
        passed = len(failure_reasons) == 0
        
        report = HarnessReport(
            run_id=run_id,
            model_version=self.model_loader.model_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_samples=len(samples),
            samples_by_vertical=samples_by_vertical,
            directional_accuracy=directional_accuracy,
            mean_calibration_error=mean_calibration_error,
            calibration_error_std=calibration_error_std,
            sensitivity_score=max_sensitivity,
            sensitive_dimensions=list(all_sensitive_dims)[:10],
            accuracy_by_vertical=accuracy_by_vertical,
            calibration_by_vertical=calibration_by_vertical,
            passed=passed,
            failure_reasons=failure_reasons
        )
        
        # Store report
        await self._store_report(report)
        
        # Log result
        status = "GO - Ready for First-1000" if passed else "NO-GO - Requires further training"
        logger.info(f"Evaluation {run_id}: {status}")
        logger.info(f"  Accuracy: {directional_accuracy:.2%}")
        logger.info(f"  Calibration Error: {mean_calibration_error:.4f}")
        logger.info(f"  Sensitivity: {max_sensitivity:.2f}")
        
        return report
    
    def _create_failure_report(self, run_id: str, reasons: List[str]) -> HarnessReport:
        """Create a failure report when evaluation cannot proceed."""
        return HarnessReport(
            run_id=run_id,
            model_version="unknown",
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_samples=0,
            samples_by_vertical={},
            directional_accuracy=0.0,
            mean_calibration_error=1.0,
            calibration_error_std=0.0,
            sensitivity_score=0.0,
            sensitive_dimensions=[],
            accuracy_by_vertical={},
            calibration_by_vertical={},
            passed=False,
            failure_reasons=reasons
        )
    
    async def _store_report(self, report: HarnessReport):
        """Store evaluation report."""
        await self.redis.xadd(
            "evaluation_reports",
            {"report": json.dumps(report.to_dict())},
            maxlen=100
        )
        
        # Store latest result for quick access
        await self.redis.set(
            f"eval_latest:{report.model_version}",
            json.dumps(report.to_dict())
        )
        
        # Update promotion status
        await self.redis.hset(
            "lam_promotion_status",
            mapping={
                "model_version": report.model_version,
                "passed": str(report.passed),
                "accuracy": str(report.directional_accuracy),
                "calibration_error": str(report.mean_calibration_error),
                "timestamp": report.timestamp
            }
        )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="LAM Evaluation Harness",
    description="Calibration gatekeeper for LAM promotion",
    version="1.0.0"
)

redis_client: Optional[redis.Redis] = None
mongo_client: Optional[AsyncIOMotorClient] = None
harness: Optional[EvaluationHarness] = None


class EvaluationResponse(BaseModel):
    run_id: str
    passed: bool
    accuracy: float
    calibration_error: float
    sensitivity: float
    failure_reasons: List[str]


@app.on_event("startup")
async def startup():
    global redis_client, mongo_client, harness
    redis_client = redis.from_url(REDIS_URL)
    mongo_client = AsyncIOMotorClient(MONGO_URL)
    
    model_loader = LAMModelLoader()
    data_loader = EvaluationDataLoader(mongo_client, redis_client)
    harness = EvaluationHarness(model_loader, data_loader, redis_client)


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()
    if mongo_client:
        mongo_client.close()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation(background_tasks: BackgroundTasks):
    """Run full LAM evaluation."""
    report = await harness.run_full_evaluation()
    
    return EvaluationResponse(
        run_id=report.run_id,
        passed=report.passed,
        accuracy=report.directional_accuracy,
        calibration_error=report.mean_calibration_error,
        sensitivity=report.sensitivity_score,
        failure_reasons=report.failure_reasons
    )


@app.get("/status")
async def get_promotion_status():
    """Get current LAM promotion status."""
    status = await redis_client.hgetall("lam_promotion_status")
    if not status:
        return {"status": "no_evaluation_run"}
    
    return {
        "model_version": status.get(b"model_version", b"").decode(),
        "passed": status.get(b"passed", b"false").decode() == "True",
        "accuracy": float(status.get(b"accuracy", 0)),
        "calibration_error": float(status.get(b"calibration_error", 1)),
        "timestamp": status.get(b"timestamp", b"").decode()
    }


@app.get("/reports")
async def get_reports(limit: int = 10):
    """Get recent evaluation reports."""
    reports = await redis_client.xrange(
        "evaluation_reports",
        min="-",
        max="+",
        count=limit
    )
    
    return {
        "reports": [
            json.loads(data[b"report"]) for _, data in reports
        ]
    }


@app.post("/promote")
async def promote_model():
    """Manually promote model if evaluation passed."""
    status = await redis_client.hgetall("lam_promotion_status")
    
    if not status or status.get(b"passed", b"false").decode() != "True":
        raise HTTPException(
            status_code=400,
            detail="Model has not passed evaluation"
        )
    
    # Transition sites to LAM-led mode
    cursor = 0
    promoted_sites = 0
    
    while True:
        cursor, keys = await redis_client.scan(
            cursor, match="site_mode:*", count=100
        )
        
        for key in keys:
            samples = await redis_client.hget(key, "sample_count")
            if samples and int(samples) >= 100:
                await redis_client.hset(key, "mode", "lam_led")
                await redis_client.hset(key, "promotion_time", datetime.now(timezone.utc).isoformat())
                promoted_sites += 1
        
        if cursor == 0:
            break
    
    return {
        "status": "promoted",
        "model_version": status.get(b"model_version", b"").decode(),
        "promoted_sites": promoted_sites
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
