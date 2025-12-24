#!/usr/bin/env python3
"""
CURRICULUM LEARNING — Pedagogical Training for LAM
===================================================

Transforms the LAM from a raw statistical model into a reliable marketing
strategist by filtering data into difficulty buckets.

Curriculum Stages:
1. High-Signal Wins: Clear structural victories (Δ > +0.3)
2. Clear Failures: Obvious mistakes (Δ < -0.3) — The Inversion Bridge
3. Marginal Cases: Small gains/losses (-0.3 < Δ < +0.3)
4. Conflicting Signals: High variance, mixed results — Market Drift

Learning Flow:
- Stage 1 teaches "what works"
- Stage 2 teaches "the negative space" (Structural Guardrails)
- Stage 3 teaches "nuance and edge cases"
- Stage 4 teaches "when to increase uncertainty"

The model graduates to the next stage only after mastering the current one.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/lam"))

# Curriculum thresholds
STAGE_THRESHOLDS = {
    1: 0.85,  # High-Signal Wins: 85% accuracy required
    2: 0.80,  # Clear Failures: 80% accuracy required
    3: 0.70,  # Marginal Cases: 70% accuracy required
    4: 0.60,  # Conflicting Signals: 60% accuracy required
}

# Delta thresholds for classification
HIGH_SIGNAL_THRESHOLD = 0.3
MARGINAL_THRESHOLD = 0.1
VARIANCE_THRESHOLD = 0.5  # For conflicting signal detection

MAX_EPOCHS_PER_STAGE = 50
BATCH_SIZE = 32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("curriculum")

# =============================================================================
# ENUMS & DATA MODELS
# =============================================================================

class CurriculumStage(IntEnum):
    HIGH_SIGNAL_WINS = 1
    CLEAR_FAILURES = 2
    MARGINAL_CASES = 3
    CONFLICTING_SIGNALS = 4


@dataclass
class CurriculumSample:
    """A training sample with curriculum metadata."""
    sample_id: str
    site_id: str
    vertical: str
    
    # Features
    base_state: np.ndarray
    mutation_vector: np.ndarray
    context: Dict[str, Any]
    
    # Outcomes
    actual_delta: float
    variance: float  # Variance of outcomes for this mutation type
    sample_count: int  # How many observations
    
    # Curriculum metadata
    stage: CurriculumStage
    difficulty_score: float  # 0-1, higher = harder
    
    # Audit lineage
    gclid_hash: Optional[str] = None
    tombstone_id: Optional[str] = None
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "site_id": self.site_id,
            "vertical": self.vertical,
            "base_state": self.base_state.tolist(),
            "mutation_vector": self.mutation_vector.tolist(),
            "context": self.context,
            "actual_delta": self.actual_delta,
            "variance": self.variance,
            "sample_count": self.sample_count,
            "stage": self.stage.value,
            "difficulty_score": self.difficulty_score,
            "gclid_hash": self.gclid_hash,
            "tombstone_id": self.tombstone_id,
            "timestamp": self.timestamp
        }


@dataclass
class StageMetrics:
    """Metrics for a curriculum stage."""
    stage: CurriculumStage
    epoch: int
    samples_trained: int
    directional_accuracy: float
    calibration_error: float
    loss: float
    graduated: bool
    
    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "epoch": self.epoch,
            "samples_trained": self.samples_trained,
            "directional_accuracy": self.directional_accuracy,
            "calibration_error": self.calibration_error,
            "loss": self.loss,
            "graduated": self.graduated
        }


@dataclass
class CurriculumProgress:
    """Overall curriculum progress."""
    current_stage: CurriculumStage
    stages_completed: List[int]
    total_epochs: int
    total_samples: int
    stage_metrics: Dict[int, StageMetrics]
    started_at: str
    last_updated: str
    
    def to_dict(self) -> dict:
        return {
            "current_stage": self.current_stage.value,
            "stages_completed": self.stages_completed,
            "total_epochs": self.total_epochs,
            "total_samples": self.total_samples,
            "stage_metrics": {k: v.to_dict() for k, v in self.stage_metrics.items()},
            "started_at": self.started_at,
            "last_updated": self.last_updated
        }


@dataclass
class AuditLogEntry:
    """Complete audit trail entry for a training event."""
    log_id: str
    timestamp: str
    
    # Traffic DNA
    gclid_hash: str
    session_id: str
    site_id: str
    vertical: str
    
    # Mutation Intent
    mutation_vector: List[float]
    mutation_type: str  # "structural", "semantic", "cosmetic"
    
    # Curriculum Context
    curriculum_stage: int
    difficulty_score: float
    
    # Model Performance
    predicted_delta: float
    actual_delta: float
    surprise_delta: float  # Measures hallucination
    directionally_correct: bool
    
    # Container Lifecycle
    tombstone_id: Optional[str]
    tombstone_reason: Optional[str]
    container_lifespan_seconds: Optional[float]
    
    # Promotion Status
    contributed_to_graduation: bool
    
    def to_dict(self) -> dict:
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp,
            "gclid_hash": self.gclid_hash,
            "session_id": self.session_id,
            "site_id": self.site_id,
            "vertical": self.vertical,
            "mutation_vector": self.mutation_vector,
            "mutation_type": self.mutation_type,
            "curriculum_stage": self.curriculum_stage,
            "difficulty_score": self.difficulty_score,
            "predicted_delta": self.predicted_delta,
            "actual_delta": self.actual_delta,
            "surprise_delta": self.surprise_delta,
            "directionally_correct": self.directionally_correct,
            "tombstone_id": self.tombstone_id,
            "tombstone_reason": self.tombstone_reason,
            "container_lifespan_seconds": self.container_lifespan_seconds,
            "contributed_to_graduation": self.contributed_to_graduation
        }


# =============================================================================
# CURRICULUM CLASSIFIER
# =============================================================================

class CurriculumClassifier:
    """
    Classifies samples into curriculum stages based on signal clarity.
    
    Stage Assignment Logic:
    - Stage 1: |Δ| > 0.3 AND Δ > 0 AND variance < 0.5 (Clear wins)
    - Stage 2: |Δ| > 0.3 AND Δ < 0 AND variance < 0.5 (Clear failures)
    - Stage 3: |Δ| <= 0.3 AND variance < 0.5 (Marginal cases)
    - Stage 4: variance >= 0.5 (Conflicting signals)
    """
    
    def __init__(
        self,
        high_signal_threshold: float = HIGH_SIGNAL_THRESHOLD,
        variance_threshold: float = VARIANCE_THRESHOLD
    ):
        self.high_signal_threshold = high_signal_threshold
        self.variance_threshold = variance_threshold
    
    def classify(self, sample: Dict[str, Any]) -> Tuple[CurriculumStage, float]:
        """
        Classify a sample into a curriculum stage.
        
        Returns:
            (stage, difficulty_score)
        """
        delta = sample.get("actual_delta", 0)
        variance = sample.get("variance", 0)
        sample_count = sample.get("sample_count", 1)
        
        # Calculate difficulty score (0-1, higher = harder)
        # Based on: low sample count, high variance, small delta
        difficulty = self._calculate_difficulty(delta, variance, sample_count)
        
        # Stage 4: Conflicting Signals (high variance)
        if variance >= self.variance_threshold:
            return CurriculumStage.CONFLICTING_SIGNALS, difficulty
        
        # Stage 1: High-Signal Wins
        if delta > self.high_signal_threshold:
            return CurriculumStage.HIGH_SIGNAL_WINS, difficulty
        
        # Stage 2: Clear Failures
        if delta < -self.high_signal_threshold:
            return CurriculumStage.CLEAR_FAILURES, difficulty
        
        # Stage 3: Marginal Cases
        return CurriculumStage.MARGINAL_CASES, difficulty
    
    def _calculate_difficulty(
        self,
        delta: float,
        variance: float,
        sample_count: int
    ) -> float:
        """Calculate difficulty score for a sample."""
        # Small delta = harder to predict
        delta_difficulty = 1 - min(abs(delta) / 0.5, 1.0)
        
        # High variance = harder
        variance_difficulty = min(variance / 1.0, 1.0)
        
        # Low sample count = less reliable signal
        count_difficulty = 1 - min(sample_count / 100, 1.0)
        
        # Weighted combination
        difficulty = (
            0.4 * delta_difficulty +
            0.4 * variance_difficulty +
            0.2 * count_difficulty
        )
        
        return difficulty
    
    def filter_by_stage(
        self,
        samples: List[Dict],
        stage: CurriculumStage
    ) -> List[Dict]:
        """Filter samples to only include those in the specified stage."""
        filtered = []
        for sample in samples:
            sample_stage, difficulty = self.classify(sample)
            if sample_stage == stage:
                sample["curriculum_stage"] = stage.value
                sample["difficulty_score"] = difficulty
                filtered.append(sample)
        return filtered


# =============================================================================
# CURRICULUM DATASET
# =============================================================================

class CurriculumDataset(Dataset):
    """Dataset that serves samples for a specific curriculum stage."""
    
    def __init__(self, samples: List[CurriculumSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "base_state": torch.tensor(sample.base_state, dtype=torch.float32),
            "mutation_vector": torch.tensor(sample.mutation_vector, dtype=torch.float32),
            "actual_delta": torch.tensor(sample.actual_delta, dtype=torch.float32),
            "difficulty_score": torch.tensor(sample.difficulty_score, dtype=torch.float32),
            "stage": torch.tensor(sample.stage.value, dtype=torch.long)
        }


# =============================================================================
# CURRICULUM TRAINER
# =============================================================================

class CurriculumTrainer:
    """
    Trains the LAM through progressive curriculum stages.
    
    The model graduates to the next stage only after achieving
    the required accuracy threshold on the current stage.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        redis_client: redis.Redis,
        mongo_client: AsyncIOMotorClient,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.redis = redis_client
        self.mongo = mongo_client
        self.db = mongo_client.origin_os
        
        self.classifier = CurriculumClassifier()
        self.progress = None
        
        # Audit logger
        self.audit_buffer: List[AuditLogEntry] = []
    
    async def load_samples(self, limit: int = 10000) -> List[CurriculumSample]:
        """Load and classify samples from database."""
        samples = []
        
        # Load from MongoDB
        cursor = self.db.training_samples.find({}).limit(limit)
        async for doc in cursor:
            stage, difficulty = self.classifier.classify(doc)
            
            samples.append(CurriculumSample(
                sample_id=str(doc["_id"]),
                site_id=doc.get("site_id", ""),
                vertical=doc.get("vertical", "unknown"),
                base_state=np.array(doc.get("base_state", [])),
                mutation_vector=np.array(doc.get("mutation_vector", [])),
                context=doc.get("context", {}),
                actual_delta=doc.get("actual_delta", 0),
                variance=doc.get("variance", 0),
                sample_count=doc.get("sample_count", 1),
                stage=stage,
                difficulty_score=difficulty,
                gclid_hash=doc.get("gclid_hash"),
                tombstone_id=doc.get("tombstone_id"),
                timestamp=doc.get("timestamp", "")
            ))
        
        logger.info(f"Loaded {len(samples)} samples")
        
        # Log stage distribution
        stage_counts = {}
        for s in samples:
            stage_counts[s.stage.name] = stage_counts.get(s.stage.name, 0) + 1
        logger.info(f"Stage distribution: {stage_counts}")
        
        return samples
    
    async def train_with_curriculum(self, samples: List[CurriculumSample]):
        """
        Train through all curriculum stages progressively.
        
        The model only advances when it masters the current stage.
        """
        self.progress = CurriculumProgress(
            current_stage=CurriculumStage.HIGH_SIGNAL_WINS,
            stages_completed=[],
            total_epochs=0,
            total_samples=0,
            stage_metrics={},
            started_at=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        for stage in CurriculumStage:
            logger.info(f"{'='*60}")
            logger.info(f"ENTERING STAGE {stage.value}: {stage.name}")
            logger.info(f"{'='*60}")
            
            # Filter samples for this stage
            stage_samples = [s for s in samples if s.stage == stage]
            
            if not stage_samples:
                logger.warning(f"No samples for stage {stage.name}, skipping")
                continue
            
            # Train on this stage
            metrics = await self._train_stage(stage, stage_samples)
            
            # Record progress
            self.progress.stage_metrics[stage.value] = metrics
            self.progress.current_stage = stage
            self.progress.last_updated = datetime.now(timezone.utc).isoformat()
            
            if metrics.graduated:
                self.progress.stages_completed.append(stage.value)
                logger.info(f"✓ GRADUATED from Stage {stage.value}: {stage.name}")
            else:
                logger.warning(f"✗ PLATEAUED at Stage {stage.value}: {stage.name}")
            
            # Save checkpoint after each stage
            await self._save_checkpoint(stage)
        
        # Save final progress
        await self._save_progress()
        
        return self.progress
    
    async def _train_stage(
        self,
        stage: CurriculumStage,
        samples: List[CurriculumSample]
    ) -> StageMetrics:
        """Train on a single curriculum stage."""
        dataset = CurriculumDataset(samples)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        threshold = STAGE_THRESHOLDS[stage.value]
        best_accuracy = 0.0
        epoch = 0
        
        for epoch in range(MAX_EPOCHS_PER_STAGE):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            calibration_errors = []
            
            for batch in dataloader:
                # Move to device
                base_state = batch["base_state"].to(self.device)
                mutation = batch["mutation_vector"].to(self.device)
                actual = batch["actual_delta"].to(self.device)
                difficulty = batch["difficulty_score"].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Get predictions
                page_features = self.model.encode_page(base_state)
                pred = self.model.reward_head(page_features, mutation)
                
                # Stage-specific loss weighting
                # Easier stages get higher confidence requirements
                confidence_weight = 1.0 + (4 - stage.value) * 0.25
                
                # Difficulty-weighted loss
                # Harder samples within a stage get slightly lower weight
                sample_weight = 1.0 - (difficulty * 0.3)
                
                # MSE Loss with weighting
                loss = F.mse_loss(pred, actual, reduction='none')
                weighted_loss = (loss * sample_weight * confidence_weight).mean()
                
                # Backward pass
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += weighted_loss.item()
                
                # Calculate metrics
                pred_np = pred.detach().cpu().numpy()
                actual_np = actual.detach().cpu().numpy()
                
                # Directional accuracy
                correct += np.sum((pred_np > 0) == (actual_np > 0))
                total += len(pred_np)
                
                # Calibration error
                calibration_errors.extend(np.abs(pred_np - actual_np).tolist())
            
            # Epoch metrics
            accuracy = correct / total if total > 0 else 0
            avg_loss = epoch_loss / len(dataloader)
            avg_calibration = np.mean(calibration_errors)
            
            logger.info(
                f"  Stage {stage.value} Epoch {epoch + 1}: "
                f"acc={accuracy:.2%}, loss={avg_loss:.4f}, cal={avg_calibration:.4f}"
            )
            
            # Update progress
            self.progress.total_epochs += 1
            self.progress.total_samples += len(samples)
            
            # Check graduation
            if accuracy >= threshold:
                return StageMetrics(
                    stage=stage,
                    epoch=epoch + 1,
                    samples_trained=len(samples) * (epoch + 1),
                    directional_accuracy=accuracy,
                    calibration_error=avg_calibration,
                    loss=avg_loss,
                    graduated=True
                )
            
            best_accuracy = max(best_accuracy, accuracy)
        
        # Plateaued - didn't graduate
        return StageMetrics(
            stage=stage,
            epoch=MAX_EPOCHS_PER_STAGE,
            samples_trained=len(samples) * MAX_EPOCHS_PER_STAGE,
            directional_accuracy=best_accuracy,
            calibration_error=avg_calibration,
            loss=avg_loss,
            graduated=False
        )
    
    async def log_audit_entry(
        self,
        sample: CurriculumSample,
        predicted_delta: float,
        contributed_to_graduation: bool
    ):
        """Log an audit entry for a training event."""
        entry = AuditLogEntry(
            log_id=f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            gclid_hash=sample.gclid_hash or "",
            session_id=sample.sample_id,
            site_id=sample.site_id,
            vertical=sample.vertical,
            mutation_vector=sample.mutation_vector.tolist(),
            mutation_type=self._classify_mutation_type(sample.mutation_vector),
            curriculum_stage=sample.stage.value,
            difficulty_score=sample.difficulty_score,
            predicted_delta=predicted_delta,
            actual_delta=sample.actual_delta,
            surprise_delta=abs(predicted_delta - sample.actual_delta),
            directionally_correct=(predicted_delta > 0) == (sample.actual_delta > 0),
            tombstone_id=sample.tombstone_id,
            tombstone_reason=None,
            container_lifespan_seconds=None,
            contributed_to_graduation=contributed_to_graduation
        )
        
        self.audit_buffer.append(entry)
        
        # Flush buffer periodically
        if len(self.audit_buffer) >= 100:
            await self._flush_audit_buffer()
    
    def _classify_mutation_type(self, mutation_vector: np.ndarray) -> str:
        """Classify mutation type based on vector characteristics."""
        if len(mutation_vector) < 64:
            return "unknown"
        
        # First 64 dims = structural, rest = semantic
        structural_energy = np.sum(np.abs(mutation_vector[:64]))
        semantic_energy = np.sum(np.abs(mutation_vector[64:]))
        
        if structural_energy > semantic_energy * 2:
            return "structural"
        elif semantic_energy > structural_energy * 2:
            return "semantic"
        else:
            return "cosmetic"
    
    async def _flush_audit_buffer(self):
        """Flush audit entries to database."""
        if not self.audit_buffer:
            return
        
        entries = [e.to_dict() for e in self.audit_buffer]
        await self.db.audit_log.insert_many(entries)
        self.audit_buffer = []
    
    async def _save_checkpoint(self, stage: CurriculumStage):
        """Save model checkpoint after completing a stage."""
        checkpoint_path = MODEL_DIR / f"curriculum_stage_{stage.value}.pt"
        
        torch.save({
            "stage": stage.value,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "progress": self.progress.to_dict()
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    async def _save_progress(self):
        """Save curriculum progress to Redis."""
        await self.redis.set(
            "curriculum_progress",
            json.dumps(self.progress.to_dict())
        )
        
        # Also save to MongoDB for persistence
        await self.db.curriculum_progress.update_one(
            {"_id": "latest"},
            {"$set": self.progress.to_dict()},
            upsert=True
        )


# =============================================================================
# STAGE-SPECIFIC TRAINING STRATEGIES
# =============================================================================

class Stage1Strategy:
    """
    Stage 1: High-Signal Wins
    
    Goal: Learn "what obviously works"
    - Clear structural victories
    - High reward deltas (Δ > +0.3)
    - Low variance in outcomes
    
    Training Focus:
    - Build strong positive priors
    - Learn winning layout patterns
    - Establish baseline confidence
    """
    
    @staticmethod
    def get_loss_weight(difficulty: float) -> float:
        # High-signal wins should be learned with high confidence
        return 1.5 - (difficulty * 0.5)  # 1.0 to 1.5


class Stage2Strategy:
    """
    Stage 2: Clear Failures — The Inversion Bridge
    
    Goal: Learn "the negative space"
    - What obviously fails
    - Defines structural guardrails
    - Prevents Confident Failures
    
    Training Focus:
    - Build strong negative priors
    - Learn failure patterns (e.g., "Too much urgency kills conversion")
    - Create boundaries for Stage 1 learnings
    """
    
    @staticmethod
    def get_loss_weight(difficulty: float) -> float:
        # Clear failures are important for inversion learning
        return 1.4 - (difficulty * 0.4)  # 1.0 to 1.4
    
    @staticmethod
    def compute_inversion_signal(
        mutation_vector: np.ndarray,
        failure_delta: float
    ) -> np.ndarray:
        """
        Compute the "anti-direction" from a clear failure.
        
        If aggressive urgency (vector > 0.9) caused failure,
        the inversion signal teaches to avoid that extreme.
        """
        # Identify dimensions that were extreme
        extreme_mask = np.abs(mutation_vector) > 0.7
        
        # The inversion is proportional to failure severity
        inversion = -mutation_vector * extreme_mask * abs(failure_delta)
        
        return inversion


class Stage3Strategy:
    """
    Stage 3: Marginal Cases
    
    Goal: Learn "nuance and edge cases"
    - Small gains/losses (-0.3 < Δ < +0.3)
    - Context-dependent effectiveness
    - Fine-tuning the marginal edge
    
    Training Focus:
    - Learn conditional patterns
    - Understand context importance
    - Calibrate confidence appropriately
    """
    
    @staticmethod
    def get_loss_weight(difficulty: float) -> float:
        # Marginal cases need moderate weight
        return 1.0 - (difficulty * 0.3)  # 0.7 to 1.0


class Stage4Strategy:
    """
    Stage 4: Conflicting Signals — Market Drift
    
    Goal: Learn "when to be uncertain"
    - High variance outcomes
    - Previously successful strategies yielding mixed results
    - Signals that the market has shifted
    
    Training Focus:
    - Increase uncertainty parameter in ambiguous contexts
    - Learn to flag drift rather than commit
    - Prepare for re-exploration
    """
    
    @staticmethod
    def get_loss_weight(difficulty: float) -> float:
        # Conflicting signals get lower weight (learn uncertainty)
        return 0.8 - (difficulty * 0.3)  # 0.5 to 0.8
    
    @staticmethod
    def compute_uncertainty_increase(variance: float) -> float:
        """
        Compute how much to increase Thompson Sampling uncertainty.
        
        High variance in outcomes means the model should be less
        confident and more exploratory.
        """
        # Map variance to uncertainty multiplier
        # variance of 0.5 -> 1.5x uncertainty
        # variance of 1.0 -> 2.0x uncertainty
        return 1.0 + min(variance, 1.0)


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(
    title="Curriculum Learning",
    description="Pedagogical training for LAM",
    version="1.0.0"
)

redis_client: Optional[redis.Redis] = None
mongo_client: Optional[AsyncIOMotorClient] = None
trainer: Optional[CurriculumTrainer] = None


class TrainingResponse(BaseModel):
    status: str
    stages_completed: List[int]
    current_stage: int
    total_epochs: int


@app.on_event("startup")
async def startup():
    global redis_client, mongo_client
    redis_client = redis.from_url(REDIS_URL)
    mongo_client = AsyncIOMotorClient(MONGO_URL)


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()
    if mongo_client:
        mongo_client.close()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/train", response_model=TrainingResponse)
async def start_training(background_tasks: BackgroundTasks):
    """Start curriculum training."""
    # Import model
    from training_loop import LAMModel
    
    model = LAMModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    global trainer
    trainer = CurriculumTrainer(model, optimizer, redis_client, mongo_client)
    
    # Load samples
    samples = await trainer.load_samples()
    
    # Train with curriculum
    progress = await trainer.train_with_curriculum(samples)
    
    return TrainingResponse(
        status="completed",
        stages_completed=progress.stages_completed,
        current_stage=progress.current_stage.value,
        total_epochs=progress.total_epochs
    )


@app.get("/progress")
async def get_progress():
    """Get current curriculum progress."""
    data = await redis_client.get("curriculum_progress")
    if not data:
        return {"status": "no_training_run"}
    return json.loads(data)


@app.get("/audit")
async def get_audit_log(limit: int = 100):
    """Get recent audit log entries."""
    db = mongo_client.origin_os
    entries = await db.audit_log.find({}).sort("timestamp", -1).limit(limit).to_list(length=limit)
    
    # Convert ObjectId to string
    for entry in entries:
        entry["_id"] = str(entry["_id"])
    
    return {"entries": entries}


@app.get("/stage/{stage_id}/samples")
async def get_stage_samples(stage_id: int, limit: int = 100):
    """Get sample distribution for a curriculum stage."""
    classifier = CurriculumClassifier()
    
    db = mongo_client.origin_os
    samples = await db.training_samples.find({}).limit(limit * 4).to_list(length=limit * 4)
    
    stage = CurriculumStage(stage_id)
    stage_samples = []
    
    for doc in samples:
        sample_stage, difficulty = classifier.classify(doc)
        if sample_stage == stage:
            stage_samples.append({
                "sample_id": str(doc["_id"]),
                "actual_delta": doc.get("actual_delta", 0),
                "variance": doc.get("variance", 0),
                "difficulty_score": difficulty
            })
            if len(stage_samples) >= limit:
                break
    
    return {
        "stage": stage_id,
        "stage_name": stage.name,
        "sample_count": len(stage_samples),
        "samples": stage_samples
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8095)
