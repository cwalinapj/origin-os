# Autonomic Ad-Engine — Master Technical Specification

> **Final Master Specification Document**
> 
> This document integrates divergence-gated routing, neural-drift detection,
> multi-objective reward shaping, and Governor-led transition to LAM-native authority.

---

## 1. System Architecture Overview

The engine operates as a closed-loop "organism" that bridges high-level generative AI (LLMs) and hard-nosed behavioral data.

### 1.1 Core Components

| Component | Technical Role |
|-----------|----------------|
| **AdsBanditRouter** | FastAPI-based gateway. Uses Thompson Sampling for traffic routing. |
| **Docker MCP** | The "Executioner." Spawns containerized landing pages with strict resource limits. |
| **SessionPersistence** | MongoDB store for `SessionNeuralState` (dwell, scroll, intent). |
| **LAM Forge** | The "Brain." Performs curriculum-based distillation of behavioral data. |
| **Governor** | The "Authority." Manages transition from LLM-generated to LAM-native code. |
| **GAdsSync** | The "Feedback Loop." Updates Google Ads via API based on internal winners. |

### 1.2 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AUTONOMIC AD-ENGINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Google Ads Traffic                                                        │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │   Router    │────▶│  Governor   │────▶│ Docker MCP  │                  │
│   │  (Thompson) │     │   (LoA)     │     │ (Containers)│                  │
│   └─────────────┘     └─────────────┘     └─────────────┘                  │
│         │                                        │                          │
│         │                                        ▼                          │
│         │                               ┌─────────────┐                     │
│         │                               │   Session   │                     │
│         │                               │   Neural    │                     │
│         │                               │   State     │                     │
│         │                               └─────────────┘                     │
│         │                                        │                          │
│         ▼                                        ▼                          │
│   ┌─────────────┐                       ┌─────────────┐                     │
│   │  GAdsSync   │◀──────────────────────│  LAM Forge  │                     │
│   │  (Feedback) │                       │   (Brain)   │                     │
│   └─────────────┘                       └─────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Learning & Evolution Loop

### 2.1 Divergence-Gated Evolution

We do not mutate pages on a whim. Regeneration is only triggered when behavioral divergence exceeds a hard threshold.

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Trigger Rule** | 7% | If mean behavioral score differs by >7% between variants |
| **Sample Floor** | 50 | Minimum events per arm before comparison |
| **Hierarchy** | Structural → Cosmetic | Layout/Offer resolved before Color/Copy |

```python
def maybe_regenerate(variants):
    """Divergence-gated regeneration check."""
    if min(v.sample_count for v in variants) < 50:
        return False  # Insufficient samples
    
    scores = [v.mean_score for v in variants]
    divergence = (max(scores) - min(scores)) / max(scores)
    
    return divergence > 0.07  # 7% threshold
```

### 2.2 Neural-Drift & Global Re-Exploration

The system streams behavioral embeddings into a shared layer. If the global "what users respond to" function shifts, the system forces a re-split across all sites.

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Neural Drift Threshold** | 0.15 | Magnitude of centroid shift in latent space |
| **Action** | FORCE_REEXPLORE | Invalidates priors and resets Thompson uncertainty |

```python
def check_neural_drift(current_centroid, baseline_centroid):
    """Detect global behavioral shift."""
    drift = np.linalg.norm(current_centroid - baseline_centroid)
    
    if drift > 0.15:
        return "FORCE_REEXPLORE"
    return "CONTINUE"
```

---

## 3. The LAM Forge: Advanced Distillation

### 3.1 Curriculum Learning Stages

The LAM is trained in increasing order of complexity to prevent overfitting on noise:

| Stage | Signal Type | Threshold | Purpose |
|-------|-------------|-----------|---------|
| **1** | High-signal wins | Δ > +10% | Learn clear winners |
| **2** | Clear failures | Δ < -10% | Inversion signals |
| **3** | Marginal gains | +2% to +5% | Fine-tuning |
| **4** | Conflicting signals | High variance | Drift detection |

```python
# lam/train/curriculum.py

def difficulty_bucket(sample):
    delta = abs(sample["outcome_after"]["mean_score_delta"])
    if delta >= 0.15:
        return 1  # strong signal
    if delta >= 0.08:
        return 2  # medium signal
    if delta >= 0.03:
        return 3  # weak signal
    return 4  # noisy / ambiguous
```

### 3.2 Vertical-Specialized Heads

To prevent "cross-domain contamination," the LAM uses a shared backbone with task-specific policy heads.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SHARED BACKBONE                          │
│                                                                 │
│  Learns "Visual Grammar":                                       │
│  • F-pattern reading                                            │
│  • Contrast → CTA clicks                                        │
│  • Behavioral physics (scroll, dwell)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ E-commerce│        │   B2B    │        │   SaaS   │
    │   Head   │        │   Head   │        │   Head   │
    │          │        │          │        │          │
    │ Urgency  │        │  Trust   │        │ Feature  │
    │ FOMO     │        │ Authority│        │  Trial   │
    └──────────┘        └──────────┘        └──────────┘
```

```python
# lam/model.py

class LAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()  # Shared backbone
        self.heads = nn.ModuleDict({
            "ecommerce": PolicyHead(),
            "b2b": PolicyHead(),
            "saas": PolicyHead(),
            "default": PolicyHead()
        })
    
    def forward(self, x, vertical):
        h = self.encoder(x)
        head = self.heads.get(vertical, self.heads["default"])
        return head(h)
```

---

## 4. Multi-Objective Reward Shaping

The engine optimizes a **Reward Vector** to ensure brand longevity and Google Ads account health.

### 4.1 Reward Function

$$R = 0.5(\text{Conv}) + 0.3(\text{Engage}) - 0.2(\text{Bounce}) - 0.1(\text{Drift})$$

| Component | Weight | Description |
|-----------|--------|-------------|
| **Conversion** | +0.5 | Hard sales data |
| **Engagement** | +0.3 | Dwell time and scroll depth (log-dampened) |
| **Bounce Penalty** | -0.2 | Protection against high CPC via Quality Score |
| **Semantic Drift** | -0.1 | Penalizes deviation from brand DNA |

```python
# lam/reward.py

def shaped_reward(sample):
    outcome = sample["outcome_after"]
    enforcement = sample["enforcement_distance"]
    
    reward = 0.0
    reward += 0.3 * outcome.get("mean_score_delta", 0.0)      # Engagement
    reward += 0.5 * outcome.get("conversion_delta", 0.0)      # Conversion
    reward -= 0.2 * max(outcome.get("bounce_delta", 0.0), 0)  # Bounce
    reward -= 0.1 * enforcement.get("semantic_drift", 0.0)    # Brand drift
    
    return reward
```

### 4.2 Anti-Dark-Pattern Safeguards

The reward function prevents optimization of dark patterns:

| Dark Pattern | Detection | Penalty |
|--------------|-----------|---------|
| Hidden close button | Opacity < 0.3 | -0.4 |
| Fake countdown | Urgency keyword density | -0.2 |
| Confirm shaming | Shame keyword match | -0.3 |
| Misleading copy | Scarcity regex patterns | -0.2 |

---

## 5. Governance & Autonomy (The LoA Scale)

The Governor manages the **Level of Autonomy (LoA)** for each vertical.

### 5.1 LoA Levels

| Level | Mode | Samples | Accuracy | Authority |
|-------|------|---------|----------|-----------|
| **1** | Shadow | < 100 | - | LLMs generate; LAM predicts in silence |
| **2** | Advisory | > 100 | > 70% | LLMs generate; LAM re-ranks or vetoes |
| **3** | Co-Pilot | > 500 | > 80% | LAM generates vector; LLM writes code |
| **4** | Governor | > 1000 | > 85% | LAM generates code; LLM audits only |

```python
# lam/governor/engine.py

class Governor:
    def __init__(self, vertical):
        self.vertical = vertical

    async def determine_loa(self):
        accuracy = await get_offline_accuracy(self.vertical)
        sample_count = await get_vertical_sample_count(self.vertical)

        if sample_count > 1000 and accuracy > 0.85:
            return 4  # FULL LAM AUTHORITY
        elif sample_count > 500 and accuracy > 0.80:
            return 3  # LAM GUIDED
        elif sample_count > 100 and accuracy > 0.70:
            return 2  # LAM ADVISORY
        return 1      # LLM DOMINANT

    async def get_generation_strategy(self):
        loa = await self.determine_loa()
        
        if loa == 4:
            return "LAM_NATIVE"
        elif loa == 3:
            return "LAM_VECTOR_LLM_CODE"
        else:
            return "LLM_ZERO_SHOT"
```

### 5.2 LAM-Native Mutation Output (LoA 4)

When Governor reaches LoA 4, the LAM outputs structured patches directly:

```json
{
  "operation": "node_swap",
  "target_id": "hero_section",
  "new_component": "high_trust_variant_v9",
  "css_overrides": {
    "background": "linear-gradient(to bottom, #f0f4f8, #ffffff)",
    "font-size": "1.2rem"
  },
  "metadata": {
    "intent": "Trust+0.8",
    "loa": 4
  }
}
```

---

## 6. Failure Inversion & Safety

When a "Confident Failure" occurs (high predicted win, actual loss), the system performs **Failure Inversion**.

### 6.1 Inversion Mechanics

| Type | Formula | Description |
|------|---------|-------------|
| **Structural** | $V_{inv} = -\eta \cdot V_{failed}$ | Direct negation scaled by learning rate |
| **Semantic** | Orthogonal Projection | Project away from failed direction |

```python
def compute_inversion_vector(failed_vector, failure_type="structural", eta=0.5):
    if failure_type == "structural":
        return -eta * failed_vector
    else:
        # Orthogonal projection for semantic
        norm = np.linalg.norm(failed_vector)
        if norm > 0:
            unit = failed_vector / norm
            return np.eye(len(failed_vector)) - np.outer(unit, unit)
        return np.zeros_like(failed_vector)
```

### 6.2 The Tombstone

A permanent record of failure used to penalize that direction in training:

```python
tombstone = {
    "id": "tomb_20241215_abc123",
    "site_id": "site_xyz",
    "vertical": "b2b",
    "mutation_vector": [...],
    "predicted_reward": 0.35,
    "actual_reward": -0.12,
    "surprise_delta": 0.47,
    "timestamp": "2024-12-15T10:30:00Z",
    "reason": "confident_failure"
}
```

### 6.3 Circuit Breaker

| Condition | Action |
|-----------|--------|
| > 10 failures/minute | Redirect to Global Static Baseline |
| > 3 consecutive failures | Pause regeneration for vertical |
| Surprise delta > 0.5 | Downgrade LoA to Level 2 |

---

## 7. Feature Schema

### 7.1 Input Features (11-dimensional)

```python
# lam/features.py

def featurize(sample):
    ctx = sample["context_features"]
    beh = sample["behavior_before"]
    lin = sample["lineage"]
    enf = sample["enforcement_distance"]
    
    features = [
        ctx.get("hour_of_day", 0) / 24.0,           # Time context
        ctx.get("traffic_entropy", 0.0),            # Traffic diversity
        beh.get("mean_score", 0.0),                 # Behavioral baseline
        beh.get("bounce_rate", 0.0),                # Exit rate
        beh.get("avg_scroll", 0.0),                 # Engagement depth
        beh.get("avg_dwell_time", 0.0) / 60.0,      # Time on page
        lin.get("ghost_weight", 0.0),               # Inherited failure
        lin.get("parent_lifetime_events", 0) / 1000.0,  # Container maturity
        enf.get("dom_complexity", 0.0),             # Page complexity
        enf.get("semantic_drift", 0.0),             # Brand deviation
        enf.get("visual_shift", 0.0),               # Layout change
    ]
    
    return torch.tensor(features, dtype=torch.float32)
```

---

## 8. File Structure

```
services/lam-forge/
├── lam/
│   ├── __init__.py
│   ├── model.py              # Per-Vertical LAM Heads
│   ├── reward.py             # Multi-Objective Reward Shaping
│   ├── features.py           # Feature extraction (11-dim)
│   ├── train/
│   │   ├── __init__.py
│   │   ├── curriculum.py     # Difficulty bucketing
│   │   └── train.py          # Curriculum-based training
│   ├── eval/
│   │   ├── __init__.py
│   │   └── replay.py         # Counterfactual replay
│   ├── governor/
│   │   ├── __init__.py
│   │   └── engine.py         # LoA controller
│   ├── checkpoints/          # Model weights
│   └── datasets/             # Training data (JSONL)
├── Dockerfile
├── requirements.txt
└── docker-compose services:
    - lam-forge-training (:----)
    - lam-forge-inference (:8060)
    - lam-forge-gads (:8070)
    - neural-monitor (:8030)
    - knowledge-mesh (:8040)
    - diff-enforcer (:8050)
    - evaluation-harness (:8090)
    - curriculum-trainer (:8095)
    - reward-shaping (:8100)
    - vertical-heads (:8110)
```

---

## 9. Quick Start

```bash
# Clone and start
cd origin-os-git
docker-compose up -d

# Check Governor status
curl http://localhost:8120/loa/ecommerce

# Run offline evaluation
python -m lam.eval.replay

# Start curriculum training
python -m lam.train.train lam/datasets/lam_dataset.jsonl

# Check reward shaping
curl -X POST http://localhost:8100/shape \
  -H "Content-Type: application/json" \
  -d '{"outcome_after": {"conversion_delta": 0.1, "mean_score_delta": 0.3}}'
```

---

## 10. Summary

The Autonomic Ad-Engine is a closed-loop system that:

1. **Ingests** traffic via Thompson Sampling router
2. **Observes** micro-behaviors via SessionNeuralState
3. **Triggers** regeneration at 7% divergence threshold
4. **Governs** autonomy level based on vertical maturity
5. **Learns** via curriculum-based LAM training
6. **Shapes** rewards to prevent dark patterns
7. **Syncs** winners back to Google Ads

The system progressively hands authority from external LLMs to the internal LAM as each vertical proves statistical reliability through offline counterfactual replay.

---

*Document Version: 1.0.0*
*Last Updated: December 2024*
