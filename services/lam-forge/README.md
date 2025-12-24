# LAM Forge
# =========

Large Action Model training and inference system for Origin OS.

## Components

### `inversion_math.py`
Mathematical framework for Failure Inversion:
- **Surprise Vector**: Calculates prediction error significance
- **Structural Inversion**: Geometric flip for layout failures
- **Behavioral Inversion**: Semantic projection for copy failures
- **Penalty Weight Decay**: Adaptive penalty based on vertical maturity

### `inference_spawner.py`
Container spawner for inference mode:
- Pre-warms container pools
- Caches structures by hash
- Manages container lifecycle
- Transitions sites from exploration to inference mode

### `training_loop.py`
Triplet loss training architecture:
- Processes inversion packets
- Integrates ghost memory
- Saves checkpoints
- Syncs with knowledge mesh

### `gads_deployment.py`
Google Ads API integration:
- Final URL promotion post-convergence
- Conversion adjustment uploads (behavioral signals)
- Sitelink mutation deployment
- Batch processing for API limits

### `metrics.py`
Prometheus metrics for monitoring:
- Surprise magnitude distribution
- Inversion events by type
- Training progress
- Container stats
- Google Ads sync status

## Quick Start

```bash
# Build
docker build -t origin-os/lam-forge .

# Run inference spawner
docker run -p 8060:8060 \
  -e REDIS_URL=redis://redis:6379 \
  origin-os/lam-forge

# Run training loop
docker run \
  -e REDIS_URL=redis://redis:6379 \
  -v /models:/models \
  --gpus all \
  origin-os/lam-forge python training_loop.py

# Run Google Ads deployment
docker run -p 8070:8070 \
  -e REDIS_URL=redis://redis:6379 \
  -e GOOGLE_ADS_CONFIG=/credentials/google-ads.yaml \
  -v /credentials:/credentials:ro \
  origin-os/lam-forge python -m uvicorn gads_deployment:app --host 0.0.0.0 --port 8070
```

## Key Concepts

### Failure Inversion
When the model predicts high reward but gets low actual reward, the error vector
contains valuable information. We amplify this signal with a penalty weight:

```
W(t) = W_max · e^(-λ · maturity) · min(surprise / threshold, 2.0)
```

### Ghost Memory
When variants are tombstoned (killed), we preserve their "ghost" — the directional
knowledge of what worked/failed — for contrastive learning.

### The Final 1000
After 1000 samples and convergence, sites transition from Thompson Sampling
to direct LAM inference for instant page generation.

## API Endpoints

### Inference Spawner (`:8060`)
- `POST /generate` - Generate page for user context
- `GET /containers` - List active containers
- `POST /transition/{site_id}` - Transition to inference mode

### Google Ads Deployment (`:8070`)
- `POST /promote` - Promote winning URL
- `POST /sitelink` - Deploy sitelink mutation
- `POST /batch-flush` - Flush behavioral scores
