#!/usr/bin/env python3
"""
LAM EXPERIMENT ENGINE — A/B Testing with Tool Chain Learning
=============================================================
The LAM agent uses this to:
1. Try different Figma plugins/tools for code generation
2. Deploy variants to device-specific containers
3. Measure conversions via GTM
4. Learn which tool chains work best for which site types
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(os.getenv("EXPERIMENT_DATA_DIR", "/data/experiments"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

FIGMA_URL = os.getenv("FIGMA_URL", "http://figma:8000")
VERCEL_URL = os.getenv("VERCEL_URL", "http://vercel:8000")
DEVICE_ROUTER_URL = os.getenv("DEVICE_ROUTER_URL", "http://device-router:8000")
GTM_INTERCEPT_URL = os.getenv("GTM_INTERCEPT_URL", "http://gtm-intercept:8000")
WEB_SCRAPER_URL = os.getenv("WEB_SCRAPER_URL", "http://web-scraper:8000")
IMAGE_COMPARE_URL = os.getenv("IMAGE_COMPARE_URL", "http://image-compare:8000")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lam-experiment")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="LAM Experiment Engine",
    description="A/B Testing with Tool Chain Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ENUMS & MODELS
# =============================================================================

class ToolChain(str, Enum):
    # Figma Plugins
    TELEPORT_HQ = "teleport_hq"           # Figma to React/Vue/HTML
    HTML_GENERATOR = "html_generator"      # Figma to HTML
    DHIWISE = "dhiwise"                    # Figma to Next.js/React/Flutter
    CODE_SNIPPET = "code_snippet"          # Component variants
    INSPECT_STYLES = "inspect_styles"      # CSS extraction
    FIGMA_TO_FLUTTER = "figma_to_flutter"
    FIGMA_TO_WORDPRESS = "figma_to_wordpress"
    
    # Direct methods
    FIGMA_API_REACT = "figma_api_react"
    FIGMA_API_HTML = "figma_api_html"
    FIGMA_API_TAILWIND = "figma_api_tailwind"
    
    # Clone methods
    WEB_SCRAPE_CLONE = "web_scrape_clone"
    PIXEL_PERFECT_CLONE = "pixel_perfect_clone"
    
    # AI-assisted
    V0_DEV = "v0_dev"  # Vercel v0
    CUSTOM = "custom"

class DeviceType(str, Enum):
    MOBILE_IOS = "mobile-ios"
    MOBILE_ANDROID = "mobile-android"
    TABLET_PORTRAIT = "tablet-portrait"
    TABLET_LANDSCAPE = "tablet-landscape"
    DESKTOP_SMALL = "desktop-small"
    DESKTOP_LARGE = "desktop-large"
    DESKTOP_ULTRA = "desktop-ultra"

class ConversionGoal(str, Enum):
    CLICK_NEXT_PAGE = "click_next_page"
    CLICK_TO_CALL = "click_to_call"
    FORM_SUBMIT = "form_submit"
    SCROLL_DEPTH_50 = "scroll_depth_50"
    SCROLL_DEPTH_100 = "scroll_depth_100"
    TIME_ON_PAGE_30S = "time_on_page_30s"
    TIME_ON_PAGE_60S = "time_on_page_60s"

class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    GENERATING = "generating"
    COMPARING = "comparing"
    DEPLOYING = "deploying"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class VariantSource(BaseModel):
    type: str  # figma, url, html
    figma_file_key: Optional[str] = None
    figma_node_id: Optional[str] = None
    source_url: Optional[str] = None
    raw_html: Optional[str] = None

class VariantConfig(BaseModel):
    variant_id: str
    name: str
    tool_chain: ToolChain
    source: VariantSource
    tool_params: Dict[str, Any] = {}
    device_types: List[DeviceType] = [DeviceType.DESKTOP_LARGE]

class DeviceVariantResult(BaseModel):
    device_type: DeviceType
    deployed_url: Optional[str] = None
    vercel_deployment_id: Optional[str] = None
    impressions: int = 0
    conversions: int = 0
    conversion_rate: float = 0.0

class VariantResult(BaseModel):
    variant_id: str
    generated_code: Optional[str] = None
    visual_accuracy_score: float = 0.0
    device_results: Dict[str, DeviceVariantResult] = {}
    total_impressions: int = 0
    total_conversions: int = 0
    overall_conversion_rate: float = 0.0
    human_ranking: Optional[int] = None
    human_notes: Optional[str] = None

class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    site_category: Optional[str] = None  # e-commerce, saas, local-business, etc.
    reference_source: VariantSource
    variants: List[VariantConfig]
    conversion_goals: List[ConversionGoal]
    device_types: List[DeviceType] = [DeviceType.DESKTOP_LARGE, DeviceType.MOBILE_IOS]
    traffic_split: Optional[List[float]] = None
    gtm_container_id: Optional[str] = None

class Experiment(BaseModel):
    experiment_id: str
    name: str
    description: Optional[str] = None
    site_category: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    reference_source: VariantSource
    variants: List[VariantConfig]
    conversion_goals: List[ConversionGoal]
    device_types: List[DeviceType]
    traffic_split: List[float]
    gtm_container_id: Optional[str] = None
    results: Dict[str, VariantResult] = {}
    created_at: str
    updated_at: str
    winner: Optional[str] = None
    winner_by_device: Dict[str, str] = {}

class HumanFeedback(BaseModel):
    variant_id: str
    ranking: int  # 1-10
    notes: Optional[str] = None
    visual_match_rating: Optional[int] = None  # 1-10 how close to original

# =============================================================================
# EXPERIMENT ENGINE
# =============================================================================

class ExperimentEngine:
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.experiments: Dict[str, Experiment] = {}
        self._load_experiments()
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=120.0)
        return self.http_client
    
    def _load_experiments(self):
        for exp_file in DATA_DIR.glob("exp_*.json"):
            try:
                data = json.loads(exp_file.read_text())
                exp = Experiment(**data)
                self.experiments[exp.experiment_id] = exp
            except Exception as e:
                logger.error(f"Failed to load {exp_file}: {e}")
    
    def _save_experiment(self, exp: Experiment):
        exp_file = DATA_DIR / f"exp_{exp.experiment_id}.json"
        exp_file.write_text(json.dumps(exp.model_dump(), indent=2))
    
    # =========================================================================
    # EXPERIMENT LIFECYCLE
    # =========================================================================
    
    async def create_experiment(self, config: ExperimentCreate) -> Experiment:
        exp_id = hashlib.md5(
            f"{config.name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        if not config.traffic_split:
            n = len(config.variants)
            config.traffic_split = [1.0 / n] * n
        
        now = datetime.now(timezone.utc).isoformat()
        
        exp = Experiment(
            experiment_id=exp_id,
            name=config.name,
            description=config.description,
            site_category=config.site_category,
            reference_source=config.reference_source,
            variants=config.variants,
            conversion_goals=config.conversion_goals,
            device_types=config.device_types,
            traffic_split=config.traffic_split,
            gtm_container_id=config.gtm_container_id,
            created_at=now,
            updated_at=now
        )
        
        self.experiments[exp_id] = exp
        self._save_experiment(exp)
        return exp
    
    async def generate_all_variants(self, experiment_id: str) -> Experiment:
        """Generate code for all variants using their tool chains"""
        
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")
        
        exp.status = ExperimentStatus.GENERATING
        exp.updated_at = datetime.now(timezone.utc).isoformat()
        
        for variant in exp.variants:
            try:
                code = await self._generate_variant_code(variant)
                
                if variant.variant_id not in exp.results:
                    exp.results[variant.variant_id] = VariantResult(
                        variant_id=variant.variant_id
                    )
                
                exp.results[variant.variant_id].generated_code = code
                logger.info(f"Generated {variant.variant_id} using {variant.tool_chain}")
                
            except Exception as e:
                logger.error(f"Generation failed for {variant.variant_id}: {e}")
                if variant.variant_id not in exp.results:
                    exp.results[variant.variant_id] = VariantResult(
                        variant_id=variant.variant_id
                    )
                exp.results[variant.variant_id].human_notes = f"Generation failed: {e}"
        
        self._save_experiment(exp)
        return exp
    
    async def _generate_variant_code(self, variant: VariantConfig) -> str:
        """Generate code using the specified tool chain"""
        
        client = await self._get_client()
        
        # Map tool chains to generation methods
        if variant.tool_chain in [
            ToolChain.FIGMA_API_REACT,
            ToolChain.FIGMA_API_HTML,
            ToolChain.FIGMA_API_TAILWIND
        ]:
            framework = {
                ToolChain.FIGMA_API_REACT: "react",
                ToolChain.FIGMA_API_HTML: "html",
                ToolChain.FIGMA_API_TAILWIND: "tailwind"
            }[variant.tool_chain]
            
            response = await client.post(
                f"{FIGMA_URL}/generate",
                json={
                    "file_key": variant.source.figma_file_key,
                    "node_id": variant.source.figma_node_id,
                    "framework": framework
                }
            )
            response.raise_for_status()
            return response.json().get("code", "")
        
        elif variant.tool_chain == ToolChain.WEB_SCRAPE_CLONE:
            response = await client.post(
                f"{WEB_SCRAPER_URL}/scrape",
                json={
                    "url": variant.source.source_url,
                    "formats": ["html"]
                }
            )
            response.raise_for_status()
            return response.json().get("html", "")
        
        elif variant.tool_chain == ToolChain.V0_DEV:
            # Use v0.dev API if available
            return await self._generate_with_v0(variant)
        
        elif variant.tool_chain == ToolChain.CUSTOM:
            return variant.source.raw_html or ""
        
        else:
            # Default: use Figma API with React
            response = await client.post(
                f"{FIGMA_URL}/generate",
                json={
                    "file_key": variant.source.figma_file_key,
                    "node_id": variant.source.figma_node_id,
                    "framework": "react"
                }
            )
            response.raise_for_status()
            return response.json().get("code", "")
    
    async def _generate_with_v0(self, variant: VariantConfig) -> str:
        """Generate using Vercel v0"""
        # v0.dev integration would go here
        # For now, fall back to Figma API
        client = await self._get_client()
        response = await client.post(
            f"{FIGMA_URL}/generate",
            json={
                "file_key": variant.source.figma_file_key,
                "node_id": variant.source.figma_node_id,
                "framework": "react"
            }
        )
        response.raise_for_status()
        return response.json().get("code", "")
    
    async def deploy_variants(self, experiment_id: str) -> Experiment:
        """Deploy all variants to Vercel as device-specific containers"""
        
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")
        
        exp.status = ExperimentStatus.DEPLOYING
        client = await self._get_client()
        
        for variant in exp.variants:
            result = exp.results.get(variant.variant_id)
            if not result or not result.generated_code:
                continue
            
            # Deploy to each device type
            for device_type in exp.device_types:
                try:
                    # Create Vercel project for this variant+device
                    project_name = f"{exp.experiment_id}-{variant.variant_id}-{device_type.value}"
                    
                    # Deploy to Vercel
                    deploy_response = await client.post(
                        f"{VERCEL_URL}/deploy",
                        json={
                            "project_id": project_name,
                            "target": "production"
                        }
                    )
                    
                    if deploy_response.status_code == 200:
                        deploy_data = deploy_response.json()
                        
                        if device_type.value not in result.device_results:
                            result.device_results[device_type.value] = DeviceVariantResult(
                                device_type=device_type
                            )
                        
                        result.device_results[device_type.value].deployed_url = \
                            f"https://{project_name}.vercel.app"
                        result.device_results[device_type.value].vercel_deployment_id = \
                            deploy_data.get("id")
                        
                        # Register with Device Router
                        await client.post(
                            f"{DEVICE_ROUTER_URL}/sites/{exp.experiment_id}/containers",
                            params={
                                "device_type": device_type.value,
                                "container_url": f"https://{project_name}.vercel.app"
                            }
                        )
                
                except Exception as e:
                    logger.error(f"Deploy failed for {variant.variant_id}/{device_type}: {e}")
        
        exp.status = ExperimentStatus.RUNNING
        exp.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_experiment(exp)
        
        return exp
    
    # =========================================================================
    # TRACKING
    # =========================================================================
    
    async def record_impression(
        self,
        experiment_id: str,
        variant_id: str,
        device_type: str
    ):
        exp = self.experiments.get(experiment_id)
        if not exp or variant_id not in exp.results:
            return
        
        result = exp.results[variant_id]
        result.total_impressions += 1
        
        if device_type in result.device_results:
            result.device_results[device_type].impressions += 1
            dr = result.device_results[device_type]
            if dr.impressions > 0:
                dr.conversion_rate = dr.conversions / dr.impressions
        
        if result.total_impressions > 0:
            result.overall_conversion_rate = result.total_conversions / result.total_impressions
        
        self._save_experiment(exp)
    
    async def record_conversion(
        self,
        experiment_id: str,
        variant_id: str,
        device_type: str,
        goal: ConversionGoal
    ):
        exp = self.experiments.get(experiment_id)
        if not exp or variant_id not in exp.results:
            return
        
        result = exp.results[variant_id]
        result.total_conversions += 1
        
        if device_type in result.device_results:
            result.device_results[device_type].conversions += 1
            dr = result.device_results[device_type]
            if dr.impressions > 0:
                dr.conversion_rate = dr.conversions / dr.impressions
        
        if result.total_impressions > 0:
            result.overall_conversion_rate = result.total_conversions / result.total_impressions
        
        self._save_experiment(exp)
    
    # =========================================================================
    # HUMAN FEEDBACK
    # =========================================================================
    
    async def submit_feedback(
        self,
        experiment_id: str,
        feedback: HumanFeedback
    ) -> Experiment:
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")
        
        if feedback.variant_id in exp.results:
            exp.results[feedback.variant_id].human_ranking = feedback.ranking
            exp.results[feedback.variant_id].human_notes = feedback.notes
            
            if feedback.visual_match_rating:
                exp.results[feedback.variant_id].visual_accuracy_score = \
                    feedback.visual_match_rating * 10  # Convert 1-10 to 0-100
        
        exp.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_experiment(exp)
        
        return exp
    
    # =========================================================================
    # ANALYSIS & LEARNING
    # =========================================================================
    
    async def complete_experiment(self, experiment_id: str) -> Experiment:
        """Complete experiment and determine winners"""
        
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")
        
        # Calculate overall winner
        best_score = 0
        winner = None
        
        for variant_id, result in exp.results.items():
            # Score = 40% conversion + 30% visual accuracy + 30% human ranking
            score = (
                result.overall_conversion_rate * 0.4 +
                (result.visual_accuracy_score / 100) * 0.3 +
                ((result.human_ranking or 5) / 10) * 0.3
            )
            
            if score > best_score:
                best_score = score
                winner = variant_id
        
        exp.winner = winner
        
        # Calculate winner by device
        for device_type in exp.device_types:
            best_device_score = 0
            device_winner = None
            
            for variant_id, result in exp.results.items():
                if device_type.value in result.device_results:
                    dr = result.device_results[device_type.value]
                    device_score = dr.conversion_rate
                    
                    if device_score > best_device_score:
                        best_device_score = device_score
                        device_winner = variant_id
            
            if device_winner:
                exp.winner_by_device[device_type.value] = device_winner
        
        exp.status = ExperimentStatus.COMPLETED
        exp.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_experiment(exp)
        
        # Log training data
        await self._log_training_data(exp)
        
        return exp
    
    async def _log_training_data(self, exp: Experiment):
        """Log experiment results for LAM training"""
        
        training_file = DATA_DIR / "training_data.jsonl"
        
        for variant_id, result in exp.results.items():
            variant = next(
                (v for v in exp.variants if v.variant_id == variant_id),
                None
            )
            
            if not variant:
                continue
            
            for device_type, device_result in result.device_results.items():
                training_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "experiment_id": exp.experiment_id,
                    "site_category": exp.site_category,
                    "variant_id": variant_id,
                    "tool_chain": variant.tool_chain.value,
                    "tool_params": variant.tool_params,
                    "device_type": device_type,
                    "impressions": device_result.impressions,
                    "conversions": device_result.conversions,
                    "conversion_rate": device_result.conversion_rate,
                    "visual_accuracy": result.visual_accuracy_score,
                    "human_ranking": result.human_ranking,
                    "is_overall_winner": variant_id == exp.winner,
                    "is_device_winner": exp.winner_by_device.get(device_type) == variant_id
                }
                
                with open(training_file, "a") as f:
                    f.write(json.dumps(training_entry) + "\n")
    
    def get_tool_recommendations(
        self,
        site_category: Optional[str] = None,
        device_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get tool chain recommendations based on training data"""
        
        training_file = DATA_DIR / "training_data.jsonl"
        if not training_file.exists():
            return {"message": "No training data yet", "recommendations": []}
        
        tool_stats: Dict[str, Dict] = {}
        
        with open(training_file) as f:
            for line in f:
                entry = json.loads(line)
                
                # Filter by criteria
                if site_category and entry.get("site_category") != site_category:
                    continue
                if device_type and entry.get("device_type") != device_type:
                    continue
                
                tool = entry["tool_chain"]
                
                if tool not in tool_stats:
                    tool_stats[tool] = {
                        "experiments": 0,
                        "wins": 0,
                        "device_wins": 0,
                        "conversion_rates": [],
                        "human_rankings": []
                    }
                
                stats = tool_stats[tool]
                stats["experiments"] += 1
                
                if entry.get("is_overall_winner"):
                    stats["wins"] += 1
                if entry.get("is_device_winner"):
                    stats["device_wins"] += 1
                
                stats["conversion_rates"].append(entry.get("conversion_rate", 0))
                if entry.get("human_ranking"):
                    stats["human_rankings"].append(entry["human_ranking"])
        
        # Calculate aggregates
        recommendations = []
        for tool, stats in tool_stats.items():
            avg_conversion = sum(stats["conversion_rates"]) / len(stats["conversion_rates"]) if stats["conversion_rates"] else 0
            avg_ranking = sum(stats["human_rankings"]) / len(stats["human_rankings"]) if stats["human_rankings"] else 5
            win_rate = stats["wins"] / stats["experiments"] if stats["experiments"] > 0 else 0
            
            recommendations.append({
                "tool_chain": tool,
                "experiments": stats["experiments"],
                "win_rate": round(win_rate, 3),
                "avg_conversion_rate": round(avg_conversion, 4),
                "avg_human_ranking": round(avg_ranking, 1),
                "score": round(win_rate * 0.5 + avg_conversion * 0.3 + (avg_ranking / 10) * 0.2, 3)
            })
        
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "site_category": site_category,
            "device_type": device_type,
            "recommendations": recommendations,
            "top_recommendation": recommendations[0]["tool_chain"] if recommendations else None
        }


# Global engine
engine = ExperimentEngine()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "experiments": len(engine.experiments)}

# CRUD
@app.post("/experiments")
async def create_experiment(config: ExperimentCreate):
    exp = await engine.create_experiment(config)
    return exp.model_dump()

@app.get("/experiments")
async def list_experiments(status: Optional[str] = None):
    exps = list(engine.experiments.values())
    if status:
        exps = [e for e in exps if e.status.value == status]
    return {"experiments": [e.model_dump() for e in exps]}

@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    exp = engine.experiments.get(experiment_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return exp.model_dump()

# Workflow
@app.post("/experiments/{experiment_id}/generate")
async def generate_variants(experiment_id: str):
    exp = await engine.generate_all_variants(experiment_id)
    return exp.model_dump()

@app.post("/experiments/{experiment_id}/deploy")
async def deploy_variants(experiment_id: str):
    exp = await engine.deploy_variants(experiment_id)
    return exp.model_dump()

@app.post("/experiments/{experiment_id}/complete")
async def complete_experiment(experiment_id: str):
    exp = await engine.complete_experiment(experiment_id)
    return exp.model_dump()

# Tracking
@app.post("/experiments/{experiment_id}/impression")
async def record_impression(experiment_id: str, variant_id: str, device_type: str):
    await engine.record_impression(experiment_id, variant_id, device_type)
    return {"status": "recorded"}

@app.post("/experiments/{experiment_id}/conversion")
async def record_conversion(
    experiment_id: str,
    variant_id: str,
    device_type: str,
    goal: ConversionGoal
):
    await engine.record_conversion(experiment_id, variant_id, device_type, goal)
    return {"status": "recorded"}

# Feedback
@app.post("/experiments/{experiment_id}/feedback")
async def submit_feedback(experiment_id: str, feedback: HumanFeedback):
    exp = await engine.submit_feedback(experiment_id, feedback)
    return exp.model_dump()

# Recommendations
@app.get("/recommendations")
async def get_recommendations(
    site_category: Optional[str] = None,
    device_type: Optional[str] = None
):
    return engine.get_tool_recommendations(site_category, device_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
