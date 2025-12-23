#!/usr/bin/env python3
"""
IMAGE GEN MCP — AI Image Generation for Origin OS
==================================================
Supports multiple providers:
- OpenAI DALL-E 3
- Stability AI (Stable Diffusion)
- Replicate (various models)
- Fal.ai
"""

import os
import json
import base64
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
FAL_API_KEY = os.getenv("FAL_API_KEY", "")

DEFAULT_PROVIDER = os.getenv("IMAGE_GEN_PROVIDER", "openai")

DATA_DIR = Path(os.getenv("IMAGE_DATA_DIR", "/data/images"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-gen")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Image Gen MCP",
    description="AI Image Generation for Origin OS",
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
# MODELS
# =============================================================================

class ImageGenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    provider: str = "openai"  # openai, stability, replicate, fal
    model: Optional[str] = None  # dall-e-3, sdxl, flux, etc.
    size: str = "1024x1024"  # 1024x1024, 1792x1024, 1024x1792
    quality: str = "standard"  # standard, hd
    style: Optional[str] = None  # vivid, natural
    n: int = 1  # Number of images

class ImageEditRequest(BaseModel):
    image_url: str
    prompt: str
    mask_url: Optional[str] = None
    provider: str = "openai"
    size: str = "1024x1024"

class ImageVariationRequest(BaseModel):
    image_url: str
    n: int = 1
    size: str = "1024x1024"

class ImageUpscaleRequest(BaseModel):
    image_url: str
    scale: int = 2  # 2x, 4x

class GeneratedImage(BaseModel):
    url: str
    local_path: Optional[str] = None
    prompt: str
    provider: str
    model: str
    created_at: str

# =============================================================================
# IMAGE GENERATION SERVICE
# =============================================================================

class ImageGenService:
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.generation_count = 0
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=120.0)
        return self.http_client
    
    async def generate(self, request: ImageGenRequest) -> List[GeneratedImage]:
        """Generate images using specified provider"""
        provider = request.provider.lower()
        
        if provider == "openai":
            return await self._generate_openai(request)
        elif provider == "stability":
            return await self._generate_stability(request)
        elif provider == "replicate":
            return await self._generate_replicate(request)
        elif provider == "fal":
            return await self._generate_fal(request)
        else:
            raise HTTPException(400, f"Unknown provider: {provider}")
    
    async def _generate_openai(self, request: ImageGenRequest) -> List[GeneratedImage]:
        """Generate using OpenAI DALL-E"""
        if not OPENAI_API_KEY:
            raise HTTPException(400, "OPENAI_API_KEY not configured")
        
        client = await self._get_client()
        
        model = request.model or "dall-e-3"
        
        response = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": model,
                "prompt": request.prompt,
                "n": request.n if model == "dall-e-2" else 1,
                "size": request.size,
                "quality": request.quality,
                "style": request.style or "vivid"
            }
        )
        
        response.raise_for_status()
        data = response.json()
        
        images = []
        for img_data in data.get("data", []):
            url = img_data.get("url", "")
            local_path = await self._save_image(url, request.prompt)
            
            images.append(GeneratedImage(
                url=url,
                local_path=str(local_path) if local_path else None,
                prompt=request.prompt,
                provider="openai",
                model=model,
                created_at=datetime.now().isoformat()
            ))
        
        self.generation_count += len(images)
        return images
    
    async def _generate_stability(self, request: ImageGenRequest) -> List[GeneratedImage]:
        """Generate using Stability AI"""
        if not STABILITY_API_KEY:
            raise HTTPException(400, "STABILITY_API_KEY not configured")
        
        client = await self._get_client()
        
        model = request.model or "stable-diffusion-xl-1024-v1-0"
        
        # Parse size
        width, height = map(int, request.size.split("x"))
        
        response = await client.post(
            f"https://api.stability.ai/v1/generation/{model}/text-to-image",
            headers={
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "text_prompts": [
                    {"text": request.prompt, "weight": 1.0},
                    {"text": request.negative_prompt or "", "weight": -1.0}
                ],
                "cfg_scale": 7,
                "width": width,
                "height": height,
                "samples": request.n,
                "steps": 30
            }
        )
        
        response.raise_for_status()
        data = response.json()
        
        images = []
        for artifact in data.get("artifacts", []):
            # Stability returns base64
            b64_data = artifact.get("base64", "")
            local_path = await self._save_base64_image(b64_data, request.prompt)
            
            images.append(GeneratedImage(
                url=f"/images/{local_path.name}" if local_path else "",
                local_path=str(local_path) if local_path else None,
                prompt=request.prompt,
                provider="stability",
                model=model,
                created_at=datetime.now().isoformat()
            ))
        
        self.generation_count += len(images)
        return images
    
    async def _generate_replicate(self, request: ImageGenRequest) -> List[GeneratedImage]:
        """Generate using Replicate"""
        if not REPLICATE_API_TOKEN:
            raise HTTPException(400, "REPLICATE_API_TOKEN not configured")
        
        client = await self._get_client()
        
        # Default to Flux
        model = request.model or "black-forest-labs/flux-schnell"
        
        # Create prediction
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "version": model,
                "input": {
                    "prompt": request.prompt,
                    "num_outputs": request.n,
                    "aspect_ratio": "1:1"
                }
            }
        )
        
        response.raise_for_status()
        prediction = response.json()
        
        # Poll for completion
        prediction_url = prediction.get("urls", {}).get("get")
        while prediction.get("status") not in ["succeeded", "failed"]:
            await asyncio.sleep(1)
            response = await client.get(
                prediction_url,
                headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"}
            )
            prediction = response.json()
        
        if prediction.get("status") == "failed":
            raise HTTPException(500, "Image generation failed")
        
        images = []
        for url in prediction.get("output", []):
            local_path = await self._save_image(url, request.prompt)
            
            images.append(GeneratedImage(
                url=url,
                local_path=str(local_path) if local_path else None,
                prompt=request.prompt,
                provider="replicate",
                model=model,
                created_at=datetime.now().isoformat()
            ))
        
        self.generation_count += len(images)
        return images
    
    async def _generate_fal(self, request: ImageGenRequest) -> List[GeneratedImage]:
        """Generate using Fal.ai"""
        if not FAL_API_KEY:
            raise HTTPException(400, "FAL_API_KEY not configured")
        
        client = await self._get_client()
        
        model = request.model or "fal-ai/flux/schnell"
        
        response = await client.post(
            f"https://fal.run/{model}",
            headers={
                "Authorization": f"Key {FAL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "prompt": request.prompt,
                "num_images": request.n,
                "image_size": request.size
            }
        )
        
        response.raise_for_status()
        data = response.json()
        
        images = []
        for img in data.get("images", []):
            url = img.get("url", "")
            local_path = await self._save_image(url, request.prompt)
            
            images.append(GeneratedImage(
                url=url,
                local_path=str(local_path) if local_path else None,
                prompt=request.prompt,
                provider="fal",
                model=model,
                created_at=datetime.now().isoformat()
            ))
        
        self.generation_count += len(images)
        return images
    
    async def _save_image(self, url: str, prompt: str) -> Optional[Path]:
        """Download and save image locally"""
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = prompt[:30].replace(" ", "_").replace("/", "_")
            filename = f"{timestamp}_{slug}.png"
            
            filepath = DATA_DIR / filename
            filepath.write_bytes(response.content)
            
            return filepath
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return None
    
    async def _save_base64_image(self, b64_data: str, prompt: str) -> Optional[Path]:
        """Save base64 image locally"""
        try:
            image_data = base64.b64decode(b64_data)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = prompt[:30].replace(" ", "_").replace("/", "_")
            filename = f"{timestamp}_{slug}.png"
            
            filepath = DATA_DIR / filename
            filepath.write_bytes(image_data)
            
            return filepath
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return None


# Global service
image_service = ImageGenService()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "generations": image_service.generation_count,
        "providers": {
            "openai": bool(OPENAI_API_KEY),
            "stability": bool(STABILITY_API_KEY),
            "replicate": bool(REPLICATE_API_TOKEN),
            "fal": bool(FAL_API_KEY)
        }
    }

@app.get("/info")
async def info():
    return {
        "service": "image-gen",
        "version": "1.0.0",
        "providers": ["openai", "stability", "replicate", "fal"],
        "models": {
            "openai": ["dall-e-3", "dall-e-2"],
            "stability": ["stable-diffusion-xl-1024-v1-0", "sd3"],
            "replicate": ["flux-schnell", "flux-pro", "sdxl"],
            "fal": ["flux/schnell", "flux/pro"]
        }
    }

@app.post("/generate")
async def generate_image(request: ImageGenRequest):
    """Generate images"""
    images = await image_service.generate(request)
    return {"images": [img.model_dump() for img in images]}

@app.get("/images/{filename}")
async def get_image(filename: str):
    """Serve generated image"""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(filepath)

@app.get("/images")
async def list_images(limit: int = 20):
    """List generated images"""
    images = sorted(DATA_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "images": [
            {"filename": img.name, "path": str(img)}
            for img in images[:limit]
        ]
    }

# MCP Tool Interface
@app.post("/mcp/tool")
async def mcp_tool(tool: str, params: Dict[str, Any]):
    """Execute MCP tool"""
    if tool == "generate_image":
        request = ImageGenRequest(**params)
        images = await image_service.generate(request)
        return {"images": [img.model_dump() for img in images]}
    else:
        raise HTTPException(400, f"Unknown tool: {tool}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
