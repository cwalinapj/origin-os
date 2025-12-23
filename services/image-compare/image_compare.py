#!/usr/bin/env python3
"""
IMAGE COMPARE MCP — Visual Accuracy Scoring
============================================
Compares original site screenshots to generated containers.
Uses multiple algorithms for accurate similarity scoring.
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import base64
import io

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Try to import image processing libs
try:
    from PIL import Image
    import numpy as np
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-compare")

DATA_DIR = Path(os.getenv("IMAGE_COMPARE_DATA_DIR", "/data/image-compare"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Image Compare MCP", version="1.0.0")

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

class CompareRequest(BaseModel):
    image1_url: Optional[str] = None
    image2_url: Optional[str] = None
    image1_base64: Optional[str] = None
    image2_base64: Optional[str] = None

class CompareResult(BaseModel):
    similarity_score: float  # 0-100
    pixel_match: float  # 0-100
    structural_similarity: float  # 0-100
    color_similarity: float  # 0-100
    layout_similarity: float  # 0-100
    confidence: float  # 0-1

class ScreenshotRequest(BaseModel):
    url: str
    viewport_width: int = 1920
    viewport_height: int = 1080
    full_page: bool = False
    device_type: Optional[str] = None

# =============================================================================
# IMAGE COMPARISON ENGINE
# =============================================================================

class ImageCompareEngine:
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=60.0)
        return self.http_client
    
    async def load_image(self, url: str = None, base64_data: str = None) -> Optional["Image.Image"]:
        """Load image from URL or base64"""
        if not PILLOW_AVAILABLE:
            return None
        
        try:
            if base64_data:
                # Remove data URL prefix if present
                if "," in base64_data:
                    base64_data = base64_data.split(",")[1]
                image_data = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_data))
            
            elif url:
                client = await self._get_client()
                response = await client.get(url)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
        
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
        
        return None
    
    def compare_images(self, img1: "Image.Image", img2: "Image.Image") -> CompareResult:
        """Compare two images and return similarity scores"""
        
        if not PILLOW_AVAILABLE:
            return CompareResult(
                similarity_score=0,
                pixel_match=0,
                structural_similarity=0,
                color_similarity=0,
                layout_similarity=0,
                confidence=0
            )
        
        # Resize to same dimensions
        target_size = (max(img1.width, img2.width), max(img1.height, img2.height))
        img1_resized = img1.resize(target_size, Image.Resampling.LANCZOS)
        img2_resized = img2.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB
        if img1_resized.mode != 'RGB':
            img1_resized = img1_resized.convert('RGB')
        if img2_resized.mode != 'RGB':
            img2_resized = img2_resized.convert('RGB')
        
        # Convert to numpy arrays
        arr1 = np.array(img1_resized)
        arr2 = np.array(img2_resized)
        
        # 1. Pixel-level comparison
        pixel_match = self._pixel_similarity(arr1, arr2)
        
        # 2. Structural similarity (simplified SSIM)
        structural_similarity = self._structural_similarity(arr1, arr2)
        
        # 3. Color histogram similarity
        color_similarity = self._color_similarity(img1_resized, img2_resized)
        
        # 4. Edge/layout similarity
        layout_similarity = self._layout_similarity(arr1, arr2)
        
        # Combined score (weighted average)
        similarity_score = (
            pixel_match * 0.2 +
            structural_similarity * 0.3 +
            color_similarity * 0.2 +
            layout_similarity * 0.3
        )
        
        # Confidence based on image sizes
        size_ratio = min(img1.width * img1.height, img2.width * img2.height) / max(img1.width * img1.height, img2.width * img2.height)
        confidence = min(1.0, size_ratio + 0.5)
        
        return CompareResult(
            similarity_score=round(similarity_score, 2),
            pixel_match=round(pixel_match, 2),
            structural_similarity=round(structural_similarity, 2),
            color_similarity=round(color_similarity, 2),
            layout_similarity=round(layout_similarity, 2),
            confidence=round(confidence, 3)
        )
    
    def _pixel_similarity(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate pixel-level similarity"""
        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        max_diff = 255 * 3  # RGB
        similarity = 1 - (np.mean(diff) / max_diff)
        return similarity * 100
    
    def _structural_similarity(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Simplified structural similarity (SSIM-like)"""
        # Convert to grayscale
        gray1 = np.mean(arr1, axis=2)
        gray2 = np.mean(arr2, axis=2)
        
        # Calculate means and variances
        mu1 = np.mean(gray1)
        mu2 = np.mean(gray2)
        
        sigma1_sq = np.var(gray1)
        sigma2_sq = np.var(gray2)
        sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
        
        # SSIM formula constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return max(0, ssim) * 100
    
    def _color_similarity(self, img1: "Image.Image", img2: "Image.Image") -> float:
        """Compare color histograms"""
        # Get histograms for each channel
        hist1_r = img1.histogram()[:256]
        hist1_g = img1.histogram()[256:512]
        hist1_b = img1.histogram()[512:768]
        
        hist2_r = img2.histogram()[:256]
        hist2_g = img2.histogram()[256:512]
        hist2_b = img2.histogram()[512:768]
        
        # Normalize histograms
        total1 = sum(hist1_r) or 1
        total2 = sum(hist2_r) or 1
        
        hist1_r = [x / total1 for x in hist1_r]
        hist1_g = [x / total1 for x in hist1_g]
        hist1_b = [x / total1 for x in hist1_b]
        
        hist2_r = [x / total2 for x in hist2_r]
        hist2_g = [x / total2 for x in hist2_g]
        hist2_b = [x / total2 for x in hist2_b]
        
        # Calculate histogram intersection
        def histogram_intersection(h1, h2):
            return sum(min(a, b) for a, b in zip(h1, h2))
        
        sim_r = histogram_intersection(hist1_r, hist2_r)
        sim_g = histogram_intersection(hist1_g, hist2_g)
        sim_b = histogram_intersection(hist1_b, hist2_b)
        
        return ((sim_r + sim_g + sim_b) / 3) * 100
    
    def _layout_similarity(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Compare edge maps for layout similarity"""
        # Simple edge detection using gradient
        gray1 = np.mean(arr1, axis=2)
        gray2 = np.mean(arr2, axis=2)
        
        # Sobel-like gradient (simplified)
        def simple_edges(img):
            gx = np.abs(np.diff(img, axis=1))
            gy = np.abs(np.diff(img, axis=0))
            # Pad to same size
            gx = np.pad(gx, ((0, 0), (0, 1)), mode='constant')
            gy = np.pad(gy, ((0, 1), (0, 0)), mode='constant')
            return np.sqrt(gx ** 2 + gy ** 2)
        
        edges1 = simple_edges(gray1)
        edges2 = simple_edges(gray2)
        
        # Normalize
        edges1 = edges1 / (np.max(edges1) or 1)
        edges2 = edges2 / (np.max(edges2) or 1)
        
        # Calculate similarity
        diff = np.abs(edges1 - edges2)
        similarity = 1 - np.mean(diff)
        
        return similarity * 100
    
    async def take_screenshot(self, request: ScreenshotRequest) -> Optional[bytes]:
        """Take screenshot of a URL (requires external service)"""
        # This would integrate with a screenshot service like:
        # - Puppeteer/Playwright service
        # - screenshotapi.net
        # - urlbox.io
        # For now, return None
        logger.warning("Screenshot service not configured")
        return None


engine = ImageCompareEngine()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "pillow_available": PILLOW_AVAILABLE,
        "numpy_available": PILLOW_AVAILABLE  # numpy comes with pillow usually
    }

@app.post("/compare")
async def compare_images(request: CompareRequest):
    """Compare two images and return similarity scores"""
    
    if not PILLOW_AVAILABLE:
        raise HTTPException(500, "Image processing not available (install Pillow)")
    
    # Load images
    img1 = await engine.load_image(request.image1_url, request.image1_base64)
    img2 = await engine.load_image(request.image2_url, request.image2_base64)
    
    if img1 is None:
        raise HTTPException(400, "Failed to load image 1")
    if img2 is None:
        raise HTTPException(400, "Failed to load image 2")
    
    result = engine.compare_images(img1, img2)
    return result.model_dump()

@app.post("/compare/upload")
async def compare_uploaded_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    """Compare two uploaded images"""
    
    if not PILLOW_AVAILABLE:
        raise HTTPException(500, "Image processing not available (install Pillow)")
    
    img1 = Image.open(io.BytesIO(await image1.read()))
    img2 = Image.open(io.BytesIO(await image2.read()))
    
    result = engine.compare_images(img1, img2)
    return result.model_dump()

@app.post("/screenshot")
async def take_screenshot(request: ScreenshotRequest):
    """Take screenshot of a URL"""
    screenshot = await engine.take_screenshot(request)
    
    if screenshot is None:
        raise HTTPException(501, "Screenshot service not configured")
    
    return {
        "url": request.url,
        "image_base64": base64.b64encode(screenshot).decode()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
