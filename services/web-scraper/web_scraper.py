#!/usr/bin/env python3
"""
WEB SCRAPER MCP — Web Data Extraction for Origin OS
====================================================
Supports:
- Firecrawl API (primary)
- Direct scraping fallback
- JavaScript rendering
- Structured data extraction
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v1"

DATA_DIR = Path(os.getenv("SCRAPER_DATA_DIR", "/data/scraper"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-scraper")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Web Scraper MCP",
    description="Web Data Extraction for Origin OS",
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

class ScrapeRequest(BaseModel):
    url: str
    formats: List[str] = ["markdown", "html"]  # markdown, html, links, screenshot
    only_main_content: bool = True
    include_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None
    wait_for: Optional[str] = None  # CSS selector to wait for
    timeout: int = 30000

class CrawlRequest(BaseModel):
    url: str
    max_depth: int = 2
    max_pages: int = 10
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    formats: List[str] = ["markdown"]

class MapRequest(BaseModel):
    url: str
    search: Optional[str] = None  # Search query to filter URLs
    limit: int = 100

class ExtractRequest(BaseModel):
    url: str
    schema: Dict[str, Any]  # JSON schema for extraction
    prompt: Optional[str] = None

class BatchScrapeRequest(BaseModel):
    urls: List[str]
    formats: List[str] = ["markdown"]
    only_main_content: bool = True

# =============================================================================
# WEB SCRAPER SERVICE
# =============================================================================

class WebScraperService:
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.scrape_count = 0
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=60.0)
        return self.http_client
    
    async def scrape(self, request: ScrapeRequest) -> Dict[str, Any]:
        """Scrape a single URL"""
        
        if FIRECRAWL_API_KEY:
            return await self._scrape_firecrawl(request)
        else:
            return await self._scrape_direct(request)
    
    async def _scrape_firecrawl(self, request: ScrapeRequest) -> Dict[str, Any]:
        """Scrape using Firecrawl API"""
        client = await self._get_client()
        
        payload = {
            "url": request.url,
            "formats": request.formats,
            "onlyMainContent": request.only_main_content,
            "timeout": request.timeout
        }
        
        if request.include_tags:
            payload["includeTags"] = request.include_tags
        if request.exclude_tags:
            payload["excludeTags"] = request.exclude_tags
        if request.wait_for:
            payload["waitFor"] = request.wait_for
        
        response = await client.post(
            f"{FIRECRAWL_BASE_URL}/scrape",
            headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
            json=payload
        )
        
        response.raise_for_status()
        data = response.json()
        
        self.scrape_count += 1
        
        # Save to cache
        await self._cache_result(request.url, data)
        
        return data.get("data", {})
    
    async def _scrape_direct(self, request: ScrapeRequest) -> Dict[str, Any]:
        """Direct scraping fallback (no JS rendering)"""
        client = await self._get_client()
        
        response = await client.get(
            request.url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; OriginOS/1.0)"
            }
        )
        
        response.raise_for_status()
        html = response.text
        
        self.scrape_count += 1
        
        result = {"url": request.url}
        
        if "html" in request.formats:
            result["html"] = html
        
        if "markdown" in request.formats:
            result["markdown"] = self._html_to_markdown(html)
        
        if "links" in request.formats:
            result["links"] = self._extract_links(html, request.url)
        
        return result
    
    async def crawl(self, request: CrawlRequest) -> Dict[str, Any]:
        """Crawl multiple pages"""
        
        if not FIRECRAWL_API_KEY:
            raise HTTPException(400, "Crawling requires FIRECRAWL_API_KEY")
        
        client = await self._get_client()
        
        # Start crawl job
        payload = {
            "url": request.url,
            "maxDepth": request.max_depth,
            "limit": request.max_pages,
            "scrapeOptions": {
                "formats": request.formats
            }
        }
        
        if request.include_patterns:
            payload["includePaths"] = request.include_patterns
        if request.exclude_patterns:
            payload["excludePaths"] = request.exclude_patterns
        
        response = await client.post(
            f"{FIRECRAWL_BASE_URL}/crawl",
            headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
            json=payload
        )
        
        response.raise_for_status()
        job = response.json()
        
        # Poll for completion
        job_id = job.get("id")
        if not job_id:
            return job
        
        while True:
            status_response = await client.get(
                f"{FIRECRAWL_BASE_URL}/crawl/{job_id}",
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"}
            )
            status = status_response.json()
            
            if status.get("status") == "completed":
                self.scrape_count += len(status.get("data", []))
                return status
            elif status.get("status") == "failed":
                raise HTTPException(500, "Crawl failed")
            
            import asyncio
            await asyncio.sleep(2)
    
    async def map_site(self, request: MapRequest) -> List[str]:
        """Map all URLs on a site"""
        
        if not FIRECRAWL_API_KEY:
            raise HTTPException(400, "Site mapping requires FIRECRAWL_API_KEY")
        
        client = await self._get_client()
        
        payload = {
            "url": request.url,
            "limit": request.limit
        }
        
        if request.search:
            payload["search"] = request.search
        
        response = await client.post(
            f"{FIRECRAWL_BASE_URL}/map",
            headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
            json=payload
        )
        
        response.raise_for_status()
        data = response.json()
        
        return data.get("links", [])
    
    async def extract(self, request: ExtractRequest) -> Dict[str, Any]:
        """Extract structured data from URL"""
        
        if not FIRECRAWL_API_KEY:
            raise HTTPException(400, "Extraction requires FIRECRAWL_API_KEY")
        
        client = await self._get_client()
        
        payload = {
            "url": request.url,
            "schema": request.schema
        }
        
        if request.prompt:
            payload["prompt"] = request.prompt
        
        response = await client.post(
            f"{FIRECRAWL_BASE_URL}/extract",
            headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
            json=payload
        )
        
        response.raise_for_status()
        return response.json().get("data", {})
    
    async def batch_scrape(self, request: BatchScrapeRequest) -> List[Dict[str, Any]]:
        """Scrape multiple URLs"""
        
        if not FIRECRAWL_API_KEY:
            # Fall back to sequential scraping
            results = []
            for url in request.urls:
                try:
                    result = await self.scrape(ScrapeRequest(
                        url=url,
                        formats=request.formats,
                        only_main_content=request.only_main_content
                    ))
                    results.append(result)
                except Exception as e:
                    results.append({"url": url, "error": str(e)})
            return results
        
        client = await self._get_client()
        
        # Start batch job
        payload = {
            "urls": request.urls,
            "formats": request.formats,
            "onlyMainContent": request.only_main_content
        }
        
        response = await client.post(
            f"{FIRECRAWL_BASE_URL}/batch/scrape",
            headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
            json=payload
        )
        
        response.raise_for_status()
        job = response.json()
        
        # Poll for completion
        job_id = job.get("id")
        if not job_id:
            return []
        
        while True:
            status_response = await client.get(
                f"{FIRECRAWL_BASE_URL}/batch/scrape/{job_id}",
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"}
            )
            status = status_response.json()
            
            if status.get("status") == "completed":
                self.scrape_count += len(status.get("data", []))
                return status.get("data", [])
            elif status.get("status") == "failed":
                raise HTTPException(500, "Batch scrape failed")
            
            import asyncio
            await asyncio.sleep(2)
    
    async def _cache_result(self, url: str, data: Dict):
        """Cache scrape result"""
        try:
            slug = re.sub(r'[^\w]', '_', url)[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_file = DATA_DIR / f"{timestamp}_{slug}.json"
            cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Cache write failed: {e}")
    
    def _html_to_markdown(self, html: str) -> str:
        """Simple HTML to Markdown conversion"""
        # Very basic conversion
        text = html
        text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n', text, flags=re.DOTALL)
        text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n', text, flags=re.DOTALL)
        text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n', text, flags=re.DOTALL)
        text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
        text = re.sub(r'<br[^>]*>', '\n', text)
        text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        return text.strip()
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML"""
        links = re.findall(r'href=["\']([^"\']+)["\']', html)
        # Normalize URLs
        from urllib.parse import urljoin
        return list(set(urljoin(base_url, link) for link in links))


# Global service
scraper_service = WebScraperService()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "scrape_count": scraper_service.scrape_count,
        "firecrawl_enabled": bool(FIRECRAWL_API_KEY)
    }

@app.post("/scrape")
async def scrape(request: ScrapeRequest):
    """Scrape a URL"""
    result = await scraper_service.scrape(request)
    return result

@app.post("/crawl")
async def crawl(request: CrawlRequest):
    """Crawl a website"""
    result = await scraper_service.crawl(request)
    return result

@app.post("/map")
async def map_site(request: MapRequest):
    """Map all URLs on a site"""
    urls = await scraper_service.map_site(request)
    return {"urls": urls, "count": len(urls)}

@app.post("/extract")
async def extract(request: ExtractRequest):
    """Extract structured data"""
    result = await scraper_service.extract(request)
    return result

@app.post("/batch")
async def batch_scrape(request: BatchScrapeRequest):
    """Scrape multiple URLs"""
    results = await scraper_service.batch_scrape(request)
    return {"results": results}

# MCP Tool Interface
@app.post("/mcp/tool")
async def mcp_tool(tool: str, params: Dict[str, Any]):
    """Execute MCP tool"""
    if tool == "scrape":
        request = ScrapeRequest(**params)
        return await scraper_service.scrape(request)
    elif tool == "crawl":
        request = CrawlRequest(**params)
        return await scraper_service.crawl(request)
    elif tool == "map":
        request = MapRequest(**params)
        urls = await scraper_service.map_site(request)
        return {"urls": urls}
    elif tool == "extract":
        request = ExtractRequest(**params)
        return await scraper_service.extract(request)
    else:
        raise HTTPException(400, f"Unknown tool: {tool}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
