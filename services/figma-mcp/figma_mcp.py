#!/usr/bin/env python3
"""
FIGMA MCP — Figma Design Integration for Origin OS
===================================================
Real Figma API integration for:
- Reading designs and components
- Exporting assets (PNG, SVG, PDF)
- Getting design tokens (colors, typography, spacing)
- Design-to-code generation
- Comments and collaboration
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

FIGMA_API_KEY = os.getenv("FIGMA_API_KEY", "")
FIGMA_BASE_URL = "https://api.figma.com/v1"

VAULT_URL = os.getenv("VAULT_URL", "http://vault:8000")
CODEX_URL = os.getenv("CODEX_URL", "http://codex:8000")

DATA_DIR = Path(os.getenv("FIGMA_DATA_DIR", "/data/figma"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("figma-mcp")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Figma MCP",
    description="Figma Design Integration for Origin OS",
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

class FileRequest(BaseModel):
    file_key: str
    node_ids: Optional[List[str]] = None

class ExportRequest(BaseModel):
    file_key: str
    node_ids: List[str]
    format: str = "png"  # png, svg, pdf, jpg
    scale: float = 1.0

class ComponentRequest(BaseModel):
    file_key: str
    component_id: Optional[str] = None

class DesignTokensRequest(BaseModel):
    file_key: str
    include_colors: bool = True
    include_typography: bool = True
    include_effects: bool = True
    include_spacing: bool = True

class CodeGenRequest(BaseModel):
    file_key: str
    node_id: str
    framework: str = "react"  # react, vue, html, tailwind
    include_styles: bool = True

# =============================================================================
# FIGMA CLIENT
# =============================================================================

class FigmaClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or FIGMA_API_KEY
        self.base_url = FIGMA_BASE_URL
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={"X-Figma-Token": self.api_key}
            )
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================
    
    async def get_file(self, file_key: str, node_ids: List[str] = None) -> Dict:
        """Get Figma file data"""
        client = await self._get_client()
        
        url = f"{self.base_url}/files/{file_key}"
        params = {}
        if node_ids:
            params["ids"] = ",".join(node_ids)
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_file_nodes(self, file_key: str, node_ids: List[str]) -> Dict:
        """Get specific nodes from a file"""
        client = await self._get_client()
        
        url = f"{self.base_url}/files/{file_key}/nodes"
        params = {"ids": ",".join(node_ids)}
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_file_components(self, file_key: str) -> Dict:
        """Get all components in a file"""
        client = await self._get_client()
        
        url = f"{self.base_url}/files/{file_key}/components"
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    
    async def get_file_styles(self, file_key: str) -> Dict:
        """Get all styles in a file"""
        client = await self._get_client()
        
        url = f"{self.base_url}/files/{file_key}/styles"
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    async def export_images(
        self,
        file_key: str,
        node_ids: List[str],
        format: str = "png",
        scale: float = 1.0
    ) -> Dict[str, str]:
        """Export nodes as images, returns URLs"""
        client = await self._get_client()
        
        url = f"{self.base_url}/images/{file_key}"
        params = {
            "ids": ",".join(node_ids),
            "format": format,
            "scale": scale
        }
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("images", {})
    
    async def download_image(self, image_url: str, save_path: Path) -> Path:
        """Download an exported image"""
        client = await self._get_client()
        
        response = await client.get(image_url)
        response.raise_for_status()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(response.content)
        return save_path
    
    # =========================================================================
    # DESIGN TOKENS
    # =========================================================================
    
    async def extract_design_tokens(self, file_key: str) -> Dict:
        """Extract design tokens from a Figma file"""
        file_data = await self.get_file(file_key)
        styles = await self.get_file_styles(file_key)
        
        tokens = {
            "colors": {},
            "typography": {},
            "effects": {},
            "spacing": [],
            "radii": []
        }
        
        # Extract color styles
        for style in styles.get("meta", {}).get("styles", []):
            if style.get("style_type") == "FILL":
                tokens["colors"][style["name"]] = {
                    "key": style["key"],
                    "description": style.get("description", "")
                }
            elif style.get("style_type") == "TEXT":
                tokens["typography"][style["name"]] = {
                    "key": style["key"],
                    "description": style.get("description", "")
                }
            elif style.get("style_type") == "EFFECT":
                tokens["effects"][style["name"]] = {
                    "key": style["key"],
                    "description": style.get("description", "")
                }
        
        return tokens
    
    def _extract_colors_from_node(self, node: Dict, colors: Dict):
        """Recursively extract colors from nodes"""
        if "fills" in node:
            for fill in node["fills"]:
                if fill.get("type") == "SOLID" and "color" in fill:
                    c = fill["color"]
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        int(c["r"] * 255),
                        int(c["g"] * 255),
                        int(c["b"] * 255)
                    )
                    name = node.get("name", "unnamed")
                    colors[name] = {
                        "hex": hex_color,
                        "rgba": f"rgba({int(c['r']*255)}, {int(c['g']*255)}, {int(c['b']*255)}, {c.get('a', 1)})"
                    }
        
        for child in node.get("children", []):
            self._extract_colors_from_node(child, colors)
    
    # =========================================================================
    # COMMENTS
    # =========================================================================
    
    async def get_comments(self, file_key: str) -> List[Dict]:
        """Get comments on a file"""
        client = await self._get_client()
        
        url = f"{self.base_url}/files/{file_key}/comments"
        response = await client.get(url)
        response.raise_for_status()
        return response.json().get("comments", [])
    
    async def post_comment(self, file_key: str, message: str, node_id: str = None) -> Dict:
        """Post a comment on a file"""
        client = await self._get_client()
        
        url = f"{self.base_url}/files/{file_key}/comments"
        data = {"message": message}
        if node_id:
            data["client_meta"] = {"node_id": node_id}
        
        response = await client.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # CODE GENERATION
    # =========================================================================
    
    async def generate_code(
        self,
        file_key: str,
        node_id: str,
        framework: str = "react"
    ) -> str:
        """Generate code from a Figma node"""
        # Get node data
        nodes = await self.get_file_nodes(file_key, [node_id])
        node_data = nodes.get("nodes", {}).get(node_id, {}).get("document", {})
        
        if framework == "react":
            return self._generate_react(node_data)
        elif framework == "html":
            return self._generate_html(node_data)
        elif framework == "tailwind":
            return self._generate_tailwind(node_data)
        else:
            return self._generate_html(node_data)
    
    def _generate_react(self, node: Dict) -> str:
        """Generate React component from node"""
        name = node.get("name", "Component").replace(" ", "")
        node_type = node.get("type", "FRAME")
        
        # Extract styles
        styles = self._extract_styles(node)
        
        code = f'''import React from 'react';

const {name} = () => {{
  return (
    <div style={{{styles}}}>
      {self._generate_children_react(node.get("children", []))}
    </div>
  );
}};

export default {name};
'''
        return code
    
    def _generate_html(self, node: Dict) -> str:
        """Generate HTML from node"""
        name = node.get("name", "component")
        styles = self._extract_styles(node)
        
        html = f'''<div class="{name}" style="{self._styles_to_css(styles)}">
  {self._generate_children_html(node.get("children", []))}
</div>
'''
        return html
    
    def _generate_tailwind(self, node: Dict) -> str:
        """Generate Tailwind HTML from node"""
        classes = self._extract_tailwind_classes(node)
        
        html = f'''<div class="{classes}">
  {self._generate_children_tailwind(node.get("children", []))}
</div>
'''
        return html
    
    def _extract_styles(self, node: Dict) -> Dict:
        """Extract CSS-like styles from node"""
        styles = {}
        
        box = node.get("absoluteBoundingBox", {})
        if box:
            styles["width"] = f"{box.get('width', 0)}px"
            styles["height"] = f"{box.get('height', 0)}px"
        
        if "backgroundColor" in node:
            c = node["backgroundColor"]
            styles["backgroundColor"] = f"rgba({int(c['r']*255)}, {int(c['g']*255)}, {int(c['b']*255)}, {c.get('a', 1)})"
        
        if "cornerRadius" in node:
            styles["borderRadius"] = f"{node['cornerRadius']}px"
        
        return styles
    
    def _styles_to_css(self, styles: Dict) -> str:
        """Convert styles dict to CSS string"""
        return "; ".join(f"{k}: {v}" for k, v in styles.items())
    
    def _extract_tailwind_classes(self, node: Dict) -> str:
        """Extract Tailwind classes from node"""
        classes = []
        
        box = node.get("absoluteBoundingBox", {})
        if box:
            w = box.get("width", 0)
            h = box.get("height", 0)
            classes.append(f"w-[{int(w)}px]")
            classes.append(f"h-[{int(h)}px]")
        
        if "cornerRadius" in node:
            r = node["cornerRadius"]
            if r >= 9999:
                classes.append("rounded-full")
            elif r > 0:
                classes.append(f"rounded-[{int(r)}px]")
        
        return " ".join(classes)
    
    def _generate_children_react(self, children: List[Dict]) -> str:
        return "\n      ".join(
            f"<div>{c.get('name', 'child')}</div>"
            for c in children[:10]
        )
    
    def _generate_children_html(self, children: List[Dict]) -> str:
        return "\n  ".join(
            f"<div class=\"{c.get('name', 'child')}\">{c.get('name', '')}</div>"
            for c in children[:10]
        )
    
    def _generate_children_tailwind(self, children: List[Dict]) -> str:
        return "\n  ".join(
            f"<div>{c.get('name', 'child')}</div>"
            for c in children[:10]
        )


# Global client
figma_client = FigmaClient()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "figma-mcp"}

@app.get("/info")
async def info():
    return {
        "service": "figma-mcp",
        "version": "1.0.0",
        "capabilities": [
            "get_file",
            "get_components",
            "get_styles", 
            "export_images",
            "extract_tokens",
            "generate_code",
            "comments"
        ]
    }

@app.post("/file")
async def get_file(request: FileRequest):
    """Get Figma file data"""
    try:
        data = await figma_client.get_file(request.file_key, request.node_ids)
        return data
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.post("/components")
async def get_components(request: ComponentRequest):
    """Get components from a file"""
    try:
        data = await figma_client.get_file_components(request.file_key)
        return data
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.post("/styles")
async def get_styles(request: FileRequest):
    """Get styles from a file"""
    try:
        data = await figma_client.get_file_styles(request.file_key)
        return data
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.post("/export")
async def export_images(request: ExportRequest):
    """Export nodes as images"""
    try:
        urls = await figma_client.export_images(
            request.file_key,
            request.node_ids,
            request.format,
            request.scale
        )
        return {"images": urls}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.post("/tokens")
async def extract_tokens(request: DesignTokensRequest):
    """Extract design tokens"""
    try:
        tokens = await figma_client.extract_design_tokens(request.file_key)
        return tokens
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.post("/generate")
async def generate_code(request: CodeGenRequest):
    """Generate code from design"""
    try:
        code = await figma_client.generate_code(
            request.file_key,
            request.node_id,
            request.framework
        )
        return {"code": code, "framework": request.framework}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.get("/comments/{file_key}")
async def get_comments(file_key: str):
    """Get comments on a file"""
    try:
        comments = await figma_client.get_comments(file_key)
        return {"comments": comments}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

# =============================================================================
# MCP TOOL INTERFACE
# =============================================================================

@app.post("/mcp/tool")
async def mcp_tool(tool: str, params: Dict[str, Any]):
    """Execute MCP tool"""
    try:
        if tool == "figma_get_file":
            return await figma_client.get_file(params["file_key"], params.get("node_ids"))
        
        elif tool == "figma_get_components":
            return await figma_client.get_file_components(params["file_key"])
        
        elif tool == "figma_export":
            return await figma_client.export_images(
                params["file_key"],
                params["node_ids"],
                params.get("format", "png"),
                params.get("scale", 1.0)
            )
        
        elif tool == "figma_tokens":
            return await figma_client.extract_design_tokens(params["file_key"])
        
        elif tool == "figma_codegen":
            return await figma_client.generate_code(
                params["file_key"],
                params["node_id"],
                params.get("framework", "react")
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
