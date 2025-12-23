#!/usr/bin/env python3
"""
Origin OS MCP - Model Context Protocol Server
Orchestrates LLM calls and communicates with Codex via JWT-signed requests
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Optional

from jwt_auth import create_service_token, get_auth_header

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Origin OS MCP")

# Service URLs
CODEX_URL = os.getenv("CODEX_URL", "http://codex:8000")
JWT_SECRET = os.getenv("JWT_SECRET", "origin-os-default-secret-change-me")

# MCP scopes
MCP_SCOPES = [
    "gtm:read",
    "gtm:write",
    "gtm:publish",
    "containers:read",
    "containers:write"
]


class CodexClient:
    """
    JWT-authenticated client for Codex API
    """
    
    def __init__(self, base_url: str = CODEX_URL):
        self.base_url = base_url
        self._token = None
        self._token_created = 0
    
    def _get_token(self) -> str:
        """Get or refresh service token"""
        import time
        # Refresh token every 4 minutes (before 5 min expiry)
        if not self._token or (time.time() - self._token_created) > 240:
            self._token = create_service_token("mcp", MCP_SCOPES)
            self._token_created = time.time()
        return self._token
    
    def _headers(self) -> Dict[str, str]:
        """Get request headers with JWT"""
        return {
            **get_auth_header(self._get_token()),
            "Content-Type": "application/json",
            "X-Service": "mcp"
        }
    
    async def get_status(self) -> Dict:
        """Get Codex status"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/status",
                headers=self._headers(),
                timeout=10.0
            )
            return response.json()
    
    async def list_containers(self) -> List[Dict]:
        """List all GTM containers"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/containers",
                headers=self._headers(),
                timeout=10.0
            )
            return response.json().get("containers", [])
    
    async def create_tag(self, container: str, config: Dict) -> Dict:
        """Create a GTM tag"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/tag",
                headers=self._headers(),
                json={"container": container, "config": config},
                timeout=30.0
            )
            return response.json()
    
    async def create_trigger(self, container: str, config: Dict) -> Dict:
        """Create a GTM trigger"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/trigger",
                headers=self._headers(),
                json={"container": container, "config": config},
                timeout=30.0
            )
            return response.json()
    
    async def list_tags(self, container: str) -> List[Dict]:
        """List tags in a container"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/tags/{container}",
                headers=self._headers(),
                timeout=10.0
            )
            return response.json().get("tags", [])
    
    async def publish(self, container: str) -> Dict:
        """Publish a container"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/publish/{container}",
                headers=self._headers(),
                timeout=30.0
            )
            return response.json()


# Initialize Codex client
codex = CodexClient()


# Request/Response models
class MCPToolCall(BaseModel):
    tool: str
    params: Dict = {}


class MCPToolResult(BaseModel):
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None


# MCP Tool handlers
TOOLS = {
    "gtm_status": {
        "description": "Get GTM connection status and available containers",
        "params": []
    },
    "gtm_list_containers": {
        "description": "List all GTM containers",
        "params": []
    },
    "gtm_list_tags": {
        "description": "List tags in a container",
        "params": ["container"]
    },
    "gtm_create_tag": {
        "description": "Create a new GTM tag",
        "params": ["container", "config"]
    },
    "gtm_create_trigger": {
        "description": "Create a new GTM trigger",
        "params": ["container", "config"]
    },
    "gtm_publish": {
        "description": "Publish a GTM container",
        "params": ["container"]
    }
}


async def execute_tool(tool: str, params: Dict) -> MCPToolResult:
    """Execute an MCP tool"""
    try:
        if tool == "gtm_status":
            data = await codex.get_status()
            return MCPToolResult(success=True, data=data)
        
        elif tool == "gtm_list_containers":
            containers = await codex.list_containers()
            return MCPToolResult(success=True, data={"containers": containers})
        
        elif tool == "gtm_list_tags":
            container = params.get("container")
            if not container:
                return MCPToolResult(success=False, error="container required")
            tags = await codex.list_tags(container)
            return MCPToolResult(success=True, data={"tags": tags})
        
        elif tool == "gtm_create_tag":
            container = params.get("container")
            config = params.get("config")
            if not container or not config:
                return MCPToolResult(success=False, error="container and config required")
            result = await codex.create_tag(container, config)
            return MCPToolResult(success=True, data=result)
        
        elif tool == "gtm_create_trigger":
            container = params.get("container")
            config = params.get("config")
            if not container or not config:
                return MCPToolResult(success=False, error="container and config required")
            result = await codex.create_trigger(container, config)
            return MCPToolResult(success=True, data=result)
        
        elif tool == "gtm_publish":
            container = params.get("container")
            if not container:
                return MCPToolResult(success=False, error="container required")
            result = await codex.publish(container)
            return MCPToolResult(success=True, data=result)
        
        else:
            return MCPToolResult(success=False, error=f"Unknown tool: {tool}")
    
    except Exception as e:
        return MCPToolResult(success=False, error=str(e))


# API Endpoints
@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    return {"tools": TOOLS}


@app.post("/execute")
async def execute(call: MCPToolCall):
    """Execute an MCP tool"""
    result = await execute_tool(call.tool, call.params)
    return result


@app.get("/status")
async def status():
    """Get MCP and Codex status"""
    try:
        codex_status = await codex.get_status()
        return {
            "service": "mcp",
            "status": "healthy",
            "jwt_enabled": True,
            "codex": codex_status
        }
    except Exception as e:
        return {
            "service": "mcp",
            "status": "degraded",
            "jwt_enabled": True,
            "codex_error": str(e)
        }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mcp"}


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("🔌 ORIGIN OS MCP")
    print("=" * 50)
    print(f"JWT Auth: Enabled")
    print(f"Codex URL: {CODEX_URL}")
    print(f"Scopes: {MCP_SCOPES}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
