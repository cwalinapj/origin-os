#!/usr/bin/env python3
"""
Origin OS MCP Hub - Unified MCP Server Gateway
Exposes all MCP servers via HTTP REST API for any LLM to use
"""

import os
import json
import asyncio
import subprocess
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

app = FastAPI(title="Origin OS MCP Hub", version="2.0")

# =============================================================================
# MCP Server Registry
# =============================================================================

MCP_SERVERS = {
    # GTM/Codex - Internal service
    "gtm": {
        "name": "GTM Manager",
        "type": "http",
        "url": os.getenv("CODEX_URL", "http://codex:8000"),
        "description": "Google Tag Manager operations",
        "tools": [
            "gtm_status",
            "gtm_list_containers",
            "gtm_list_tags",
            "gtm_create_tag",
            "gtm_create_trigger",
            "gtm_publish"
        ]
    },
    
    # Firecrawl - Web scraping
    "firecrawl": {
        "name": "Firecrawl",
        "type": "api",
        "url": "https://api.firecrawl.dev/v1",
        "api_key_env": "FIRECRAWL_API_KEY",
        "description": "Web scraping and crawling",
        "tools": [
            "scrape_url",
            "crawl_site",
            "extract_data",
            "screenshot"
        ]
    },
    
    # Docker - Container management
    "docker": {
        "name": "Docker Manager",
        "type": "socket",
        "socket": "/var/run/docker.sock",
        "description": "Container management",
        "tools": [
            "list_containers",
            "run_container",
            "stop_container",
            "inspect_container",
            "build_image",
            "logs"
        ]
    },
    
    # Filesystem - File operations
    "filesystem": {
        "name": "Filesystem",
        "type": "local",
        "base_path": os.getenv("FS_BASE_PATH", "/data"),
        "description": "File read/write operations",
        "tools": [
            "read_file",
            "write_file",
            "list_directory",
            "create_directory",
            "delete_file",
            "search_files"
        ]
    },
    
    # Memory - Persistent storage
    "memory": {
        "name": "Memory",
        "type": "local",
        "storage_path": os.getenv("MEMORY_PATH", "/data/memory"),
        "description": "Persistent memory storage",
        "tools": [
            "create_entities",
            "search_nodes",
            "add_observations",
            "read_graph"
        ]
    },
    
    # SEMrush - SEO data
    "semrush": {
        "name": "SEMrush",
        "type": "api",
        "url": "https://api.semrush.com",
        "api_key_env": "SEMRUSH_API_KEY",
        "description": "SEO and keyword data",
        "tools": [
            "domain_overview",
            "keyword_research",
            "backlink_analysis",
            "competitor_analysis"
        ]
    }
}

# =============================================================================
# MCP Tool Implementations
# =============================================================================

class MCPClient:
    """Base client for MCP server communication"""
    
    def __init__(self, server_config: Dict):
        self.config = server_config
        self.name = server_config["name"]
    
    async def execute(self, tool: str, params: Dict) -> Dict:
        raise NotImplementedError


class HTTPMCPClient(MCPClient):
    """Client for HTTP-based MCP servers (like Codex)"""
    
    async def execute(self, tool: str, params: Dict) -> Dict:
        url = self.config["url"]
        
        # Map tool to endpoint
        endpoints = {
            "gtm_status": ("GET", "/status"),
            "gtm_list_containers": ("GET", "/containers"),
            "gtm_list_tags": ("GET", f"/tags/{params.get('container', '')}"),
            "gtm_create_tag": ("POST", "/tag"),
            "gtm_create_trigger": ("POST", "/trigger"),
            "gtm_publish": ("POST", f"/publish/{params.get('container', '')}")
        }
        
        if tool not in endpoints:
            return {"error": f"Unknown tool: {tool}"}
        
        method, path = endpoints[tool]
        
        # Get JWT token
        from jwt_auth import create_service_token, get_auth_header
        token = create_service_token("mcp-hub", ["*"])
        headers = get_auth_header(token)
        
        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(f"{url}{path}", headers=headers, timeout=30)
            else:
                response = await client.post(f"{url}{path}", headers=headers, json=params, timeout=30)
            
            return response.json()


class FirecrawlMCPClient(MCPClient):
    """Client for Firecrawl API"""
    
    async def execute(self, tool: str, params: Dict) -> Dict:
        api_key = os.getenv(self.config["api_key_env"])
        if not api_key:
            return {"error": "FIRECRAWL_API_KEY not set"}
        
        base_url = self.config["url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            if tool == "scrape_url":
                response = await client.post(
                    f"{base_url}/scrape",
                    headers=headers,
                    json={"url": params.get("url"), "formats": ["markdown", "html"]},
                    timeout=60
                )
            elif tool == "crawl_site":
                response = await client.post(
                    f"{base_url}/crawl",
                    headers=headers,
                    json={
                        "url": params.get("url"),
                        "limit": params.get("limit", 10),
                        "formats": ["markdown"]
                    },
                    timeout=120
                )
            elif tool == "screenshot":
                response = await client.post(
                    f"{base_url}/scrape",
                    headers=headers,
                    json={"url": params.get("url"), "formats": ["screenshot"]},
                    timeout=60
                )
            else:
                return {"error": f"Unknown Firecrawl tool: {tool}"}
            
            if response.status_code == 200:
                return response.json()
            return {"error": response.text}


class DockerMCPClient(MCPClient):
    """Client for Docker operations via socket"""
    
    async def execute(self, tool: str, params: Dict) -> Dict:
        import docker
        client = docker.from_env()
        
        try:
            if tool == "list_containers":
                containers = client.containers.list(all=params.get("all", False))
                return {"containers": [{"id": c.id[:12], "name": c.name, "status": c.status} for c in containers]}
            
            elif tool == "run_container":
                container = client.containers.run(
                    params.get("image"),
                    name=params.get("name"),
                    detach=True,
                    ports=params.get("ports"),
                    environment=params.get("env")
                )
                return {"container_id": container.id[:12], "name": container.name}
            
            elif tool == "stop_container":
                container = client.containers.get(params.get("container_id"))
                container.stop()
                return {"stopped": params.get("container_id")}
            
            elif tool == "logs":
                container = client.containers.get(params.get("container_id"))
                logs = container.logs(tail=params.get("tail", 100)).decode()
                return {"logs": logs}
            
            elif tool == "inspect_container":
                container = client.containers.get(params.get("container_id"))
                return container.attrs
            
            else:
                return {"error": f"Unknown Docker tool: {tool}"}
        
        except Exception as e:
            return {"error": str(e)}


class FilesystemMCPClient(MCPClient):
    """Client for filesystem operations"""
    
    async def execute(self, tool: str, params: Dict) -> Dict:
        base_path = self.config["base_path"]
        
        # Ensure path is within base_path for security
        def safe_path(path: str) -> str:
            import os.path
            full = os.path.normpath(os.path.join(base_path, path))
            if not full.startswith(base_path):
                raise ValueError("Path traversal attempt blocked")
            return full
        
        try:
            if tool == "read_file":
                path = safe_path(params.get("path", ""))
                with open(path, "r") as f:
                    return {"content": f.read()}
            
            elif tool == "write_file":
                path = safe_path(params.get("path", ""))
                with open(path, "w") as f:
                    f.write(params.get("content", ""))
                return {"written": path}
            
            elif tool == "list_directory":
                path = safe_path(params.get("path", ""))
                import os
                entries = os.listdir(path)
                return {"entries": entries}
            
            elif tool == "create_directory":
                path = safe_path(params.get("path", ""))
                import os
                os.makedirs(path, exist_ok=True)
                return {"created": path}
            
            else:
                return {"error": f"Unknown filesystem tool: {tool}"}
        
        except Exception as e:
            return {"error": str(e)}


class MemoryMCPClient(MCPClient):
    """Client for memory/knowledge graph operations"""
    
    def __init__(self, server_config: Dict):
        super().__init__(server_config)
        self.storage_path = server_config.get("storage_path", "/data/memory")
        self.graph_file = os.path.join(self.storage_path, "graph.json")
        self._ensure_storage()
    
    def _ensure_storage(self):
        os.makedirs(self.storage_path, exist_ok=True)
        if not os.path.exists(self.graph_file):
            with open(self.graph_file, "w") as f:
                json.dump({"entities": [], "relations": []}, f)
    
    def _load_graph(self) -> Dict:
        with open(self.graph_file, "r") as f:
            return json.load(f)
    
    def _save_graph(self, graph: Dict):
        with open(self.graph_file, "w") as f:
            json.dump(graph, f, indent=2)
    
    async def execute(self, tool: str, params: Dict) -> Dict:
        graph = self._load_graph()
        
        if tool == "create_entities":
            entities = params.get("entities", [])
            for e in entities:
                e["created_at"] = datetime.utcnow().isoformat()
                graph["entities"].append(e)
            self._save_graph(graph)
            return {"created": len(entities)}
        
        elif tool == "search_nodes":
            query = params.get("query", "").lower()
            results = [e for e in graph["entities"] if query in json.dumps(e).lower()]
            return {"results": results}
        
        elif tool == "read_graph":
            return graph
        
        elif tool == "add_observations":
            observations = params.get("observations", [])
            for obs in observations:
                entity_name = obs.get("entityName")
                for e in graph["entities"]:
                    if e.get("name") == entity_name:
                        if "observations" not in e:
                            e["observations"] = []
                        e["observations"].extend(obs.get("contents", []))
            self._save_graph(graph)
            return {"added": len(observations)}
        
        else:
            return {"error": f"Unknown memory tool: {tool}"}


# =============================================================================
# Client Factory
# =============================================================================

def get_client(server_id: str) -> MCPClient:
    """Get appropriate client for server type"""
    if server_id not in MCP_SERVERS:
        raise HTTPException(status_code=404, detail=f"Unknown server: {server_id}")
    
    config = MCP_SERVERS[server_id]
    server_type = config["type"]
    
    if server_type == "http":
        return HTTPMCPClient(config)
    elif server_type == "api" and "firecrawl" in server_id:
        return FirecrawlMCPClient(config)
    elif server_type == "socket":
        return DockerMCPClient(config)
    elif server_type == "local" and "memory" in server_id:
        return MemoryMCPClient(config)
    elif server_type == "local":
        return FilesystemMCPClient(config)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported server type: {server_type}")


# =============================================================================
# API Models
# =============================================================================

class ToolCall(BaseModel):
    server: str
    tool: str
    params: Dict = {}


class ToolResult(BaseModel):
    success: bool
    server: str
    tool: str
    result: Optional[Any] = None
    error: Optional[str] = None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """MCP Hub info"""
    return {
        "service": "Origin OS MCP Hub",
        "version": "2.0",
        "servers": list(MCP_SERVERS.keys()),
        "endpoints": {
            "list_servers": "GET /servers",
            "list_tools": "GET /servers/{server_id}/tools",
            "execute": "POST /execute"
        }
    }


@app.get("/servers")
async def list_servers():
    """List all available MCP servers"""
    servers = []
    for server_id, config in MCP_SERVERS.items():
        servers.append({
            "id": server_id,
            "name": config["name"],
            "description": config["description"],
            "type": config["type"],
            "tools": config["tools"]
        })
    return {"servers": servers}


@app.get("/servers/{server_id}/tools")
async def list_tools(server_id: str):
    """List tools for a specific server"""
    if server_id not in MCP_SERVERS:
        raise HTTPException(status_code=404, detail=f"Unknown server: {server_id}")
    
    config = MCP_SERVERS[server_id]
    return {
        "server": server_id,
        "name": config["name"],
        "tools": config["tools"]
    }


@app.post("/execute", response_model=ToolResult)
async def execute_tool(call: ToolCall):
    """Execute a tool on an MCP server"""
    try:
        client = get_client(call.server)
        result = await client.execute(call.tool, call.params)
        
        if "error" in result:
            return ToolResult(
                success=False,
                server=call.server,
                tool=call.tool,
                error=result["error"]
            )
        
        return ToolResult(
            success=True,
            server=call.server,
            tool=call.tool,
            result=result
        )
    
    except Exception as e:
        return ToolResult(
            success=False,
            server=call.server,
            tool=call.tool,
            error=str(e)
        )


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "mcp-hub"}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🔌 ORIGIN OS MCP HUB")
    print("=" * 60)
    print("\nAvailable MCP Servers:")
    for server_id, config in MCP_SERVERS.items():
        print(f"  • {server_id}: {config['name']}")
        print(f"    Tools: {', '.join(config['tools'])}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
