#!/usr/bin/env python3
"""
TOOL REGISTRY API — Discovery & Plugin Request Service
=======================================================

Endpoints:
- GET /tools                    List all tools
- GET /tools/{id}               Get specific tool
- GET /tools/category/{cat}     Filter by category
- GET /agent/{name}/config      Get agent's tool config
- POST /agent/{name}/request    Request plugin enable
"""

import os
import json
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from registry import TOOLS, get_all_tools, get_tools_by_category, get_tool, check_requirements


app = FastAPI(
    title="Tool Registry",
    description="Discovery and plugin management for LLM agents",
    version="1.0.0"
)


# =============================================================================
# MODELS
# =============================================================================

class PluginRequest(BaseModel):
    tool_id: str
    reason: str
    priority: str = "normal"  # low, normal, high, critical


class AgentConfig(BaseModel):
    agent_name: str
    enabled_plugins: List[str] = []


# =============================================================================
# PLUGIN STATE (In production, use Redis/DB)
# =============================================================================

AGENT_PLUGINS: Dict[str, List[str]] = {
    "claude": ["llm.anthropic", "search.tavily", "code.github"],
    "gpt4": ["llm.openai", "image.dalle", "search.web"],
    "gemini": ["llm.google", "search.exa", "docs.google"],
    "llama": ["code.interpreter", "mcp.filesystem"],
    "mistral": ["llm.mistral", "search.tavily"],
}

PENDING_REQUESTS: List[dict] = []


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "tools_count": len(TOOLS)}


@app.get("/tools")
async def list_tools():
    """List all available tools."""
    return {
        "tools": get_all_tools(),
        "categories": ["core", "plugin", "local", "experimental"],
        "total": len(TOOLS)
    }


@app.get("/tools/{tool_id}")
async def get_tool_by_id(tool_id: str):
    """Get a specific tool by ID."""
    tool = get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


@app.get("/tools/category/{category}")
async def list_tools_by_category(category: str):
    """List tools by category."""
    tools = get_tools_by_category(category)
    return {"category": category, "tools": tools, "count": len(tools)}


@app.get("/agent/{agent_name}/config")
async def get_agent_config(agent_name: str):
    """Get tool configuration for a specific agent."""
    
    enabled = AGENT_PLUGINS.get(agent_name, [])
    env_vars = dict(os.environ)
    
    config = {
        "agent": agent_name,
        "registry_url": "/tools",
        "request_url": f"/agent/{agent_name}/request",
        "tools": {
            "core": [],
            "enabled": [],
            "available": [],
            "unavailable": []
        }
    }
    
    for tool_id, tool in TOOLS.items():
        tool_info = {
            "id": tool_id,
            "name": tool.name,
            "description": tool.description,
            "docs": tool.docs_url
        }
        
        if tool.category == "core":
            config["tools"]["core"].append(tool_info)
        elif tool.plugin_id in enabled:
            # Check if requirements are met
            req_check = check_requirements(tool_id, env_vars)
            if req_check["met"]:
                config["tools"]["enabled"].append(tool_info)
            else:
                tool_info["missing_env"] = req_check["missing"]
                config["tools"]["unavailable"].append(tool_info)
        else:
            tool_info["plugin_id"] = tool.plugin_id
            tool_info["requires_env"] = tool.env_key
            config["tools"]["available"].append(tool_info)
    
    return config


@app.post("/agent/{agent_name}/request")
async def request_plugin(agent_name: str, request: PluginRequest):
    """Request a plugin to be enabled for an agent."""
    
    tool = get_tool(request.tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Check if already enabled
    enabled = AGENT_PLUGINS.get(agent_name, [])
    if tool.get("plugin_id") in enabled:
        return {"status": "already_enabled", "tool": request.tool_id}
    
    # Add to pending requests
    pending = {
        "agent": agent_name,
        "tool_id": request.tool_id,
        "plugin_id": tool.get("plugin_id"),
        "reason": request.reason,
        "priority": request.priority,
        "requires_env": tool.get("env_key"),
        "status": "pending"
    }
    PENDING_REQUESTS.append(pending)
    
    return {
        "status": "requested",
        "tool": request.tool_id,
        "message": f"Plugin request submitted. Requires {tool.get('env_key')} to be configured.",
        "request_id": len(PENDING_REQUESTS) - 1
    }


@app.get("/requests/pending")
async def list_pending_requests():
    """List all pending plugin requests."""
    return {"requests": PENDING_REQUESTS}


@app.post("/requests/{request_id}/approve")
async def approve_request(request_id: int):
    """Approve a pending plugin request."""
    if request_id >= len(PENDING_REQUESTS):
        raise HTTPException(status_code=404, detail="Request not found")
    
    req = PENDING_REQUESTS[request_id]
    agent = req["agent"]
    plugin_id = req["plugin_id"]
    
    if agent not in AGENT_PLUGINS:
        AGENT_PLUGINS[agent] = []
    
    AGENT_PLUGINS[agent].append(plugin_id)
    req["status"] = "approved"
    
    return {"status": "approved", "agent": agent, "plugin": plugin_id}


# =============================================================================
# AGENT MANIFEST GENERATOR
# =============================================================================

@app.get("/agent/{agent_name}/manifest")
async def get_agent_manifest(agent_name: str):
    """
    Generate a manifest file for an agent with all tool information.
    This can be included in the agent's container or context.
    """
    
    enabled = AGENT_PLUGINS.get(agent_name, [])
    
    manifest = {
        "agent": agent_name,
        "version": "1.0.0",
        "generated_at": "runtime",
        "tools": {
            "core": [
                {"id": "llm_gateway", "endpoint": "http://llm-gateway:8200"},
                {"id": "vector_db", "env": "PINECONE_API_KEY"},
                {"id": "s3_storage", "env": "AWS_ACCESS_KEY_ID"},
                {"id": "redis_cache", "endpoint": "redis://redis:6379"},
                {"id": "mongo_store", "endpoint": "mongodb://mongo:27017"},
            ],
            "enabled_plugins": enabled,
            "request_more": f"POST /agent/{agent_name}/request"
        },
        "registry": {
            "list_all": "GET /tools",
            "by_category": "GET /tools/category/{category}",
            "search": "GET /tools?q={query}"
        },
        "docs": "https://docs.origin-os.ai/tools"
    }
    
    return manifest


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300)
