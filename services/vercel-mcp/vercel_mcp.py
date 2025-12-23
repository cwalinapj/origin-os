#!/usr/bin/env python3
"""
VERCEL MCP — Vercel Deployment Integration for Origin OS
=========================================================
Real Vercel API integration for:
- Project management
- Deployments
- Environment variables
- Domains
- Logs and analytics
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

VERCEL_TOKEN = os.getenv("VERCEL_TOKEN", "")
VERCEL_TEAM_ID = os.getenv("VERCEL_TEAM_ID", "")
VERCEL_BASE_URL = "https://api.vercel.com"

VAULT_URL = os.getenv("VAULT_URL", "http://vault:8000")
CODEX_URL = os.getenv("CODEX_URL", "http://codex:8000")

DATA_DIR = Path(os.getenv("VERCEL_DATA_DIR", "/data/vercel"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vercel-mcp")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Vercel MCP",
    description="Vercel Deployment Integration for Origin OS",
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

class ProjectRequest(BaseModel):
    name: str
    framework: Optional[str] = None  # nextjs, nuxt, gatsby, etc.
    git_repository: Optional[str] = None

class DeployRequest(BaseModel):
    project_id: str
    target: str = "production"  # production, preview
    git_ref: Optional[str] = None

class EnvVarRequest(BaseModel):
    project_id: str
    key: str
    value: str
    target: List[str] = ["production", "preview", "development"]
    env_type: str = "encrypted"  # plain, encrypted, secret

class DomainRequest(BaseModel):
    project_id: str
    domain: str

class LogsRequest(BaseModel):
    deployment_id: str
    follow: bool = False

# =============================================================================
# VERCEL CLIENT
# =============================================================================

class VercelClient:
    def __init__(self, token: str = None, team_id: str = None):
        self.token = token or VERCEL_TOKEN
        self.team_id = team_id or VERCEL_TEAM_ID
        self.base_url = VERCEL_BASE_URL
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={"Authorization": f"Bearer {self.token}"}
            )
        return self._client
    
    def _add_team(self, params: Dict) -> Dict:
        """Add team ID to params if set"""
        if self.team_id:
            params["teamId"] = self.team_id
        return params
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    # =========================================================================
    # USER/TEAM
    # =========================================================================
    
    async def get_user(self) -> Dict:
        """Get current user info"""
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/v2/user")
        response.raise_for_status()
        return response.json()
    
    async def get_teams(self) -> List[Dict]:
        """Get user's teams"""
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/v2/teams")
        response.raise_for_status()
        return response.json().get("teams", [])
    
    # =========================================================================
    # PROJECTS
    # =========================================================================
    
    async def list_projects(self, limit: int = 20) -> List[Dict]:
        """List all projects"""
        client = await self._get_client()
        params = self._add_team({"limit": limit})
        response = await client.get(f"{self.base_url}/v9/projects", params=params)
        response.raise_for_status()
        return response.json().get("projects", [])
    
    async def get_project(self, project_id: str) -> Dict:
        """Get project details"""
        client = await self._get_client()
        params = self._add_team({})
        response = await client.get(
            f"{self.base_url}/v9/projects/{project_id}",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def create_project(
        self,
        name: str,
        framework: str = None,
        git_repository: str = None
    ) -> Dict:
        """Create a new project"""
        client = await self._get_client()
        
        data = {"name": name}
        if framework:
            data["framework"] = framework
        if git_repository:
            data["gitRepository"] = {
                "type": "github",
                "repo": git_repository
            }
        
        params = self._add_team({})
        response = await client.post(
            f"{self.base_url}/v10/projects",
            params=params,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def delete_project(self, project_id: str) -> bool:
        """Delete a project"""
        client = await self._get_client()
        params = self._add_team({})
        response = await client.delete(
            f"{self.base_url}/v9/projects/{project_id}",
            params=params
        )
        response.raise_for_status()
        return True
    
    # =========================================================================
    # DEPLOYMENTS
    # =========================================================================
    
    async def list_deployments(self, project_id: str = None, limit: int = 20) -> List[Dict]:
        """List deployments"""
        client = await self._get_client()
        params = self._add_team({"limit": limit})
        if project_id:
            params["projectId"] = project_id
        
        response = await client.get(f"{self.base_url}/v6/deployments", params=params)
        response.raise_for_status()
        return response.json().get("deployments", [])
    
    async def get_deployment(self, deployment_id: str) -> Dict:
        """Get deployment details"""
        client = await self._get_client()
        params = self._add_team({})
        response = await client.get(
            f"{self.base_url}/v13/deployments/{deployment_id}",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def create_deployment(
        self,
        project_id: str,
        target: str = "production",
        git_ref: str = None
    ) -> Dict:
        """Create a new deployment"""
        client = await self._get_client()
        
        data = {
            "name": project_id,
            "target": target
        }
        if git_ref:
            data["gitSource"] = {"ref": git_ref}
        
        params = self._add_team({})
        response = await client.post(
            f"{self.base_url}/v13/deployments",
            params=params,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def cancel_deployment(self, deployment_id: str) -> Dict:
        """Cancel a deployment"""
        client = await self._get_client()
        params = self._add_team({})
        response = await client.patch(
            f"{self.base_url}/v12/deployments/{deployment_id}/cancel",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # ENVIRONMENT VARIABLES
    # =========================================================================
    
    async def list_env_vars(self, project_id: str) -> List[Dict]:
        """List environment variables for a project"""
        client = await self._get_client()
        params = self._add_team({})
        response = await client.get(
            f"{self.base_url}/v10/projects/{project_id}/env",
            params=params
        )
        response.raise_for_status()
        return response.json().get("envs", [])
    
    async def create_env_var(
        self,
        project_id: str,
        key: str,
        value: str,
        target: List[str] = None,
        env_type: str = "encrypted"
    ) -> Dict:
        """Create an environment variable"""
        client = await self._get_client()
        
        target = target or ["production", "preview", "development"]
        data = {
            "key": key,
            "value": value,
            "target": target,
            "type": env_type
        }
        
        params = self._add_team({})
        response = await client.post(
            f"{self.base_url}/v10/projects/{project_id}/env",
            params=params,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def delete_env_var(self, project_id: str, env_id: str) -> bool:
        """Delete an environment variable"""
        client = await self._get_client()
        params = self._add_team({})
        response = await client.delete(
            f"{self.base_url}/v10/projects/{project_id}/env/{env_id}",
            params=params
        )
        response.raise_for_status()
        return True
    
    # =========================================================================
    # DOMAINS
    # =========================================================================
    
    async def list_domains(self, project_id: str = None) -> List[Dict]:
        """List domains"""
        client = await self._get_client()
        params = self._add_team({})
        if project_id:
            params["projectId"] = project_id
        
        response = await client.get(f"{self.base_url}/v5/domains", params=params)
        response.raise_for_status()
        return response.json().get("domains", [])
    
    async def add_domain(self, project_id: str, domain: str) -> Dict:
        """Add a domain to a project"""
        client = await self._get_client()
        
        data = {"name": domain}
        params = self._add_team({})
        
        response = await client.post(
            f"{self.base_url}/v10/projects/{project_id}/domains",
            params=params,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def remove_domain(self, project_id: str, domain: str) -> bool:
        """Remove a domain from a project"""
        client = await self._get_client()
        params = self._add_team({})
        
        response = await client.delete(
            f"{self.base_url}/v9/projects/{project_id}/domains/{domain}",
            params=params
        )
        response.raise_for_status()
        return True
    
    # =========================================================================
    # LOGS
    # =========================================================================
    
    async def get_deployment_logs(self, deployment_id: str) -> List[Dict]:
        """Get deployment build logs"""
        client = await self._get_client()
        params = self._add_team({})
        
        response = await client.get(
            f"{self.base_url}/v2/deployments/{deployment_id}/events",
            params=params
        )
        response.raise_for_status()
        return response.json()


# Global client
vercel_client = VercelClient()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "vercel-mcp"}

@app.get("/info")
async def info():
    return {
        "service": "vercel-mcp",
        "version": "1.0.0",
        "capabilities": [
            "projects",
            "deployments",
            "env_vars",
            "domains",
            "logs"
        ]
    }

@app.get("/user")
async def get_user():
    """Get current user info"""
    try:
        return await vercel_client.get_user()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.get("/teams")
async def get_teams():
    """Get user's teams"""
    try:
        teams = await vercel_client.get_teams()
        return {"teams": teams}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

# Projects
@app.get("/projects")
async def list_projects(limit: int = 20):
    """List all projects"""
    try:
        projects = await vercel_client.list_projects(limit)
        return {"projects": projects}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Get project details"""
    try:
        return await vercel_client.get_project(project_id)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.post("/projects")
async def create_project(request: ProjectRequest):
    """Create a new project"""
    try:
        return await vercel_client.create_project(
            request.name,
            request.framework,
            request.git_repository
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

# Deployments
@app.get("/deployments")
async def list_deployments(project_id: str = None, limit: int = 20):
    """List deployments"""
    try:
        deployments = await vercel_client.list_deployments(project_id, limit)
        return {"deployments": deployments}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.get("/deployments/{deployment_id}")
async def get_deployment(deployment_id: str):
    """Get deployment details"""
    try:
        return await vercel_client.get_deployment(deployment_id)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.post("/deploy")
async def create_deployment(request: DeployRequest):
    """Create a new deployment"""
    try:
        return await vercel_client.create_deployment(
            request.project_id,
            request.target,
            request.git_ref
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

# Environment Variables
@app.get("/projects/{project_id}/env")
async def list_env_vars(project_id: str):
    """List environment variables"""
    try:
        envs = await vercel_client.list_env_vars(project_id)
        return {"envs": envs}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.post("/env")
async def create_env_var(request: EnvVarRequest):
    """Create an environment variable"""
    try:
        return await vercel_client.create_env_var(
            request.project_id,
            request.key,
            request.value,
            request.target,
            request.env_type
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

# Domains
@app.get("/domains")
async def list_domains(project_id: str = None):
    """List domains"""
    try:
        domains = await vercel_client.list_domains(project_id)
        return {"domains": domains}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.post("/domains")
async def add_domain(request: DomainRequest):
    """Add a domain to a project"""
    try:
        return await vercel_client.add_domain(request.project_id, request.domain)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

# Logs
@app.get("/deployments/{deployment_id}/logs")
async def get_deployment_logs(deployment_id: str):
    """Get deployment logs"""
    try:
        logs = await vercel_client.get_deployment_logs(deployment_id)
        return {"logs": logs}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

# =============================================================================
# MCP TOOL INTERFACE
# =============================================================================

@app.post("/mcp/tool")
async def mcp_tool(tool: str, params: Dict[str, Any]):
    """Execute MCP tool"""
    try:
        if tool == "vercel_list_projects":
            return await vercel_client.list_projects(params.get("limit", 20))
        
        elif tool == "vercel_get_project":
            return await vercel_client.get_project(params["project_id"])
        
        elif tool == "vercel_create_project":
            return await vercel_client.create_project(
                params["name"],
                params.get("framework"),
                params.get("git_repository")
            )
        
        elif tool == "vercel_deploy":
            return await vercel_client.create_deployment(
                params["project_id"],
                params.get("target", "production"),
                params.get("git_ref")
            )
        
        elif tool == "vercel_list_deployments":
            return await vercel_client.list_deployments(
                params.get("project_id"),
                params.get("limit", 20)
            )
        
        elif tool == "vercel_get_deployment":
            return await vercel_client.get_deployment(params["deployment_id"])
        
        elif tool == "vercel_create_env":
            return await vercel_client.create_env_var(
                params["project_id"],
                params["key"],
                params["value"],
                params.get("target"),
                params.get("env_type", "encrypted")
            )
        
        elif tool == "vercel_add_domain":
            return await vercel_client.add_domain(
                params["project_id"],
                params["domain"]
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
