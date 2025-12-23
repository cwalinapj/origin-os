#!/usr/bin/env python3
"""
CURSOR MCP — Cursor IDE Integration for Origin OS
==================================================
Provides MCP tools for:
- File editing and navigation
- Code generation requests to Cursor AI
- Workspace management
- Integration with Auto-Claude for autonomous coding
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

CURSOR_API_KEY = os.getenv("CURSOR_API_KEY", "")
CURSOR_WORKSPACE_ID = os.getenv("CURSOR_WORKSPACE_ID", "")
VAULT_URL = os.getenv("VAULT_URL", "http://vault:8000")
CODEX_URL = os.getenv("CODEX_URL", "http://codex:8000")
MCP_HUB_URL = os.getenv("MCP_HUB_URL", "http://mcp-hub:8000")
AUTO_CLAUDE_URL = os.getenv("AUTO_CLAUDE_URL", "http://auto-claude:8000")

DATA_DIR = Path(os.getenv("CURSOR_DATA_DIR", "/data/cursor"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cursor-mcp")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Cursor MCP",
    description="Cursor IDE Integration for Origin OS",
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

class FileEdit(BaseModel):
    path: str
    content: str
    create_if_missing: bool = True

class FileRead(BaseModel):
    path: str

class CodeGenRequest(BaseModel):
    prompt: str
    context_files: List[str] = []
    language: Optional[str] = None
    model: str = "claude"  # claude | gpt-4 | cursor

class WorkspaceAction(BaseModel):
    action: str  # open | close | list | search
    path: Optional[str] = None
    query: Optional[str] = None

class MCPToolCall(BaseModel):
    tool: str
    params: Dict[str, Any]

class CursorSession(BaseModel):
    session_id: str
    workspace_path: str
    active_files: List[str] = []
    created_at: str
    last_activity: str

# =============================================================================
# CURSOR CLIENT
# =============================================================================

class CursorClient:
    """Client for interacting with Cursor IDE"""
    
    def __init__(self, api_key: str = None, workspace_id: str = None):
        self.api_key = api_key or CURSOR_API_KEY
        self.workspace_id = workspace_id or CURSOR_WORKSPACE_ID
        self.sessions: Dict[str, CursorSession] = {}
        self.http_client = httpx.AsyncClient(timeout=60.0)
    
    async def read_file(self, path: str) -> str:
        """Read file contents"""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path.read_text()
    
    async def write_file(self, path: str, content: str, create: bool = True) -> bool:
        """Write content to file"""
        file_path = Path(path)
        if not file_path.exists() and not create:
            raise FileNotFoundError(f"File not found: {path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        logger.info(f"Written file: {path}")
        return True
    
    async def edit_file(self, path: str, old_text: str, new_text: str) -> bool:
        """Edit file by replacing text"""
        content = await self.read_file(path)
        if old_text not in content:
            raise ValueError(f"Text not found in file: {path}")
        new_content = content.replace(old_text, new_text, 1)
        await self.write_file(path, new_content)
        return True
    
    async def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """List files in directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        return [str(f) for f in dir_path.rglob(pattern) if f.is_file()]
    
    async def search_files(self, directory: str, query: str) -> List[Dict[str, Any]]:
        """Search for text in files"""
        results = []
        dir_path = Path(directory)
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    if query.lower() in content.lower():
                        # Find line numbers
                        lines = content.split("\n")
                        matches = []
                        for i, line in enumerate(lines, 1):
                            if query.lower() in line.lower():
                                matches.append({"line": i, "text": line.strip()})
                        results.append({
                            "file": str(file_path),
                            "matches": matches[:10]  # Limit matches
                        })
                except:
                    pass
        return results[:50]  # Limit results
    
    async def generate_code(
        self,
        prompt: str,
        context_files: List[str] = None,
        language: str = None,
        model: str = "claude"
    ) -> str:
        """Generate code using specified model"""
        
        # Build context from files
        context = ""
        if context_files:
            for file_path in context_files[:5]:  # Limit context files
                try:
                    content = await self.read_file(file_path)
                    context += f"\n### {file_path}\n```\n{content[:2000]}\n```\n"
                except:
                    pass
        
        # Route to appropriate model
        if model == "claude":
            return await self._generate_claude(prompt, context, language)
        elif model in ["gpt-4", "gpt-4o", "openai"]:
            return await self._generate_openai(prompt, context, language)
        else:
            return await self._generate_openrouter(prompt, context, language, model)
    
    async def _generate_claude(self, prompt: str, context: str, language: str) -> str:
        """Generate code using Claude via Auto-Claude"""
        try:
            response = await self.http_client.post(
                f"{AUTO_CLAUDE_URL}/api/generate",
                json={
                    "prompt": prompt,
                    "context": context,
                    "language": language,
                    "model": "claude"
                }
            )
            if response.status_code == 200:
                return response.json().get("code", "")
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
        
        # Fallback - return prompt acknowledgment
        return f"# Generated code for: {prompt}\n# Context files: {len(context.split('###')) - 1}\n# TODO: Implement"
    
    async def _generate_openai(self, prompt: str, context: str, language: str) -> str:
        """Generate code using OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        try:
            response = await self.http_client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
                    "messages": [
                        {"role": "system", "content": f"You are a code generator. Output only code, no explanations. Language: {language or 'auto-detect'}"},
                        {"role": "user", "content": f"{context}\n\n{prompt}"}
                    ],
                    "max_tokens": 4000
                }
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
        
        return f"# OpenAI generation failed\n# Prompt: {prompt}"
    
    async def _generate_openrouter(self, prompt: str, context: str, language: str, model: str) -> str:
        """Generate code using OpenRouter"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        try:
            response = await self.http_client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://origin-os.local"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": f"You are a code generator. Output only code. Language: {language or 'auto'}"},
                        {"role": "user", "content": f"{context}\n\n{prompt}"}
                    ]
                }
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
        
        return f"# OpenRouter generation failed\n# Model: {model}"
    
    async def create_session(self, workspace_path: str) -> CursorSession:
        """Create a new Cursor session"""
        session_id = f"cursor-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        now = datetime.now(timezone.utc).isoformat()
        
        session = CursorSession(
            session_id=session_id,
            workspace_path=workspace_path,
            active_files=[],
            created_at=now,
            last_activity=now
        )
        
        self.sessions[session_id] = session
        
        # Save session to disk
        session_file = DATA_DIR / f"{session_id}.json"
        session_file.write_text(session.model_dump_json(indent=2))
        
        return session


# Global client instance
cursor_client = CursorClient()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "cursor-mcp"}


@app.get("/info")
async def info():
    return {
        "service": "cursor-mcp",
        "version": "1.0.0",
        "capabilities": [
            "file_read",
            "file_write",
            "file_edit",
            "file_search",
            "code_generate",
            "workspace_manage",
            "session_manage"
        ],
        "models": {
            "claude": ["claude-sonnet-4", "claude-opus-4"],
            "openai": ["gpt-4o", "gpt-4-turbo", "o1", "o3"],
            "openrouter": ["any model via openrouter"]
        }
    }


# -----------------------------------------------------------------------------
# FILE OPERATIONS
# -----------------------------------------------------------------------------

@app.post("/file/read")
async def read_file(request: FileRead):
    """Read file contents"""
    try:
        content = await cursor_client.read_file(request.path)
        return {"path": request.path, "content": content}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/file/write")
async def write_file(request: FileEdit):
    """Write file contents"""
    try:
        await cursor_client.write_file(
            request.path,
            request.content,
            request.create_if_missing
        )
        return {"path": request.path, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/file/list")
async def list_files(directory: str, pattern: str = "*"):
    """List files in directory"""
    files = await cursor_client.list_files(directory, pattern)
    return {"directory": directory, "files": files, "count": len(files)}


@app.post("/file/search")
async def search_files(directory: str, query: str):
    """Search for text in files"""
    results = await cursor_client.search_files(directory, query)
    return {"directory": directory, "query": query, "results": results}


# -----------------------------------------------------------------------------
# CODE GENERATION
# -----------------------------------------------------------------------------

@app.post("/generate")
async def generate_code(request: CodeGenRequest):
    """Generate code using specified model"""
    try:
        code = await cursor_client.generate_code(
            prompt=request.prompt,
            context_files=request.context_files,
            language=request.language,
            model=request.model
        )
        return {
            "prompt": request.prompt,
            "model": request.model,
            "code": code
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# SESSION MANAGEMENT
# -----------------------------------------------------------------------------

@app.post("/session/create")
async def create_session(workspace_path: str):
    """Create a new Cursor session"""
    session = await cursor_client.create_session(workspace_path)
    return session.model_dump()


@app.get("/session/list")
async def list_sessions():
    """List all sessions"""
    sessions = []
    for session_file in DATA_DIR.glob("cursor-*.json"):
        try:
            session_data = json.loads(session_file.read_text())
            sessions.append(session_data)
        except:
            pass
    return {"sessions": sessions}


# -----------------------------------------------------------------------------
# MCP TOOL INTERFACE
# -----------------------------------------------------------------------------

@app.post("/mcp/tool")
async def mcp_tool_call(request: MCPToolCall):
    """Execute MCP tool call"""
    tool = request.tool
    params = request.params
    
    try:
        if tool == "read_file":
            content = await cursor_client.read_file(params["path"])
            return {"result": content}
        
        elif tool == "write_file":
            await cursor_client.write_file(
                params["path"],
                params["content"],
                params.get("create", True)
            )
            return {"result": "success"}
        
        elif tool == "edit_file":
            await cursor_client.edit_file(
                params["path"],
                params["old_text"],
                params["new_text"]
            )
            return {"result": "success"}
        
        elif tool == "search":
            results = await cursor_client.search_files(
                params["directory"],
                params["query"]
            )
            return {"result": results}
        
        elif tool == "generate":
            code = await cursor_client.generate_code(
                prompt=params["prompt"],
                context_files=params.get("context_files", []),
                language=params.get("language"),
                model=params.get("model", "claude")
            )
            return {"result": code}
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# INTEGRATION WITH ORIGIN OS
# -----------------------------------------------------------------------------

@app.post("/integrate/auto-claude")
async def integrate_auto_claude(task: str, project_path: str):
    """Send task to Auto-Claude for autonomous coding"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AUTO_CLAUDE_URL}/api/task",
            json={
                "task": task,
                "project_path": project_path,
                "source": "cursor-mcp"
            }
        )
        return response.json()


@app.post("/integrate/codex")
async def integrate_codex(action: str, params: Dict[str, Any]):
    """Check action against Codex governance"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{CODEX_URL}/api/enforce",
            json={
                "action": action,
                "params": params,
                "source": "cursor-mcp"
            }
        )
        return response.json()


@app.post("/integrate/vault")
async def integrate_vault(operation: str, key: str, value: str = None):
    """Store/retrieve from Vault"""
    async with httpx.AsyncClient() as client:
        if operation == "get":
            response = await client.get(f"{VAULT_URL}/api/secrets/{key}")
        elif operation == "set":
            response = await client.post(
                f"{VAULT_URL}/api/secrets",
                json={"key": key, "value": value}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {operation}")
        return response.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
