#!/usr/bin/env python3
"""
Cursor Client — HTTP Client for Cursor IDE API
================================================
Handles communication with Cursor IDE and AI services
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import httpx

logger = logging.getLogger("cursor-client")

# =============================================================================
# CURSOR API CLIENT
# =============================================================================

class CursorAPIClient:
    """
    Client for Cursor IDE API integration.
    
    Note: Cursor doesn't have a public REST API. This client provides:
    1. Local file system operations for code editing
    2. Integration with Claude/OpenAI for code generation
    3. WebSocket support for real-time collaboration (future)
    """
    
    def __init__(
        self,
        api_key: str = None,
        workspace_id: str = None,
        timeout: float = 60.0
    ):
        self.api_key = api_key or os.getenv("CURSOR_API_KEY", "")
        self.workspace_id = workspace_id or os.getenv("CURSOR_WORKSPACE_ID", "")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        
        # API endpoints
        self.anthropic_url = "https://api.anthropic.com/v1/messages"
        self.openai_url = "https://api.openai.com/v1/chat/completions"
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    # =========================================================================
    # CODE GENERATION
    # =========================================================================
    
    async def generate_with_claude(
        self,
        prompt: str,
        system: str = None,
        model: str = None,
        max_tokens: int = 4096
    ) -> str:
        """Generate code using Claude API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        system = system or "You are an expert programmer. Generate clean, well-documented code."
        
        client = await self._get_client()
        
        response = await client.post(
            self.anthropic_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]
    
    async def generate_with_openai(
        self,
        prompt: str,
        system: str = None,
        model: str = None,
        max_tokens: int = 4096
    ) -> str:
        """Generate code using OpenAI API"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        system = system or "You are an expert programmer. Generate clean, well-documented code."
        
        client = await self._get_client()
        
        response = await client.post(
            self.openai_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def generate_with_openrouter(
        self,
        prompt: str,
        system: str = None,
        model: str = None,
        max_tokens: int = 4096
    ) -> str:
        """Generate code using OpenRouter (access to multiple models)"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
        system = system or "You are an expert programmer. Generate clean, well-documented code."
        
        client = await self._get_client()
        
        response = await client.post(
            self.openrouter_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://origin-os.local",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def generate(
        self,
        prompt: str,
        provider: str = "claude",
        **kwargs
    ) -> str:
        """Generate code using specified provider"""
        if provider == "claude":
            return await self.generate_with_claude(prompt, **kwargs)
        elif provider in ["openai", "gpt-4", "gpt-4o"]:
            return await self.generate_with_openai(prompt, **kwargs)
        elif provider == "openrouter":
            return await self.generate_with_openrouter(prompt, **kwargs)
        else:
            # Try OpenRouter with the provider as model name
            return await self.generate_with_openrouter(prompt, model=provider, **kwargs)
    
    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================
    
    async def read_file(self, path: str) -> str:
        """Read file contents"""
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path.read_text()
    
    async def write_file(self, path: str, content: str) -> bool:
        """Write content to file"""
        file_path = Path(path).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return True
    
    async def apply_diff(self, path: str, old_text: str, new_text: str) -> bool:
        """Apply a diff to a file"""
        content = await self.read_file(path)
        if old_text not in content:
            raise ValueError(f"Text not found in {path}")
        new_content = content.replace(old_text, new_text, 1)
        await self.write_file(path, new_content)
        return True
    
    # =========================================================================
    # PROJECT ANALYSIS
    # =========================================================================
    
    async def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """Analyze a project structure"""
        path = Path(project_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_path}")
        
        analysis = {
            "path": str(path),
            "files": [],
            "languages": set(),
            "frameworks": [],
            "entry_points": []
        }
        
        # Count files by extension
        ext_count = {}
        for file in path.rglob("*"):
            if file.is_file() and not any(p.startswith(".") for p in file.parts):
                ext = file.suffix.lower()
                ext_count[ext] = ext_count.get(ext, 0) + 1
                analysis["files"].append(str(file.relative_to(path)))
        
        # Detect languages
        lang_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "React",
            ".tsx": "React TypeScript",
            ".go": "Go",
            ".rs": "Rust",
            ".java": "Java",
            ".rb": "Ruby",
            ".php": "PHP",
            ".swift": "Swift",
            ".kt": "Kotlin",
        }
        
        for ext, lang in lang_map.items():
            if ext in ext_count:
                analysis["languages"].add(lang)
        
        analysis["languages"] = list(analysis["languages"])
        analysis["file_count"] = len(analysis["files"])
        analysis["files"] = analysis["files"][:100]  # Limit for display
        
        return analysis


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_client(**kwargs) -> CursorAPIClient:
    """Create a new Cursor API client"""
    return CursorAPIClient(**kwargs)


async def quick_generate(prompt: str, provider: str = "claude") -> str:
    """Quick code generation helper"""
    client = CursorAPIClient()
    try:
        return await client.generate(prompt, provider=provider)
    finally:
        await client.close()
