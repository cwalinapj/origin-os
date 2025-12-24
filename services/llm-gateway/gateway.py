#!/usr/bin/env python3
"""
LLM GATEWAY — Multi-Provider Routing
=====================================

Routes LLM requests through appropriate gateways:
- Vercel AI Gateway: General LLM operations (reasoning, code, analysis)
- OpenRouter: Copy generation and image generation (model self-selection)

Each LLM can choose its preferred model for copy generation via OpenRouter,
enabling model-specific optimization for creative tasks.
"""

import os
import httpx
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger("llm-gateway")


# =============================================================================
# CONFIGURATION
# =============================================================================

VERCEL_AI_GATEWAY_URL = "https://ai-gateway.vercel.sh/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

VERCEL_API_KEY = os.getenv("VERCEL_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


class TaskType(Enum):
    """Types of LLM tasks with different routing."""
    REASONING = "reasoning"          # Vercel AI Gateway
    CODE_GENERATION = "code"         # Vercel AI Gateway
    ANALYSIS = "analysis"            # Vercel AI Gateway
    COPY_GENERATION = "copy"         # OpenRouter (model self-selects)
    IMAGE_GENERATION = "image"       # OpenRouter
    MUTATION_PLANNING = "mutation"   # Vercel AI Gateway
    EVALUATION = "evaluation"        # Vercel AI Gateway


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    tokens_used: int
    latency_ms: float
    metadata: Dict[str, Any]


# =============================================================================
# VERCEL AI GATEWAY — General LLM Operations
# =============================================================================

class VercelAIGateway:
    """
    Vercel AI Gateway client for general LLM operations.
    
    Supported models via Vercel:
    - anthropic/claude-3.5-sonnet
    - openai/gpt-4o
    - google/gemini-1.5-pro
    - meta/llama-3.1-405b
    """
    
    MODELS = {
        "claude": "anthropic/claude-sonnet-4-20250514",
        "gpt4": "openai/gpt-4o",
        "gemini": "google/gemini-2.0-flash",
        "llama": "meta/llama-3.1-405b-instruct"
    }
    
    def __init__(self, api_key: str = VERCEL_API_KEY):
        self.api_key = api_key
        self.base_url = VERCEL_AI_GATEWAY_URL
    
    async def complete(
        self,
        prompt: str,
        model: str = "claude",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> LLMResponse:
        """Send completion request through Vercel AI Gateway."""
        import time
        start = time.time()
        
        model_id = self.MODELS.get(model, self.MODELS["claude"])
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
        
        latency = (time.time() - start) * 1000
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=model_id,
            provider="vercel",
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            latency_ms=latency,
            metadata={"vercel_request_id": response.headers.get("x-request-id")}
        )


# =============================================================================
# OPENROUTER — Copy & Image Generation (Model Self-Selection)
# =============================================================================

class OpenRouterGateway:
    """
    OpenRouter client for copy and image generation.
    
    Allows each LLM to choose its preferred model for creative tasks.
    Models can self-select based on task requirements.
    """
    
    # Model preferences by task characteristic
    MODEL_PREFERENCES = {
        "persuasive": "anthropic/claude-3.5-sonnet",
        "creative": "openai/gpt-4o",
        "technical": "anthropic/claude-3.5-sonnet",
        "multilingual": "mistralai/mistral-large",
        "fast": "google/gemini-pro",
        "budget": "meta-llama/llama-3.1-70b-instruct"
    }
    
    IMAGE_MODELS = {
        "dalle3": "openai/dall-e-3",
        "sdxl": "stabilityai/stable-diffusion-xl"
    }
    
    def __init__(self, api_key: str = OPENROUTER_API_KEY):
        self.api_key = api_key
        self.base_url = OPENROUTER_URL
    
    async def generate_copy(
        self,
        prompt: str,
        style: str = "persuasive",
        model_override: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate marketing copy with model self-selection."""
        import time
        start = time.time()
        
        if model_override:
            model = model_override
        else:
            model = self.MODEL_PREFERENCES.get(style, "anthropic/claude-3.5-sonnet")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://origin-os.ai",
                    "X-Title": "Origin OS LAM Forge",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
        
        latency = (time.time() - start) * 1000
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            provider="openrouter",
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            latency_ms=latency,
            metadata={"style": style, "openrouter_id": data.get("id")}
        )
    
    async def select_model_for_task(
        self,
        task_description: str,
        available_models: Optional[List[str]] = None
    ) -> str:
        """Let an LLM choose the best model for a copy generation task."""
        if available_models is None:
            available_models = list(self.MODEL_PREFERENCES.values())
        
        selection_prompt = f"""Choose the best model for this copy task.

Task: {task_description}

Models: {', '.join(available_models)}

Respond with ONLY the model identifier."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemini-pro",
                    "messages": [{"role": "user", "content": selection_prompt}],
                    "temperature": 0.1,
                    "max_tokens": 100
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
        
        selected = data["choices"][0]["message"]["content"].strip()
        return selected if selected in available_models else "anthropic/claude-3.5-sonnet"


# =============================================================================
# UNIFIED LLM GATEWAY
# =============================================================================

class LLMGateway:
    """Unified gateway that routes requests to appropriate providers."""
    
    def __init__(self):
        self.vercel = VercelAIGateway()
        self.openrouter = OpenRouterGateway()
    
    async def complete(
        self,
        prompt: str,
        task_type: TaskType,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Route request to appropriate provider based on task type."""
        
        if task_type in [TaskType.COPY_GENERATION, TaskType.IMAGE_GENERATION]:
            return await self.openrouter.generate_copy(
                prompt=prompt,
                model_override=model,
                **kwargs
            )
        else:
            return await self.vercel.complete(
                prompt=prompt,
                model=model or "claude",
                **kwargs
            )
    
    async def generate_mutation(
        self,
        page_context: Dict[str, Any],
        mutation_intent: str,
        vertical: str
    ) -> Dict[str, Any]:
        """Generate page mutation using multi-model orchestration."""
        
        # Step 1: Plan mutation (Vercel)
        planning_prompt = f"""Plan structural changes for this page mutation.

Context: {json.dumps(page_context, indent=2)}
Intent: {mutation_intent}
Vertical: {vertical}

Output JSON: {{"changes": [], "copy_needs": [], "rationale": ""}}"""

        plan_response = await self.vercel.complete(
            prompt=planning_prompt,
            model="claude",
            temperature=0.3
        )
        
        try:
            plan = json.loads(plan_response.content)
        except json.JSONDecodeError:
            plan = {"changes": [], "copy_needs": [], "rationale": "Parse error"}
        
        # Step 2: Generate copy (OpenRouter with self-selection)
        copy_elements = {}
        for copy_need in plan.get("copy_needs", []):
            selected_model = await self.openrouter.select_model_for_task(
                task_description=f"Generate {copy_need['type']} for {vertical}"
            )
            
            copy_response = await self.openrouter.generate_copy(
                prompt=f"Generate {copy_need['type']}: {copy_need.get('context', '')}",
                model_override=selected_model
            )
            
            copy_elements[copy_need["id"]] = {
                "content": copy_response.content,
                "model": copy_response.model
            }
        
        return {
            "plan": plan,
            "copy_elements": copy_elements,
            "models_used": {
                "planning": "vercel/claude",
                "copy": list(set(c["model"] for c in copy_elements.values()))
            }
        }


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLM Gateway", version="1.0.0")
gateway = LLMGateway()


class CompletionRequest(BaseModel):
    prompt: str
    task_type: str = "reasoning"
    model: Optional[str] = None
    temperature: float = 0.7


class MutationRequest(BaseModel):
    page_context: Dict[str, Any]
    mutation_intent: str
    vertical: str = "ecommerce"


@app.get("/health")
async def health():
    return {"status": "healthy", "providers": ["vercel", "openrouter"]}


@app.post("/complete")
async def complete(request: CompletionRequest):
    try:
        task = TaskType(request.task_type)
    except ValueError:
        task = TaskType.REASONING
    
    response = await gateway.complete(
        prompt=request.prompt,
        task_type=task,
        model=request.model,
        temperature=request.temperature
    )
    
    return {
        "content": response.content,
        "model": response.model,
        "provider": response.provider
    }


@app.post("/mutation")
async def generate_mutation(request: MutationRequest):
    return await gateway.generate_mutation(
        page_context=request.page_context,
        mutation_intent=request.mutation_intent,
        vertical=request.vertical
    )


@app.get("/models")
async def list_models():
    return {
        "vercel": list(VercelAIGateway.MODELS.keys()),
        "openrouter": list(OpenRouterGateway.MODEL_PREFERENCES.values())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
