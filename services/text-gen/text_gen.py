#!/usr/bin/env python3
"""
TEXT GEN MCP — AI Text Generation for Origin OS
================================================
Generates "lookalike" text that matches style, tone, and format.

Capabilities:
- Style matching (analyze and replicate writing style)
- Brand voice cloning
- Content variation (same message, different words)
- SEO text generation
- A/B test copy variants
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

DEFAULT_PROVIDER = os.getenv("TEXT_GEN_PROVIDER", "openai")

DATA_DIR = Path(os.getenv("TEXT_DATA_DIR", "/data/text"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text-gen")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Text Gen MCP",
    description="AI Text Generation with Style Matching",
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

class StyleAnalysis(BaseModel):
    tone: str  # formal, casual, professional, friendly
    voice: str  # active, passive
    complexity: str  # simple, moderate, complex
    sentence_length: str  # short, medium, long
    vocabulary: str  # basic, intermediate, advanced
    personality: List[str]  # witty, serious, empathetic, etc.
    brand_attributes: List[str] = []

class TextGenRequest(BaseModel):
    prompt: str
    style_reference: Optional[str] = None  # Text to match style
    style_analysis: Optional[StyleAnalysis] = None
    tone: Optional[str] = None
    length: str = "medium"  # short, medium, long
    format: str = "paragraph"  # paragraph, bullets, headers
    provider: str = "openai"
    model: Optional[str] = None
    variations: int = 1  # Number of variations

class LookalikeRequest(BaseModel):
    original_text: str
    preserve_meaning: bool = True
    preserve_length: bool = True
    variations: int = 3
    provider: str = "openai"

class BrandVoiceRequest(BaseModel):
    brand_name: str
    sample_texts: List[str]  # Examples of brand's writing
    new_topic: str
    format: str = "paragraph"
    provider: str = "openai"

class SEOTextRequest(BaseModel):
    topic: str
    keywords: List[str]
    target_length: int = 500  # words
    style: str = "informative"
    provider: str = "openai"

class ABCopyRequest(BaseModel):
    base_message: str
    num_variants: int = 3
    target_audience: Optional[str] = None
    goal: str = "conversion"  # conversion, engagement, awareness
    provider: str = "openai"

# =============================================================================
# TEXT GENERATION SERVICE
# =============================================================================

class TextGenService:
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.generation_count = 0
        self.style_cache: Dict[str, StyleAnalysis] = {}
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=60.0)
        return self.http_client
    
    async def analyze_style(self, text: str) -> StyleAnalysis:
        """Analyze writing style of text"""
        prompt = f"""Analyze the writing style of this text and return a JSON object with:
- tone: formal, casual, professional, or friendly
- voice: active or passive
- complexity: simple, moderate, or complex
- sentence_length: short, medium, or long
- vocabulary: basic, intermediate, or advanced
- personality: list of traits like witty, serious, empathetic, etc.
- brand_attributes: any brand-specific traits

Text to analyze:
{text}

Return only valid JSON."""

        response = await self._call_llm(prompt, "openai", "gpt-4o")
        
        try:
            data = json.loads(response)
            return StyleAnalysis(**data)
        except:
            return StyleAnalysis(
                tone="neutral",
                voice="active",
                complexity="moderate",
                sentence_length="medium",
                vocabulary="intermediate",
                personality=["professional"]
            )
    
    async def generate(self, request: TextGenRequest) -> List[str]:
        """Generate text with optional style matching"""
        
        # Analyze style reference if provided
        style = request.style_analysis
        if request.style_reference and not style:
            style = await self.analyze_style(request.style_reference)
        
        # Build prompt
        system_prompt = self._build_style_prompt(style, request.tone, request.format)
        
        user_prompt = f"""Generate {request.length} text about: {request.prompt}

Format: {request.format}
Length: {request.length}"""

        results = []
        for _ in range(request.variations):
            text = await self._call_llm(
                user_prompt,
                request.provider,
                request.model,
                system_prompt
            )
            results.append(text)
            self.generation_count += 1
        
        return results
    
    async def generate_lookalike(self, request: LookalikeRequest) -> List[str]:
        """Generate text that looks like the original but uses different words"""
        
        style = await self.analyze_style(request.original_text)
        
        prompt = f"""Rewrite this text to say the same thing but with completely different words.
{"Preserve the approximate length." if request.preserve_length else ""}
{"Keep the exact same meaning." if request.preserve_meaning else ""}

Match this style:
- Tone: {style.tone}
- Voice: {style.voice}
- Complexity: {style.complexity}
- Sentence length: {style.sentence_length}

Original text:
{request.original_text}

Rewrite:"""

        results = []
        for _ in range(request.variations):
            text = await self._call_llm(prompt, request.provider)
            results.append(text)
            self.generation_count += 1
        
        return results
    
    async def generate_brand_voice(self, request: BrandVoiceRequest) -> str:
        """Generate text in brand's voice"""
        
        # Analyze brand samples
        combined_samples = "\n\n---\n\n".join(request.sample_texts)
        style = await self.analyze_style(combined_samples)
        
        # Cache for future use
        self.style_cache[request.brand_name] = style
        
        prompt = f"""You are writing as {request.brand_name}. Match this exact style:

Brand voice characteristics:
- Tone: {style.tone}
- Voice: {style.voice}
- Complexity: {style.complexity}
- Personality: {', '.join(style.personality)}
- Attributes: {', '.join(style.brand_attributes)}

Sample of how {request.brand_name} writes:
{request.sample_texts[0][:500]}

Now write about: {request.new_topic}
Format: {request.format}"""

        text = await self._call_llm(prompt, request.provider)
        self.generation_count += 1
        return text
    
    async def generate_seo_text(self, request: SEOTextRequest) -> str:
        """Generate SEO-optimized text"""
        
        keywords_str = ", ".join(request.keywords)
        
        prompt = f"""Write a {request.target_length}-word {request.style} article about: {request.topic}

Naturally incorporate these keywords: {keywords_str}

Guidelines:
- Use keywords naturally, don't stuff them
- Include keywords in first paragraph and headings
- Write for humans first, search engines second
- Use clear, scannable formatting"""

        text = await self._call_llm(prompt, request.provider)
        self.generation_count += 1
        return text
    
    async def generate_ab_variants(self, request: ABCopyRequest) -> List[Dict[str, Any]]:
        """Generate A/B test copy variants"""
        
        prompt = f"""Create {request.num_variants} different versions of this message for A/B testing.
Each version should be distinctly different in approach while conveying the same core message.

Original message: {request.base_message}
{"Target audience: " + request.target_audience if request.target_audience else ""}
Goal: {request.goal}

For each variant, explain what makes it different (emotional appeal, urgency, social proof, etc.)

Return as JSON array with objects containing:
- variant: the text
- approach: what technique it uses
- hypothesis: why it might perform better"""

        response = await self._call_llm(prompt, request.provider, "gpt-4o")
        
        try:
            variants = json.loads(response)
            self.generation_count += len(variants)
            return variants
        except:
            return [{"variant": request.base_message, "approach": "original", "hypothesis": "baseline"}]
    
    def _build_style_prompt(
        self,
        style: Optional[StyleAnalysis],
        tone: Optional[str],
        format: str
    ) -> str:
        """Build system prompt for style matching"""
        
        if not style and not tone:
            return "You are a helpful writing assistant."
        
        parts = ["You are a writing assistant that matches specific styles."]
        
        if style:
            parts.append(f"Tone: {style.tone}")
            parts.append(f"Voice: {style.voice}")
            parts.append(f"Complexity: {style.complexity}")
            parts.append(f"Sentence length: {style.sentence_length}")
            parts.append(f"Vocabulary level: {style.vocabulary}")
            if style.personality:
                parts.append(f"Personality: {', '.join(style.personality)}")
        
        if tone:
            parts.append(f"Override tone to: {tone}")
        
        parts.append(f"Output format: {format}")
        
        return "\n".join(parts)
    
    async def _call_llm(
        self,
        prompt: str,
        provider: str = "openai",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Call LLM provider"""
        
        client = await self._get_client()
        
        if provider == "openai":
            model = model or "gpt-4o"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": model, "messages": messages}
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        
        elif provider == "anthropic":
            model = model or "claude-sonnet-4-20250514"
            
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "system": system_prompt or "",
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"]
        
        elif provider == "openrouter":
            model = model or "anthropic/claude-sonnet-4"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={"model": model, "messages": messages}
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        
        else:
            raise HTTPException(400, f"Unknown provider: {provider}")


# Global service
text_service = TextGenService()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "generations": text_service.generation_count,
        "cached_styles": len(text_service.style_cache)
    }

@app.post("/analyze-style")
async def analyze_style(text: str):
    """Analyze writing style"""
    style = await text_service.analyze_style(text)
    return style.model_dump()

@app.post("/generate")
async def generate_text(request: TextGenRequest):
    """Generate text with style matching"""
    texts = await text_service.generate(request)
    return {"texts": texts}

@app.post("/lookalike")
async def generate_lookalike(request: LookalikeRequest):
    """Generate lookalike text"""
    texts = await text_service.generate_lookalike(request)
    return {"variations": texts}

@app.post("/brand-voice")
async def generate_brand_voice(request: BrandVoiceRequest):
    """Generate text in brand voice"""
    text = await text_service.generate_brand_voice(request)
    return {"text": text}

@app.post("/seo")
async def generate_seo_text(request: SEOTextRequest):
    """Generate SEO-optimized text"""
    text = await text_service.generate_seo_text(request)
    return {"text": text}

@app.post("/ab-variants")
async def generate_ab_variants(request: ABCopyRequest):
    """Generate A/B test variants"""
    variants = await text_service.generate_ab_variants(request)
    return {"variants": variants}

# MCP Tool Interface
@app.post("/mcp/tool")
async def mcp_tool(tool: str, params: Dict[str, Any]):
    """Execute MCP tool"""
    if tool == "generate_text":
        request = TextGenRequest(**params)
        texts = await text_service.generate(request)
        return {"texts": texts}
    elif tool == "lookalike":
        request = LookalikeRequest(**params)
        texts = await text_service.generate_lookalike(request)
        return {"variations": texts}
    elif tool == "analyze_style":
        style = await text_service.analyze_style(params["text"])
        return style.model_dump()
    else:
        raise HTTPException(400, f"Unknown tool: {tool}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
