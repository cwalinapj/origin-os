#!/usr/bin/env python3
"""
LLM AGENT CONFIGURATION — Runtime Config
=========================================

Provides configuration for all 5 LLM agents with their
unique infrastructure resources.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class LLMAgentConfig:
    """Configuration for a single LLM agent."""
    name: str
    model_id: str
    gtag_id: str
    s3_bucket: str
    s3_url: str
    vector_namespace: str
    vector_db_url: str
    description: str


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

GA_API_KEY = os.getenv("GA_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "origin-os-lam")
VERCEL_API_KEY = os.getenv("VERCEL_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

AGENT_DESCRIPTIONS = {
    "claude": "Anthropic Claude - Nuanced, brand-safe copy",
    "gpt4": "OpenAI GPT-4o - Versatile, creative",
    "gemini": "Google Gemini - Fast, multimodal",
    "llama": "Meta LLaMA - Open-source, efficient",
    "mistral": "Mistral AI - European, multilingual"
}

AGENT_DEFAULTS = {
    "claude": {
        "model_id": "anthropic/claude-sonnet-4-20250514",
        "gtag_id": "GT-CLAUDE001",
        "s3_bucket": "origin-os-lam-claude",
        "vector_namespace": "claude"
    },
    "gpt4": {
        "model_id": "openai/gpt-4o",
        "gtag_id": "GT-GPT4O002",
        "s3_bucket": "origin-os-lam-gpt4",
        "vector_namespace": "gpt4"
    },
    "gemini": {
        "model_id": "google/gemini-2.0-flash",
        "gtag_id": "GT-GEMINI03",
        "s3_bucket": "origin-os-lam-gemini",
        "vector_namespace": "gemini"
    },
    "llama": {
        "model_id": "meta/llama-3.1-405b-instruct",
        "gtag_id": "GT-LLAMA004",
        "s3_bucket": "origin-os-lam-llama",
        "vector_namespace": "llama"
    },
    "mistral": {
        "model_id": "mistralai/mistral-large",
        "gtag_id": "GT-MISTRAL5",
        "s3_bucket": "origin-os-lam-mistral",
        "vector_namespace": "mistral"
    }
}

AGENT_NAMES = ["claude", "gpt4", "gemini", "llama", "mistral"]


# =============================================================================
# CONFIGURATION LOADERS
# =============================================================================

def get_agent_config(agent_name: str) -> LLMAgentConfig:
    """Get configuration for a specific LLM agent."""
    agent_upper = agent_name.upper()
    defaults = AGENT_DEFAULTS.get(agent_name, {})
    
    return LLMAgentConfig(
        name=agent_name,
        model_id=os.getenv(f"{agent_upper}_MODEL_ID", defaults.get("model_id", "")),
        gtag_id=os.getenv(f"{agent_upper}_GTAG_ID", defaults.get("gtag_id", "")),
        s3_bucket=os.getenv(f"{agent_upper}_S3_BUCKET", defaults.get("s3_bucket", "")),
        s3_url=os.getenv(f"{agent_upper}_S3_URL", f"s3://{defaults.get('s3_bucket', '')}"),
        vector_namespace=os.getenv(f"{agent_upper}_VECTOR_NAMESPACE", defaults.get("vector_namespace", agent_name)),
        vector_db_url=os.getenv(f"{agent_upper}_VECTOR_DB_URL", ""),
        description=AGENT_DESCRIPTIONS.get(agent_name, "")
    )


def get_all_agents() -> Dict[str, LLMAgentConfig]:
    """Get configuration for all LLM agents."""
    return {name: get_agent_config(name) for name in AGENT_NAMES}


# =============================================================================
# PRE-LOADED CONFIGURATIONS
# =============================================================================

CLAUDE = get_agent_config("claude")
GPT4 = get_agent_config("gpt4")
GEMINI = get_agent_config("gemini")
LLAMA = get_agent_config("llama")
MISTRAL = get_agent_config("mistral")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_s3_path(agent_name: str, artifact_type: str, filename: str) -> str:
    """Get full S3 path for an artifact."""
    config = get_agent_config(agent_name)
    return f"s3://{config.s3_bucket}/{artifact_type}/{filename}"


def get_vector_namespace(agent_name: str) -> str:
    """Get Pinecone namespace for an agent."""
    config = get_agent_config(agent_name)
    return config.vector_namespace


def get_gtag_for_page(agent_name: str) -> str:
    """Get GTAG ID for injecting into generated pages."""
    config = get_agent_config(agent_name)
    return config.gtag_id
