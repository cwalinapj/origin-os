#!/usr/bin/env python3
"""
TOOL REGISTRY — Available Tools & Plugins for LLM Agents
=========================================================

Central registry of all tools available to LLM agents.
Each agent can query this to discover capabilities and request plugins.
"""

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from enum import Enum


@dataclass
class Tool:
    id: str
    name: str
    description: str
    category: str  # core, plugin, local, experimental
    env_key: Optional[str] = None
    plugin_id: Optional[str] = None
    endpoint: Optional[str] = None
    docs_url: Optional[str] = None
    requires: List[str] = field(default_factory=list)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOLS: Dict[str, Tool] = {
    
    # === CORE (Always Available) ===
    "llm_gateway": Tool("llm_gateway", "LLM Gateway", "Multi-provider LLM routing", "core", endpoint="http://llm-gateway:8200"),
    "vector_db": Tool("vector_db", "Pinecone Vector DB", "Semantic search and embeddings", "core", env_key="PINECONE_API_KEY"),
    "s3_storage": Tool("s3_storage", "AWS S3", "Artifact storage", "core", env_key="AWS_ACCESS_KEY_ID"),
    "redis_cache": Tool("redis_cache", "Redis", "Cache and state", "core", endpoint="redis://redis:6379"),
    "mongo_store": Tool("mongo_store", "MongoDB", "Document storage", "core", endpoint="mongodb://mongo:27017"),
    
    # === LLM PROVIDERS ===
    "anthropic": Tool("anthropic", "Anthropic Claude", "Claude models", "plugin", env_key="ANTHROPIC_API_KEY", plugin_id="llm.anthropic", docs_url="https://docs.anthropic.com"),
    "openai": Tool("openai", "OpenAI GPT", "GPT-4 and DALL-E", "plugin", env_key="OPENAI_API_KEY", plugin_id="llm.openai", docs_url="https://platform.openai.com/docs"),
    "google_ai": Tool("google_ai", "Google Gemini", "Multimodal AI", "plugin", env_key="GOOGLE_API_KEY", plugin_id="llm.google", docs_url="https://ai.google.dev"),
    "mistral": Tool("mistral", "Mistral AI", "Multilingual models", "plugin", env_key="MISTRAL_API_KEY", plugin_id="llm.mistral", docs_url="https://docs.mistral.ai"),
    
    # === SEARCH & RESEARCH ===
    "web_search": Tool("web_search", "Web Search (SerpAPI)", "Search the web", "plugin", env_key="SERPAPI_KEY", plugin_id="search.serp", docs_url="https://serpapi.com"),
    "tavily_search": Tool("tavily_search", "Tavily Search", "AI-optimized search", "plugin", env_key="TAVILY_API_KEY", plugin_id="search.tavily", docs_url="https://tavily.com"),
    "exa_search": Tool("exa_search", "Exa Search", "Neural search", "plugin", env_key="EXA_API_KEY", plugin_id="search.exa", docs_url="https://exa.ai"),
    "perplexity": Tool("perplexity", "Perplexity", "Research assistant", "plugin", env_key="PERPLEXITY_API_KEY", plugin_id="search.perplexity", docs_url="https://docs.perplexity.ai"),
    
    # === BROWSER & AUTOMATION ===
    "browser_use": Tool("browser_use", "Browser Use", "Automated browser control", "plugin", env_key="BROWSERBASE_API_KEY", plugin_id="browser.use", docs_url="https://browser-use.com"),
    "playwright": Tool("playwright", "Playwright", "Browser automation", "local", plugin_id="browser.playwright", docs_url="https://playwright.dev"),
    "firecrawl": Tool("firecrawl", "Firecrawl", "Web scraping", "plugin", env_key="FIRECRAWL_API_KEY", plugin_id="scrape.firecrawl", docs_url="https://firecrawl.dev"),
    "apify": Tool("apify", "Apify", "Web scraping platform", "plugin", env_key="APIFY_TOKEN", plugin_id="scrape.apify", docs_url="https://apify.com/docs"),
    
    # === CODE & DEVELOPMENT ===
    "github": Tool("github", "GitHub", "Repository management", "plugin", env_key="GITHUB_TOKEN", plugin_id="code.github", docs_url="https://docs.github.com"),
    "code_interpreter": Tool("code_interpreter", "Code Interpreter", "Python sandbox", "local", plugin_id="code.interpreter"),
    "e2b_sandbox": Tool("e2b_sandbox", "E2B Sandbox", "Secure code execution", "plugin", env_key="E2B_API_KEY", plugin_id="code.e2b", docs_url="https://e2b.dev"),
    "replit": Tool("replit", "Replit", "Cloud IDE", "plugin", env_key="REPLIT_TOKEN", plugin_id="code.replit", docs_url="https://docs.replit.com"),
    
    # === DATA & ANALYTICS ===
    "google_analytics": Tool("google_analytics", "Google Analytics", "Website analytics", "plugin", env_key="GA_API_KEY", plugin_id="analytics.ga", docs_url="https://developers.google.com/analytics"),
    "google_ads": Tool("google_ads", "Google Ads", "Ad campaigns", "plugin", env_key="GOOGLE_ADS_DEVELOPER_TOKEN", plugin_id="ads.google", requires=["GOOGLE_ADS_CLIENT_ID", "GOOGLE_ADS_CLIENT_SECRET"]),
    "bigquery": Tool("bigquery", "BigQuery", "Data warehouse", "plugin", env_key="GOOGLE_APPLICATION_CREDENTIALS", plugin_id="data.bigquery"),
    "snowflake": Tool("snowflake", "Snowflake", "Cloud data platform", "plugin", env_key="SNOWFLAKE_ACCOUNT", plugin_id="data.snowflake"),
    
    # === COMMUNICATION ===
    "slack": Tool("slack", "Slack", "Messaging", "plugin", env_key="SLACK_BOT_TOKEN", plugin_id="comms.slack", docs_url="https://api.slack.com"),
    "discord": Tool("discord", "Discord", "Community messaging", "plugin", env_key="DISCORD_BOT_TOKEN", plugin_id="comms.discord", docs_url="https://discord.com/developers"),
    "email_sendgrid": Tool("email_sendgrid", "SendGrid", "Email sending", "plugin", env_key="SENDGRID_API_KEY", plugin_id="comms.sendgrid"),
    "twilio": Tool("twilio", "Twilio", "SMS/Voice", "plugin", env_key="TWILIO_AUTH_TOKEN", plugin_id="comms.twilio", requires=["TWILIO_ACCOUNT_SID"]),
    
    # === IMAGE & MEDIA ===
    "dalle": Tool("dalle", "DALL-E", "Image generation", "plugin", env_key="OPENAI_API_KEY", plugin_id="image.dalle"),
    "stability": Tool("stability", "Stability AI", "Stable Diffusion", "plugin", env_key="STABILITY_API_KEY", plugin_id="image.stability"),
    "replicate": Tool("replicate", "Replicate", "ML model hosting", "plugin", env_key="REPLICATE_API_TOKEN", plugin_id="ml.replicate", docs_url="https://replicate.com"),
    "cloudinary": Tool("cloudinary", "Cloudinary", "Media management", "plugin", env_key="CLOUDINARY_URL", plugin_id="media.cloudinary"),
    "eleven_labs": Tool("eleven_labs", "ElevenLabs", "Voice synthesis", "plugin", env_key="ELEVEN_LABS_API_KEY", plugin_id="audio.elevenlabs"),
    
    # === DOCUMENTS ===
    "notion": Tool("notion", "Notion", "Knowledge base", "plugin", env_key="NOTION_API_KEY", plugin_id="docs.notion"),
    "airtable": Tool("airtable", "Airtable", "Structured data", "plugin", env_key="AIRTABLE_API_KEY", plugin_id="data.airtable"),
    "google_docs": Tool("google_docs", "Google Docs", "Document editing", "plugin", env_key="GOOGLE_APPLICATION_CREDENTIALS", plugin_id="docs.google"),
    "google_sheets": Tool("google_sheets", "Google Sheets", "Spreadsheets", "plugin", env_key="GOOGLE_APPLICATION_CREDENTIALS", plugin_id="sheets.google"),
    
    # === PAYMENTS & COMMERCE ===
    "stripe": Tool("stripe", "Stripe", "Payments", "plugin", env_key="STRIPE_SECRET_KEY", plugin_id="payments.stripe", docs_url="https://stripe.com/docs"),
    "shopify": Tool("shopify", "Shopify", "E-commerce", "plugin", env_key="SHOPIFY_ACCESS_TOKEN", plugin_id="commerce.shopify"),
    
    # === MCP SERVERS ===
    "mcp_filesystem": Tool("mcp_filesystem", "MCP Filesystem", "File operations", "local", plugin_id="mcp.filesystem"),
    "mcp_memory": Tool("mcp_memory", "MCP Memory", "Persistent memory", "local", plugin_id="mcp.memory"),
    "mcp_sequential_thinking": Tool("mcp_sequential_thinking", "MCP Sequential Thinking", "Step-by-step reasoning", "local", plugin_id="mcp.thinking"),
    "mcp_docker": Tool("mcp_docker", "MCP Docker", "Container management", "local", plugin_id="mcp.docker"),
}


def get_all_tools() -> Dict[str, dict]:
    return {k: asdict(v) for k, v in TOOLS.items()}

def get_tools_by_category(category: str) -> Dict[str, dict]:
    return {k: asdict(v) for k, v in TOOLS.items() if v.category == category}

def get_tool(tool_id: str) -> Optional[dict]:
    tool = TOOLS.get(tool_id)
    return asdict(tool) if tool else None

def check_requirements(tool_id: str, env_vars: Dict[str, str]) -> dict:
    tool = TOOLS.get(tool_id)
    if not tool:
        return {"met": False, "missing": [], "error": "Tool not found"}
    
    missing = []
    if tool.env_key and not env_vars.get(tool.env_key):
        missing.append(tool.env_key)
    for req in tool.requires:
        if not env_vars.get(req):
            missing.append(req)
    
    return {"met": len(missing) == 0, "missing": missing}


if __name__ == "__main__":
    print(json.dumps({"tools": get_all_tools()}, indent=2))
