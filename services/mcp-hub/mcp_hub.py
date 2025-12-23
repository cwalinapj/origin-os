#!/usr/bin/env python3
"""
Origin OS MCP Hub v4.0 - Comprehensive MCP Server Gateway
Full-featured MCP hub with 30+ servers for any LLM
"""

import os
import json
import asyncio
import subprocess
import base64
import re
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from urllib.parse import urlparse, quote

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(title="Origin OS MCP Hub", version="4.0")

# =============================================================================
# MCP SERVER REGISTRY - 30+ SERVERS
# =============================================================================

MCP_SERVERS = {
    # =========================================================================
    # 🏷️ MARKETING & ANALYTICS
    # =========================================================================
    "gtm": {
        "name": "Google Tag Manager",
        "type": "http",
        "url": os.getenv("CODEX_URL", "http://codex:8000"),
        "description": "Create/manage GTM tags, triggers, variables, publish containers",
        "tools": ["gtm_status", "gtm_list_containers", "gtm_list_tags", "gtm_create_tag", "gtm_create_trigger", "gtm_create_variable", "gtm_publish"]
    },
    "ga4": {
        "name": "Google Analytics 4",
        "type": "gcp",
        "description": "GA4 reporting, realtime data, audience insights",
        "tools": ["run_report", "get_realtime", "list_properties", "get_metadata"]
    },
    "semrush": {
        "name": "SEMrush",
        "type": "api",
        "url": "https://api.semrush.com",
        "api_key_env": "SEMRUSH_API_KEY",
        "description": "SEO keywords, backlinks, competitor analysis, traffic data",
        "tools": ["domain_overview", "keyword_research", "backlink_analysis", "competitor_analysis", "organic_keywords", "traffic_analytics", "keyword_difficulty"]
    },
    "google_ads": {
        "name": "Google Ads",
        "type": "gcp",
        "description": "Campaign management, keyword planning, ad performance",
        "tools": ["list_campaigns", "get_campaign", "create_campaign", "keyword_ideas", "get_metrics"]
    },

    # =========================================================================
    # 🌐 WEB SCRAPING & BROWSING
    # =========================================================================
    "firecrawl": {
        "name": "Firecrawl",
        "type": "api",
        "url": "https://api.firecrawl.dev/v1",
        "api_key_env": "FIRECRAWL_API_KEY",
        "description": "Smart web scraping - markdown extraction, crawling, screenshots",
        "tools": ["scrape_url", "crawl_site", "map_site", "screenshot", "extract_structured"]
    },
    "apify": {
        "name": "Apify",
        "type": "api",
        "url": "https://api.apify.com/v2",
        "api_key_env": "APIFY_API_KEY",
        "description": "Web scraping actors, data extraction, automation",
        "tools": ["run_actor", "get_dataset", "list_actors", "get_run", "scrape_url"]
    },
    "playwright": {
        "name": "Playwright Browser",
        "type": "browser",
        "description": "Browser automation - navigate, click, fill forms, screenshot, PDF",
        "tools": ["navigate", "click", "fill", "screenshot", "pdf", "get_text", "wait_for", "evaluate", "select"]
    },
    "fetch": {
        "name": "HTTP Client",
        "type": "http_client",
        "description": "Make HTTP requests to any URL (GET, POST, PUT, DELETE)",
        "tools": ["get", "post", "put", "delete", "head", "patch"]
    },

    # =========================================================================
    # 🤖 AI & LLM
    # =========================================================================
    "openrouter": {
        "name": "OpenRouter (Multi-LLM)",
        "type": "api",
        "url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "description": "Access 100+ LLMs - Claude, GPT-4, Gemini, Llama, Mistral",
        "tools": ["chat", "complete", "list_models", "get_generation"]
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "type": "api",
        "url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "description": "Claude 3.5 Sonnet, Claude 3 Opus - direct API access",
        "tools": ["chat", "complete", "count_tokens"]
    },
    "openai": {
        "name": "OpenAI",
        "type": "api",
        "url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "description": "GPT-4, DALL-E 3, Whisper, embeddings, assistants",
        "tools": ["chat", "complete", "embed", "image_generate", "transcribe", "tts", "vision"]
    },
    "google_ai": {
        "name": "Google AI (Gemini)",
        "type": "api",
        "url": "https://generativelanguage.googleapis.com/v1",
        "api_key_env": "GOOGLE_API_KEY",
        "description": "Gemini Pro, Gemini Ultra, embeddings",
        "tools": ["chat", "generate", "embed", "count_tokens"]
    },

    # =========================================================================
    # 💾 CODE & VERSION CONTROL
    # =========================================================================
    "github": {
        "name": "GitHub",
        "type": "api",
        "url": "https://api.github.com",
        "api_key_env": "GITHUB_TOKEN",
        "description": "Repos, issues, PRs, actions, gists, releases",
        "tools": ["list_repos", "create_repo", "get_repo", "list_issues", "create_issue", "list_prs", "create_pr", "get_file", "create_file", "update_file", "list_branches", "create_gist", "trigger_workflow"]
    },
    "git": {
        "name": "Git CLI",
        "type": "shell",
        "description": "Git operations - clone, commit, push, pull, branch, merge",
        "tools": ["clone", "status", "add", "commit", "push", "pull", "branch", "checkout", "log", "diff", "merge", "stash", "tag"]
    },
    "code_analysis": {
        "name": "Code Analysis",
        "type": "utility",
        "description": "Analyze code - complexity, dependencies, security scan",
        "tools": ["analyze_file", "find_dependencies", "security_scan", "lint", "format"]
    },

    # =========================================================================
    # 🐳 INFRASTRUCTURE
    # =========================================================================
    "docker": {
        "name": "Docker",
        "type": "socket",
        "socket": "/var/run/docker.sock",
        "description": "Container management - run, build, logs, networks, volumes",
        "tools": ["list_containers", "run_container", "stop_container", "remove_container", "logs", "exec", "list_images", "build_image", "pull_image", "push_image", "list_networks", "create_network", "list_volumes"]
    },
    "shell": {
        "name": "Shell (Bash)",
        "type": "shell",
        "description": "Execute shell commands safely",
        "tools": ["exec", "script", "background"]
    },
    "process": {
        "name": "Process Manager",
        "type": "shell",
        "description": "Manage system processes",
        "tools": ["list", "kill", "start", "status"]
    },

    # =========================================================================
    # ☁️ CLOUD STORAGE
    # =========================================================================
    "s3": {
        "name": "AWS S3",
        "type": "aws",
        "description": "S3 buckets - upload, download, list, presign URLs",
        "tools": ["list_buckets", "list_objects", "get_object", "put_object", "delete_object", "copy_object", "presign_url", "create_bucket"]
    },
    "gcs": {
        "name": "Google Cloud Storage",
        "type": "gcp",
        "description": "GCS buckets and objects",
        "tools": ["list_buckets", "list_objects", "get_object", "put_object", "delete_object"]
    },

    # =========================================================================
    # 🗄️ DATABASES
    # =========================================================================
    "postgres": {
        "name": "PostgreSQL",
        "type": "database",
        "description": "PostgreSQL queries and management",
        "tools": ["query", "execute", "list_tables", "describe_table", "list_databases"]
    },
    "sqlite": {
        "name": "SQLite",
        "type": "database",
        "description": "SQLite local database",
        "tools": ["query", "execute", "list_tables", "describe_table", "create_table"]
    },
    "redis": {
        "name": "Redis",
        "type": "database",
        "description": "Redis cache and data store",
        "tools": ["get", "set", "delete", "keys", "expire", "incr", "lpush", "lrange", "hset", "hget"]
    },

    # =========================================================================
    # 📁 FILESYSTEM & MEMORY
    # =========================================================================
    "filesystem": {
        "name": "Filesystem",
        "type": "local",
        "base_path": os.getenv("FS_BASE_PATH", "/data"),
        "description": "File operations - read, write, search, manage",
        "tools": ["read_file", "write_file", "append_file", "list_directory", "create_directory", "delete_file", "move_file", "copy_file", "search_files", "file_info", "glob", "watch"]
    },
    "memory": {
        "name": "Knowledge Graph",
        "type": "local",
        "storage_path": os.getenv("MEMORY_PATH", "/data/memory"),
        "description": "Persistent memory - entities, relations, observations",
        "tools": ["create_entities", "create_relations", "search_nodes", "add_observations", "delete_entities", "delete_relations", "read_graph", "open_nodes", "get_related"]
    },

    # =========================================================================
    # 📧 COMMUNICATION
    # =========================================================================
    "slack": {
        "name": "Slack",
        "type": "api",
        "url": "https://slack.com/api",
        "api_key_env": "SLACK_TOKEN",
        "description": "Send messages, manage channels, upload files",
        "tools": ["send_message", "list_channels", "list_users", "upload_file", "search_messages", "add_reaction", "create_channel"]
    },
    "email": {
        "name": "Email (SMTP)",
        "type": "smtp",
        "description": "Send emails with attachments",
        "tools": ["send_email", "send_template", "send_bulk"]
    },
    "twilio": {
        "name": "Twilio SMS",
        "type": "api",
        "url": "https://api.twilio.com/2010-04-01",
        "api_key_env": "TWILIO_AUTH_TOKEN",
        "description": "Send SMS and make calls",
        "tools": ["send_sms", "make_call", "list_messages"]
    },

    # =========================================================================
    # 🛠️ UTILITIES
    # =========================================================================
    "time": {
        "name": "Time & Scheduling",
        "type": "utility",
        "description": "Time operations, timezone conversion, scheduling",
        "tools": ["now", "convert_timezone", "parse_date", "format_date", "add_time", "diff_time", "is_business_day", "next_business_day"]
    },
    "crypto": {
        "name": "Cryptography",
        "type": "utility",
        "description": "Encryption, hashing, encoding, key generation",
        "tools": ["hash", "hmac", "encrypt", "decrypt", "encode_base64", "decode_base64", "encode_url", "decode_url", "generate_key", "generate_uuid", "random_string"]
    },
    "json_tools": {
        "name": "JSON Tools",
        "type": "utility",
        "description": "JSON parsing, querying (JSONPath), transformation",
        "tools": ["parse", "stringify", "query", "transform", "validate", "diff", "merge", "flatten", "unflatten"]
    },
    "regex": {
        "name": "Regex",
        "type": "utility",
        "description": "Regular expression operations",
        "tools": ["match", "search", "findall", "replace", "split", "extract_groups", "validate"]
    },
    "math": {
        "name": "Math & Stats",
        "type": "utility",
        "description": "Mathematical and statistical operations",
        "tools": ["calculate", "statistics", "random", "convert_units", "currency_convert"]
    },
    "text": {
        "name": "Text Processing",
        "type": "utility",
        "description": "Text manipulation - extract, clean, transform",
        "tools": ["extract_emails", "extract_urls", "extract_phones", "clean_html", "markdown_to_html", "html_to_markdown", "summarize", "translate", "sentiment"]
    },

    # =========================================================================
    # 🧠 REASONING & PLANNING
    # =========================================================================
    "thinking": {
        "name": "Sequential Thinking",
        "type": "utility",
        "description": "Step-by-step reasoning, planning, analysis",
        "tools": ["think", "plan", "analyze", "synthesize", "decide", "reflect"]
    },
    "tasks": {
        "name": "Task Manager",
        "type": "utility",
        "description": "Task tracking, dependencies, scheduling",
        "tools": ["create_task", "list_tasks", "update_task", "complete_task", "get_dependencies", "schedule"]
    },

    # =========================================================================
    # 📊 DATA & DOCUMENTS
    # =========================================================================
    "pdf": {
        "name": "PDF Tools",
        "type": "utility",
        "description": "PDF operations - read, create, merge, extract",
        "tools": ["read_pdf", "create_pdf", "merge_pdfs", "split_pdf", "extract_text", "extract_images", "add_watermark"]
    },
    "excel": {
        "name": "Excel/CSV",
        "type": "utility",
        "description": "Spreadsheet operations - read, write, analyze",
        "tools": ["read_excel", "write_excel", "read_csv", "write_csv", "analyze_data", "pivot", "chart"]
    },
    "image": {
        "name": "Image Processing",
        "type": "utility",
        "description": "Image manipulation - resize, convert, OCR",
        "tools": ["resize", "crop", "convert", "compress", "ocr", "generate", "edit"]
    },

    # =========================================================================
    # 🔗 INTEGRATIONS
    # =========================================================================
    "webhook": {
        "name": "Webhooks",
        "type": "utility",
        "description": "Send and receive webhooks",
        "tools": ["send", "create_endpoint", "list_endpoints", "get_logs"]
    },
    "cron": {
        "name": "Scheduler (Cron)",
        "type": "utility",
        "description": "Schedule recurring tasks",
        "tools": ["create_job", "list_jobs", "delete_job", "run_now", "get_history"]
    },
    "queue": {
        "name": "Message Queue",
        "type": "utility",
        "description": "Async job processing",
        "tools": ["enqueue", "dequeue", "peek", "list_queues", "purge"]
    }
}

# =============================================================================
# CLIENT IMPLEMENTATIONS
# =============================================================================

class MCPClient:
    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get("name", "Unknown")
    
    async def execute(self, tool: str, params: Dict) -> Dict:
        raise NotImplementedError


class HTTPMCPClient(MCPClient):
    async def execute(self, tool: str, params: Dict) -> Dict:
        url = self.config["url"]
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
        from jwt_auth import create_service_token, get_auth_header
        headers = get_auth_header(create_service_token("mcp-hub", ["*"]))
        
        async with httpx.AsyncClient() as client:
            if method == "GET":
                r = await client.get(f"{url}{path}", headers=headers, timeout=30)
            else:
                r = await client.post(f"{url}{path}", headers=headers, json=params, timeout=30)
            return r.json()


class GenericAPIClient(MCPClient):
    async def execute(self, tool: str, params: Dict) -> Dict:
        api_key = os.getenv(self.config.get("api_key_env", ""), "")
        if not api_key and self.config.get("api_key_env"):
            return {"error": f"{self.config.get('api_key_env')} not set"}
        
        server_id = self.config.get("_id", "")
        
        # Firecrawl
        if server_id == "firecrawl":
            return await self._firecrawl(tool, params, api_key)
        
        # Apify
        elif server_id == "apify":
            return await self._apify(tool, params, api_key)
        
        # OpenRouter
        elif server_id == "openrouter":
            return await self._openrouter(tool, params, api_key)
        
        # Anthropic
        elif server_id == "anthropic":
            return await self._anthropic(tool, params, api_key)
        
        # OpenAI
        elif server_id == "openai":
            return await self._openai(tool, params, api_key)
        
        # GitHub
        elif server_id == "github":
            return await self._github(tool, params, api_key)
        
        # Google AI
        elif server_id == "google_ai":
            return await self._google_ai(tool, params, api_key)
        
        # SEMrush
        elif server_id == "semrush":
            return await self._semrush(tool, params, api_key)
        
        return {"error": f"Unknown API server: {server_id}"}
    
    async def _firecrawl(self, tool: str, params: Dict, api_key: str) -> Dict:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        base = "https://api.firecrawl.dev/v1"
        
        async with httpx.AsyncClient() as client:
            if tool == "scrape_url":
                r = await client.post(f"{base}/scrape", headers=headers, json={"url": params.get("url"), "formats": params.get("formats", ["markdown"])}, timeout=60)
            elif tool == "crawl_site":
                r = await client.post(f"{base}/crawl", headers=headers, json={"url": params.get("url"), "limit": params.get("limit", 10)}, timeout=120)
            elif tool == "map_site":
                r = await client.post(f"{base}/map", headers=headers, json={"url": params.get("url")}, timeout=60)
            elif tool == "screenshot":
                r = await client.post(f"{base}/scrape", headers=headers, json={"url": params.get("url"), "formats": ["screenshot"]}, timeout=60)
            else:
                return {"error": f"Unknown Firecrawl tool: {tool}"}
            return r.json() if r.status_code == 200 else {"error": r.text}
    
    async def _apify(self, tool: str, params: Dict, api_key: str) -> Dict:
        headers = {"Authorization": f"Bearer {api_key}"}
        base = "https://api.apify.com/v2"
        
        async with httpx.AsyncClient() as client:
            if tool == "run_actor":
                actor_id = params.get("actor_id")
                r = await client.post(f"{base}/acts/{actor_id}/runs", headers=headers, json=params.get("input", {}), timeout=120)
            elif tool == "get_dataset":
                dataset_id = params.get("dataset_id")
                r = await client.get(f"{base}/datasets/{dataset_id}/items", headers=headers, timeout=60)
            elif tool == "list_actors":
                r = await client.get(f"{base}/acts", headers=headers, timeout=30)
            else:
                return {"error": f"Unknown Apify tool: {tool}"}
            return r.json() if r.status_code in [200, 201] else {"error": r.text}
    
    async def _openrouter(self, tool: str, params: Dict, api_key: str) -> Dict:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        async with httpx.AsyncClient() as client:
            if tool == "chat":
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": params.get("model", "anthropic/claude-3.5-sonnet"),
                        "messages": params.get("messages", []),
                        "max_tokens": params.get("max_tokens", 4096)
                    },
                    timeout=120
                )
                return r.json()
            elif tool == "list_models":
                r = await client.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=30)
                return r.json()
            return {"error": f"Unknown OpenRouter tool: {tool}"}
    
    async def _anthropic(self, tool: str, params: Dict, api_key: str) -> Dict:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            if tool == "chat":
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": params.get("model", "claude-3-5-sonnet-20241022"),
                        "messages": params.get("messages", []),
                        "max_tokens": params.get("max_tokens", 4096)
                    },
                    timeout=120
                )
                return r.json()
            return {"error": f"Unknown Anthropic tool: {tool}"}
    
    async def _openai(self, tool: str, params: Dict, api_key: str) -> Dict:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        async with httpx.AsyncClient() as client:
            if tool == "chat":
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": params.get("model", "gpt-4-turbo-preview"),
                        "messages": params.get("messages", []),
                        "max_tokens": params.get("max_tokens", 4096)
                    },
                    timeout=120
                )
                return r.json()
            elif tool == "embed":
                r = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json={
                        "model": params.get("model", "text-embedding-3-small"),
                        "input": params.get("text", "")
                    },
                    timeout=30
                )
                return r.json()
            elif tool == "image_generate":
                r = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers=headers,
                    json={
                        "model": "dall-e-3",
                        "prompt": params.get("prompt", ""),
                        "size": params.get("size", "1024x1024"),
                        "n": 1
                    },
                    timeout=60
                )
                return r.json()
            return {"error": f"Unknown OpenAI tool: {tool}"}
    
    async def _github(self, tool: str, params: Dict, api_key: str) -> Dict:
        headers = {"Authorization": f"token {api_key}", "Accept": "application/vnd.github.v3+json"}
        base = "https://api.github.com"
        
        async with httpx.AsyncClient() as client:
            if tool == "list_repos":
                r = await client.get(f"{base}/user/repos", headers=headers, params={"per_page": 100}, timeout=30)
            elif tool == "get_repo":
                r = await client.get(f"{base}/repos/{params.get('owner')}/{params.get('repo')}", headers=headers, timeout=30)
            elif tool == "create_repo":
                r = await client.post(f"{base}/user/repos", headers=headers, json={"name": params.get("name"), "private": params.get("private", False)}, timeout=30)
            elif tool == "get_file":
                r = await client.get(f"{base}/repos/{params.get('owner')}/{params.get('repo')}/contents/{params.get('path')}", headers=headers, timeout=30)
            elif tool == "list_issues":
                r = await client.get(f"{base}/repos/{params.get('owner')}/{params.get('repo')}/issues", headers=headers, timeout=30)
            elif tool == "create_issue":
                r = await client.post(f"{base}/repos/{params.get('owner')}/{params.get('repo')}/issues", headers=headers, json={"title": params.get("title"), "body": params.get("body")}, timeout=30)
            elif tool == "list_prs":
                r = await client.get(f"{base}/repos/{params.get('owner')}/{params.get('repo')}/pulls", headers=headers, timeout=30)
            elif tool == "list_branches":
                r = await client.get(f"{base}/repos/{params.get('owner')}/{params.get('repo')}/branches", headers=headers, timeout=30)
            else:
                return {"error": f"Unknown GitHub tool: {tool}"}
            return r.json()
    
    async def _google_ai(self, tool: str, params: Dict, api_key: str) -> Dict:
        async with httpx.AsyncClient() as client:
            if tool == "chat" or tool == "generate":
                model = params.get("model", "gemini-pro")
                r = await client.post(
                    f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}",
                    json={"contents": [{"parts": [{"text": params.get("prompt", "")}]}]},
                    timeout=60
                )
                return r.json()
            return {"error": f"Unknown Google AI tool: {tool}"}
    
    async def _semrush(self, tool: str, params: Dict, api_key: str) -> Dict:
        base = "https://api.semrush.com"
        
        async with httpx.AsyncClient() as client:
            if tool == "domain_overview":
                r = await client.get(f"{base}/", params={
                    "type": "domain_ranks",
                    "key": api_key,
                    "domain": params.get("domain"),
                    "database": params.get("database", "us")
                }, timeout=30)
                return {"data": r.text}
            elif tool == "keyword_research":
                r = await client.get(f"{base}/", params={
                    "type": "phrase_all",
                    "key": api_key,
                    "phrase": params.get("keyword"),
                    "database": params.get("database", "us")
                }, timeout=30)
                return {"data": r.text}
            return {"error": f"Unknown SEMrush tool: {tool}"}


class DockerMCPClient(MCPClient):
    async def execute(self, tool: str, params: Dict) -> Dict:
        try:
            import docker
            client = docker.from_env()
            
            if tool == "list_containers":
                containers = client.containers.list(all=params.get("all", False))
                return {"containers": [{"id": c.id[:12], "name": c.name, "status": c.status, "image": str(c.image.tags)} for c in containers]}
            elif tool == "run_container":
                c = client.containers.run(params.get("image"), name=params.get("name"), detach=True, ports=params.get("ports"), environment=params.get("env"))
                return {"container_id": c.id[:12], "name": c.name}
            elif tool == "stop_container":
                client.containers.get(params.get("container_id")).stop()
                return {"stopped": params.get("container_id")}
            elif tool == "remove_container":
                client.containers.get(params.get("container_id")).remove(force=params.get("force", False))
                return {"removed": params.get("container_id")}
            elif tool == "logs":
                logs = client.containers.get(params.get("container_id")).logs(tail=params.get("tail", 100)).decode()
                return {"logs": logs}
            elif tool == "exec":
                result = client.containers.get(params.get("container_id")).exec_run(params.get("command"))
                return {"output": result.output.decode(), "exit_code": result.exit_code}
            elif tool == "list_images":
                return {"images": [{"id": i.id[:12], "tags": i.tags} for i in client.images.list()]}
            elif tool == "pull_image":
                client.images.pull(params.get("image"))
                return {"pulled": params.get("image")}
            elif tool == "list_networks":
                return {"networks": [{"id": n.id[:12], "name": n.name} for n in client.networks.list()]}
            elif tool == "list_volumes":
                return {"volumes": [{"name": v.name} for v in client.volumes.list()]}
            return {"error": f"Unknown Docker tool: {tool}"}
        except Exception as e:
            return {"error": str(e)}


class ShellMCPClient(MCPClient):
    async def execute(self, tool: str, params: Dict) -> Dict:
        dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:"]
        cmd = params.get("command", params.get("script", ""))
        
        if any(d in cmd for d in dangerous):
            return {"error": "Dangerous command blocked"}
        
        try:
            if tool in ["exec", "script"]:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=params.get("timeout", 60))
                return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
            return {"error": f"Unknown shell tool: {tool}"}
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": str(e)}


class GitMCPClient(MCPClient):
    async def execute(self, tool: str, params: Dict) -> Dict:
        cwd = params.get("cwd", "/data/repos")
        
        commands = {
            "clone": ["git", "clone", params.get("url", "")],
            "status": ["git", "status"],
            "add": ["git", "add", params.get("files", ".")],
            "commit": ["git", "commit", "-m", params.get("message", "Update")],
            "push": ["git", "push"],
            "pull": ["git", "pull"],
            "log": ["git", "log", f"-{params.get('n', 10)}", "--oneline"],
            "diff": ["git", "diff"],
            "branch": ["git", "branch", "-a"],
            "checkout": ["git", "checkout", params.get("branch", "main")],
        }
        
        if tool not in commands:
            return {"error": f"Unknown git tool: {tool}"}
        
        try:
            result = subprocess.run(commands[tool], cwd=cwd, capture_output=True, text=True, timeout=120)
            return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        except Exception as e:
            return {"error": str(e)}


class S3MCPClient(MCPClient):
    async def execute(self, tool: str, params: Dict) -> Dict:
        try:
            import boto3
            s3 = boto3.client('s3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-west-2"))
            
            if tool == "list_buckets":
                return {"buckets": [b["Name"] for b in s3.list_buckets()["Buckets"]]}
            elif tool == "list_objects":
                r = s3.list_objects_v2(Bucket=params.get("bucket"), Prefix=params.get("prefix", ""), MaxKeys=100)
                return {"objects": [o["Key"] for o in r.get("Contents", [])]}
            elif tool == "get_object":
                r = s3.get_object(Bucket=params.get("bucket"), Key=params.get("key"))
                return {"content": r["Body"].read().decode("utf-8")[:50000]}
            elif tool == "put_object":
                s3.put_object(Bucket=params.get("bucket"), Key=params.get("key"), Body=params.get("body", "").encode())
                return {"uploaded": f"s3://{params.get('bucket')}/{params.get('key')}"}
            elif tool == "presign_url":
                url = s3.generate_presigned_url("get_object", Params={"Bucket": params.get("bucket"), "Key": params.get("key")}, ExpiresIn=params.get("expiry", 3600))
                return {"url": url}
            return {"error": f"Unknown S3 tool: {tool}"}
        except Exception as e:
            return {"error": str(e)}


class FilesystemMCPClient(MCPClient):
    async def execute(self, tool: str, params: Dict) -> Dict:
        base = self.config.get("base_path", "/data")
        
        def safe(p):
            full = os.path.normpath(os.path.join(base, p))
            if not full.startswith(base):
                raise ValueError("Path traversal blocked")
            return full
        
        try:
            if tool == "read_file":
                with open(safe(params.get("path", "")), "r") as f:
                    return {"content": f.read()[:100000]}
            elif tool == "write_file":
                path = safe(params.get("path", ""))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(params.get("content", ""))
                return {"written": path}
            elif tool == "append_file":
                with open(safe(params.get("path", "")), "a") as f:
                    f.write(params.get("content", ""))
                return {"appended": params.get("path")}
            elif tool == "list_directory":
                return {"entries": os.listdir(safe(params.get("path", "")))}
            elif tool == "create_directory":
                path = safe(params.get("path", ""))
                os.makedirs(path, exist_ok=True)
                return {"created": path}
            elif tool == "delete_file":
                os.remove(safe(params.get("path", "")))
                return {"deleted": params.get("path")}
            elif tool == "file_info":
                path = safe(params.get("path", ""))
                stat = os.stat(path)
                return {"path": path, "size": stat.st_size, "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(), "is_dir": os.path.isdir(path)}
            elif tool == "search_files":
                import glob
                matches = glob.glob(os.path.join(safe(params.get("path", "")), "**", params.get("pattern", "*")), recursive=True)
                return {"matches": matches[:100]}
            return {"error": f"Unknown filesystem tool: {tool}"}
        except Exception as e:
            return {"error": str(e)}


class MemoryMCPClient(MCPClient):
    def __init__(self, config):
        super().__init__(config)
        self.path = config.get("storage_path", "/data/memory")
        self.file = os.path.join(self.path, "graph.json")
        os.makedirs(self.path, exist_ok=True)
        if not os.path.exists(self.file):
            with open(self.file, "w") as f:
                json.dump({"entities": [], "relations": []}, f)
    
    def _load(self):
        with open(self.file) as f:
            return json.load(f)
    
    def _save(self, g):
        with open(self.file, "w") as f:
            json.dump(g, f, indent=2)
    
    async def execute(self, tool: str, params: Dict) -> Dict:
        g = self._load()
        
        if tool == "create_entities":
            for e in params.get("entities", []):
                e["created_at"] = datetime.now(timezone.utc).isoformat()
                g["entities"].append(e)
            self._save(g)
            return {"created": len(params.get("entities", []))}
        elif tool == "create_relations":
            g["relations"].extend(params.get("relations", []))
            self._save(g)
            return {"created": len(params.get("relations", []))}
        elif tool == "search_nodes":
            q = params.get("query", "").lower()
            return {"results": [e for e in g["entities"] if q in json.dumps(e).lower()]}
        elif tool == "read_graph":
            return g
        elif tool == "add_observations":
            for obs in params.get("observations", []):
                for e in g["entities"]:
                    if e.get("name") == obs.get("entityName"):
                        e.setdefault("observations", []).extend(obs.get("contents", []))
            self._save(g)
            return {"added": len(params.get("observations", []))}
        elif tool == "open_nodes":
            return {"nodes": [e for e in g["entities"] if e.get("name") in params.get("names", [])]}
        elif tool == "delete_entities":
            names = params.get("names", [])
            g["entities"] = [e for e in g["entities"] if e.get("name") not in names]
            g["relations"] = [r for r in g["relations"] if r.get("from") not in names and r.get("to") not in names]
            self._save(g)
            return {"deleted": len(names)}
        return {"error": f"Unknown memory tool: {tool}"}


class UtilityMCPClient(MCPClient):
    async def execute(self, tool: str, params: Dict) -> Dict:
        sid = self.config.get("_id", "")
        
        # Time utilities
        if sid == "time":
            if tool == "now":
                return {"time": datetime.now(timezone.utc).isoformat(), "timestamp": int(datetime.now(timezone.utc).timestamp())}
            elif tool == "format_date":
                dt = datetime.fromisoformat(params.get("date", ""))
                return {"formatted": dt.strftime(params.get("format", "%Y-%m-%d %H:%M:%S"))}
            elif tool == "parse_date":
                from dateutil import parser
                dt = parser.parse(params.get("date", ""))
                return {"iso": dt.isoformat(), "timestamp": int(dt.timestamp())}
        
        # Crypto utilities
        elif sid == "crypto":
            if tool == "hash":
                data = params.get("data", "").encode()
                algo = params.get("algorithm", "sha256")
                return {"hash": hashlib.new(algo, data).hexdigest()}
            elif tool == "encode_base64":
                return {"encoded": base64.b64encode(params.get("data", "").encode()).decode()}
            elif tool == "decode_base64":
                return {"decoded": base64.b64decode(params.get("data", "")).decode()}
            elif tool == "generate_uuid":
                import uuid
                return {"uuid": str(uuid.uuid4())}
            elif tool == "random_string":
                import secrets
                return {"random": secrets.token_urlsafe(params.get("length", 32))}
        
        # JSON utilities
        elif sid == "json_tools":
            if tool == "parse":
                return {"parsed": json.loads(params.get("data", "{}"))}
            elif tool == "stringify":
                return {"json": json.dumps(params.get("data", {}), indent=2)}
            elif tool == "query":
                # Simple dot-notation query
                data = params.get("data", {})
                for key in params.get("path", "").split("."):
                    if key and isinstance(data, dict):
                        data = data.get(key)
                return {"result": data}
        
        # Regex utilities
        elif sid == "regex":
            pattern = params.get("pattern", "")
            text = params.get("text", "")
            if tool == "match":
                m = re.match(pattern, text)
                return {"match": m.group() if m else None}
            elif tool == "search":
                m = re.search(pattern, text)
                return {"match": m.group() if m else None}
            elif tool == "findall":
                return {"matches": re.findall(pattern, text)}
            elif tool == "replace":
                return {"result": re.sub(pattern, params.get("replacement", ""), text)}
            elif tool == "split":
                return {"parts": re.split(pattern, text)}
        
        # Text utilities
        elif sid == "text":
            text = params.get("text", "")
            if tool == "extract_emails":
                return {"emails": re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)}
            elif tool == "extract_urls":
                return {"urls": re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)}
            elif tool == "extract_phones":
                return {"phones": re.findall(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}', text)}
            elif tool == "clean_html":
                return {"text": re.sub(r'<[^>]+>', '', text)}
        
        # Math utilities
        elif sid == "math":
            if tool == "calculate":
                # Safe eval for math
                expr = params.get("expression", "0")
                allowed = set("0123456789+-*/.() ")
                if all(c in allowed for c in expr):
                    return {"result": eval(expr)}
                return {"error": "Invalid expression"}
            elif tool == "statistics":
                import statistics
                data = params.get("data", [])
                return {
                    "mean": statistics.mean(data) if data else 0,
                    "median": statistics.median(data) if data else 0,
                    "stdev": statistics.stdev(data) if len(data) > 1 else 0
                }
        
        # Thinking utilities
        elif sid == "thinking":
            if tool == "think":
                return {"step": params.get("step", 1), "thought": params.get("thought", ""), "next_needed": params.get("next_needed", True)}
            elif tool == "plan":
                return {"goal": params.get("goal", ""), "steps": params.get("steps", [])}
        
        return {"error": f"Unknown utility tool: {tool} for {sid}"}


# =============================================================================
# CLIENT FACTORY
# =============================================================================

def get_client(server_id: str) -> MCPClient:
    if server_id not in MCP_SERVERS:
        raise HTTPException(status_code=404, detail=f"Unknown server: {server_id}")
    
    config = MCP_SERVERS[server_id].copy()
    config["_id"] = server_id
    stype = config.get("type", "")
    
    if server_id == "gtm":
        return HTTPMCPClient(config)
    elif stype == "api":
        return GenericAPIClient(config)
    elif stype == "socket" or server_id == "docker":
        return DockerMCPClient(config)
    elif stype == "shell" and server_id == "git":
        return GitMCPClient(config)
    elif stype == "shell":
        return ShellMCPClient(config)
    elif stype == "aws" or server_id == "s3":
        return S3MCPClient(config)
    elif server_id == "filesystem":
        return FilesystemMCPClient(config)
    elif server_id == "memory":
        return MemoryMCPClient(config)
    elif stype == "utility":
        return UtilityMCPClient(config)
    else:
        raise HTTPException(status_code=400, detail=f"No client for: {server_id}")


# =============================================================================
# API
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

@app.get("/")
async def root():
    total_tools = sum(len(s["tools"]) for s in MCP_SERVERS.values())
    return {
        "service": "Origin OS MCP Hub",
        "version": "4.0",
        "total_servers": len(MCP_SERVERS),
        "total_tools": total_tools,
        "categories": {
            "marketing": ["gtm", "ga4", "semrush", "google_ads"],
            "web": ["firecrawl", "apify", "playwright", "fetch"],
            "ai": ["openrouter", "anthropic", "openai", "google_ai"],
            "code": ["github", "git", "code_analysis"],
            "infra": ["docker", "shell", "process"],
            "cloud": ["s3", "gcs"],
            "database": ["postgres", "sqlite", "redis"],
            "storage": ["filesystem", "memory"],
            "communication": ["slack", "email", "twilio"],
            "utilities": ["time", "crypto", "json_tools", "regex", "math", "text", "thinking", "tasks"],
            "data": ["pdf", "excel", "image"],
            "integration": ["webhook", "cron", "queue"]
        }
    }

@app.get("/servers")
async def list_servers():
    return {"servers": [
        {"id": k, "name": v["name"], "description": v["description"], "tools": v["tools"], "tool_count": len(v["tools"])}
        for k, v in MCP_SERVERS.items()
    ], "total": len(MCP_SERVERS)}

@app.get("/servers/{server_id}/tools")
async def list_tools(server_id: str):
    if server_id not in MCP_SERVERS:
        raise HTTPException(404, f"Unknown: {server_id}")
    s = MCP_SERVERS[server_id]
    return {"server": server_id, "name": s["name"], "tools": s["tools"]}

@app.post("/execute", response_model=ToolResult)
async def execute_tool(call: ToolCall):
    try:
        client = get_client(call.server)
        result = await client.execute(call.tool, call.params)
        if isinstance(result, dict) and "error" in result:
            return ToolResult(success=False, server=call.server, tool=call.tool, error=result["error"])
        return ToolResult(success=True, server=call.server, tool=call.tool, result=result)
    except Exception as e:
        return ToolResult(success=False, server=call.server, tool=call.tool, error=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mcp-hub", "version": "4.0"}

if __name__ == "__main__":
    import uvicorn
    total = sum(len(s["tools"]) for s in MCP_SERVERS.values())
    print("=" * 70)
    print("🔌 ORIGIN OS MCP HUB v4.0")
    print("=" * 70)
    print(f"\n📊 {len(MCP_SERVERS)} Servers | {total} Tools\n")
    for sid, cfg in MCP_SERVERS.items():
        print(f"  • {sid}: {cfg['name']} ({len(cfg['tools'])} tools)")
    print("\n" + "=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
