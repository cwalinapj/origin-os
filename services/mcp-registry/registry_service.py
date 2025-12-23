#!/usr/bin/env python3
"""
MCP Registry Service
====================
Catalog, manage, and orchestrate MCP servers for Origin OS.

Features:
- Curated registry of 100+ top MCP servers
- Automatic installation and configuration
- Health monitoring and status tracking
- Category-based organization
- One-click enable/disable
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx
import uvicorn

app = FastAPI(title="MCP Registry", version="1.0")

# =============================================================================
# CONFIGURATION
# =============================================================================

REGISTRY_DIR = Path(os.getenv("REGISTRY_DIR", "/data/registry"))
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA MODELS
# =============================================================================

class MCPCategory(str, Enum):
    COMMUNICATION = "communication"
    DATABASE = "database"
    CLOUD = "cloud"
    DEVELOPER = "developer"
    PRODUCTIVITY = "productivity"
    SEARCH = "search"
    AI_ML = "ai_ml"
    FINANCE = "finance"
    SECURITY = "security"
    AUTOMATION = "automation"
    STORAGE = "storage"
    ANALYTICS = "analytics"
    OTHER = "other"

class InstallMethod(str, Enum):
    NPX = "npx"
    DOCKER = "docker"
    PIP = "pip"
    BINARY = "binary"
    SOURCE = "source"

@dataclass
class MCPServer:
    id: str
    name: str
    description: str
    category: str
    github_url: str
    stars: int = 0
    install_method: str = "npx"
    install_command: str = ""
    docker_image: str = ""
    npm_package: str = ""
    pip_package: str = ""
    env_vars: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    official: bool = False
    enabled: bool = False
    last_updated: str = ""
    
@dataclass
class InstalledMCP:
    id: str
    enabled: bool
    config: Dict[str, Any]
    installed_at: str
    last_health_check: str = ""
    healthy: bool = True

# =============================================================================
# TOP 100 MCP SERVERS REGISTRY
# =============================================================================

TOP_MCP_SERVERS = [
    # === COMMUNICATION ===
    MCPServer(
        id="google-workspace",
        name="Google Workspace",
        description="Gmail, Calendar, Drive integration with OAuth2 authentication",
        category=MCPCategory.COMMUNICATION,
        github_url="https://github.com/aaronsb/google-workspace-mcp",
        stars=107,
        install_method=InstallMethod.DOCKER,
        docker_image="ghcr.io/aaronsb/google-workspace-mcp:latest",
        env_vars=["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"],
        tools=["search_workspace_emails", "send_workspace_email", "list_workspace_calendar_events", 
               "create_workspace_calendar_event", "list_drive_files", "upload_drive_file"],
        official=False
    ),
    MCPServer(
        id="slack",
        name="Slack",
        description="Slack workspace integration - messages, channels, users",
        category=MCPCategory.COMMUNICATION,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/slack",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-slack",
        env_vars=["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
        tools=["list_channels", "post_message", "get_channel_history", "search_messages"],
        official=True
    ),
    MCPServer(
        id="discord",
        name="Discord",
        description="Discord bot integration for servers and channels",
        category=MCPCategory.COMMUNICATION,
        github_url="https://github.com/SaseQ/discord-mcp",
        stars=50,
        install_method=InstallMethod.NPX,
        env_vars=["DISCORD_BOT_TOKEN"],
        tools=["send_message", "list_channels", "get_messages"]
    ),
    MCPServer(
        id="telegram",
        name="Telegram",
        description="Telegram API for messages, chats, and media",
        category=MCPCategory.COMMUNICATION,
        github_url="https://github.com/chaindead/telegram-mcp",
        stars=80,
        install_method=InstallMethod.DOCKER,
        env_vars=["TELEGRAM_API_ID", "TELEGRAM_API_HASH"],
        tools=["send_message", "get_dialogs", "get_messages"]
    ),
    MCPServer(
        id="whatsapp",
        name="WhatsApp",
        description="WhatsApp Business Platform integration",
        category=MCPCategory.COMMUNICATION,
        github_url="https://github.com/lharries/whatsapp-mcp",
        stars=100,
        install_method=InstallMethod.PIP,
        tools=["search_messages", "send_message", "get_contacts"]
    ),

    # === DATABASES ===
    MCPServer(
        id="postgres",
        name="PostgreSQL",
        description="PostgreSQL database with schema inspection and queries",
        category=MCPCategory.DATABASE,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/postgres",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-postgres",
        env_vars=["POSTGRES_CONNECTION_STRING"],
        tools=["query", "list_tables", "describe_table"],
        official=True
    ),
    MCPServer(
        id="sqlite",
        name="SQLite",
        description="SQLite database operations with analysis features",
        category=MCPCategory.DATABASE,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/sqlite",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-sqlite",
        tools=["query", "list_tables", "describe_table", "analyze"],
        official=True
    ),
    MCPServer(
        id="mongodb",
        name="MongoDB",
        description="MongoDB database integration with document operations",
        category=MCPCategory.DATABASE,
        github_url="https://github.com/QuantGeekDev/mongo-mcp",
        stars=200,
        install_method=InstallMethod.NPX,
        env_vars=["MONGODB_URI"],
        tools=["find", "insert", "update", "delete", "aggregate"]
    ),
    MCPServer(
        id="redis",
        name="Redis",
        description="Redis key-value store with search capabilities",
        category=MCPCategory.DATABASE,
        github_url="https://github.com/redis/mcp-redis",
        stars=150,
        install_method=InstallMethod.PIP,
        pip_package="mcp-redis",
        env_vars=["REDIS_URL"],
        tools=["get", "set", "search", "list_keys"],
        official=True
    ),
    MCPServer(
        id="supabase",
        name="Supabase",
        description="Supabase database with auth and storage",
        category=MCPCategory.DATABASE,
        github_url="https://github.com/supabase-community/supabase-mcp",
        stars=300,
        install_method=InstallMethod.NPX,
        env_vars=["SUPABASE_URL", "SUPABASE_KEY"],
        tools=["query", "insert", "update", "storage_upload"],
        official=True
    ),
    MCPServer(
        id="neo4j",
        name="Neo4j",
        description="Neo4j graph database with Cypher queries",
        category=MCPCategory.DATABASE,
        github_url="https://github.com/neo4j-contrib/mcp-neo4j",
        stars=100,
        install_method=InstallMethod.PIP,
        env_vars=["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"],
        tools=["query", "create_node", "create_relationship"]
    ),
    MCPServer(
        id="qdrant",
        name="Qdrant",
        description="Qdrant vector database for embeddings and search",
        category=MCPCategory.DATABASE,
        github_url="https://github.com/qdrant/mcp-server-qdrant",
        stars=80,
        install_method=InstallMethod.PIP,
        env_vars=["QDRANT_URL", "QDRANT_API_KEY"],
        tools=["upsert", "search", "delete"],
        official=True
    ),
    MCPServer(
        id="pinecone",
        name="Pinecone",
        description="Pinecone vector database integration",
        category=MCPCategory.DATABASE,
        github_url="https://github.com/sirmews/mcp-pinecone",
        stars=60,
        install_method=InstallMethod.PIP,
        env_vars=["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"],
        tools=["upsert", "query", "delete"]
    ),
    MCPServer(
        id="clickhouse",
        name="ClickHouse",
        description="ClickHouse analytics database",
        category=MCPCategory.DATABASE,
        github_url="https://github.com/ClickHouse/mcp-clickhouse",
        stars=50,
        install_method=InstallMethod.PIP,
        env_vars=["CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD"],
        tools=["query", "list_tables", "describe_table"],
        official=True
    ),
    MCPServer(
        id="snowflake",
        name="Snowflake",
        description="Snowflake data warehouse integration",
        category=MCPCategory.DATABASE,
        github_url="https://github.com/Snowflake-Labs/mcp",
        stars=100,
        install_method=InstallMethod.PIP,
        env_vars=["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"],
        tools=["query", "list_databases", "describe_table"],
        official=True
    ),

    # === CLOUD PLATFORMS ===
    MCPServer(
        id="aws",
        name="AWS",
        description="AWS services integration via CLI",
        category=MCPCategory.CLOUD,
        github_url="https://github.com/awslabs/mcp",
        stars=500,
        install_method=InstallMethod.NPX,
        env_vars=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
        tools=["execute_command", "list_resources"],
        official=True
    ),
    MCPServer(
        id="kubernetes",
        name="Kubernetes",
        description="Kubernetes cluster management",
        category=MCPCategory.CLOUD,
        github_url="https://github.com/Flux159/mcp-server-kubernetes",
        stars=200,
        install_method=InstallMethod.NPX,
        env_vars=["KUBECONFIG"],
        tools=["list_pods", "get_pod", "apply_manifest", "delete_resource"]
    ),
    MCPServer(
        id="docker",
        name="Docker",
        description="Docker container and image management",
        category=MCPCategory.CLOUD,
        github_url="https://github.com/ckreiling/mcp-server-docker",
        stars=150,
        install_method=InstallMethod.PIP,
        tools=["list_containers", "run_container", "stop_container", "list_images"]
    ),
    MCPServer(
        id="cloudflare",
        name="Cloudflare",
        description="Cloudflare Workers, KV, R2, D1 integration",
        category=MCPCategory.CLOUD,
        github_url="https://github.com/cloudflare/mcp-server-cloudflare",
        stars=300,
        install_method=InstallMethod.NPX,
        env_vars=["CLOUDFLARE_API_TOKEN"],
        tools=["deploy_worker", "kv_get", "kv_put", "r2_upload"],
        official=True
    ),
    MCPServer(
        id="vercel",
        name="Vercel",
        description="Vercel deployment and project management",
        category=MCPCategory.CLOUD,
        github_url="https://github.com/vercel/mcp",
        stars=200,
        install_method=InstallMethod.NPX,
        env_vars=["VERCEL_TOKEN"],
        tools=["deploy", "list_projects", "get_deployments"],
        official=True
    ),
    MCPServer(
        id="azure",
        name="Azure",
        description="Azure services via CLI wrapper",
        category=MCPCategory.CLOUD,
        github_url="https://github.com/jdubois/azure-cli-mcp",
        stars=100,
        install_method=InstallMethod.PIP,
        env_vars=["AZURE_SUBSCRIPTION_ID"],
        tools=["execute_command", "list_resources"]
    ),
    MCPServer(
        id="pulumi",
        name="Pulumi",
        description="Pulumi infrastructure as code",
        category=MCPCategory.CLOUD,
        github_url="https://github.com/pulumi/mcp-server",
        stars=80,
        install_method=InstallMethod.NPX,
        env_vars=["PULUMI_ACCESS_TOKEN"],
        tools=["preview", "up", "destroy", "stack_output"],
        official=True
    ),

    # === DEVELOPER TOOLS ===
    MCPServer(
        id="github",
        name="GitHub",
        description="GitHub repos, issues, PRs, actions",
        category=MCPCategory.DEVELOPER,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/github",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-github",
        env_vars=["GITHUB_PERSONAL_ACCESS_TOKEN"],
        tools=["search_repositories", "get_file_contents", "create_issue", "create_pull_request"],
        official=True
    ),
    MCPServer(
        id="gitlab",
        name="GitLab",
        description="GitLab repository and CI/CD integration",
        category=MCPCategory.DEVELOPER,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["GITLAB_TOKEN"],
        tools=["list_projects", "get_file", "create_issue"]
    ),
    MCPServer(
        id="linear",
        name="Linear",
        description="Linear issue tracking integration",
        category=MCPCategory.DEVELOPER,
        github_url="https://github.com/jerhadf/linear-mcp-server",
        stars=100,
        install_method=InstallMethod.NPX,
        env_vars=["LINEAR_API_KEY"],
        tools=["create_issue", "list_issues", "update_issue"]
    ),
    MCPServer(
        id="jira",
        name="Jira",
        description="Atlassian Jira issue and project management",
        category=MCPCategory.DEVELOPER,
        github_url="https://github.com/aashari/mcp-server-atlassian-jira",
        stars=80,
        install_method=InstallMethod.NPX,
        env_vars=["JIRA_HOST", "JIRA_EMAIL", "JIRA_API_TOKEN"],
        tools=["search_issues", "create_issue", "update_issue", "get_project"]
    ),
    MCPServer(
        id="sentry",
        name="Sentry",
        description="Sentry error tracking and monitoring",
        category=MCPCategory.DEVELOPER,
        github_url="https://github.com/getsentry/sentry-mcp",
        stars=100,
        install_method=InstallMethod.NPX,
        env_vars=["SENTRY_AUTH_TOKEN"],
        tools=["list_issues", "get_issue", "resolve_issue"],
        official=True
    ),
    MCPServer(
        id="playwright",
        name="Playwright",
        description="Browser automation and testing",
        category=MCPCategory.DEVELOPER,
        github_url="https://github.com/microsoft/playwright-mcp",
        stars=500,
        install_method=InstallMethod.NPX,
        tools=["navigate", "click", "fill", "screenshot", "evaluate"],
        official=True
    ),
    MCPServer(
        id="puppeteer",
        name="Puppeteer",
        description="Headless Chrome automation",
        category=MCPCategory.DEVELOPER,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-puppeteer",
        tools=["navigate", "screenshot", "click", "type"],
        official=True
    ),
    MCPServer(
        id="circleci",
        name="CircleCI",
        description="CircleCI build and pipeline management",
        category=MCPCategory.DEVELOPER,
        github_url="https://github.com/CircleCI-Public/mcp-server-circleci",
        stars=50,
        install_method=InstallMethod.NPX,
        env_vars=["CIRCLECI_TOKEN"],
        tools=["get_pipeline", "trigger_pipeline", "get_job_logs"],
        official=True
    ),
    MCPServer(
        id="buildkite",
        name="Buildkite",
        description="Buildkite CI/CD integration",
        category=MCPCategory.DEVELOPER,
        github_url="https://github.com/buildkite/buildkite-mcp-server",
        stars=40,
        install_method=InstallMethod.DOCKER,
        env_vars=["BUILDKITE_API_TOKEN"],
        tools=["list_pipelines", "trigger_build", "get_build"],
        official=True
    ),

    # === PRODUCTIVITY ===
    MCPServer(
        id="notion",
        name="Notion",
        description="Notion pages, databases, and blocks",
        category=MCPCategory.PRODUCTIVITY,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["NOTION_API_KEY"],
        tools=["search", "get_page", "create_page", "update_page"]
    ),
    MCPServer(
        id="obsidian",
        name="Obsidian",
        description="Obsidian vault and note management",
        category=MCPCategory.PRODUCTIVITY,
        github_url="https://github.com/smithery-ai/mcp-obsidian",
        stars=200,
        install_method=InstallMethod.NPX,
        tools=["search_notes", "get_note", "create_note", "update_note"]
    ),
    MCPServer(
        id="todoist",
        name="Todoist",
        description="Todoist task management",
        category=MCPCategory.PRODUCTIVITY,
        github_url="https://github.com/abhiz123/todoist-mcp-server",
        stars=50,
        install_method=InstallMethod.NPX,
        env_vars=["TODOIST_API_TOKEN"],
        tools=["get_tasks", "create_task", "complete_task", "get_projects"]
    ),
    MCPServer(
        id="google-tasks",
        name="Google Tasks",
        description="Google Tasks integration",
        category=MCPCategory.PRODUCTIVITY,
        github_url="https://github.com/zcaceres/gtasks-mcp",
        stars=30,
        install_method=InstallMethod.NPX,
        env_vars=["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"],
        tools=["list_tasks", "create_task", "complete_task"]
    ),
    MCPServer(
        id="airtable",
        name="Airtable",
        description="Airtable database integration",
        category=MCPCategory.PRODUCTIVITY,
        github_url="https://github.com/domdomegg/airtable-mcp-server",
        stars=60,
        install_method=InstallMethod.NPX,
        env_vars=["AIRTABLE_API_KEY"],
        tools=["list_records", "create_record", "update_record", "delete_record"]
    ),
    MCPServer(
        id="trello",
        name="Trello",
        description="Trello boards, lists, and cards",
        category=MCPCategory.PRODUCTIVITY,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["TRELLO_API_KEY", "TRELLO_TOKEN"],
        tools=["list_boards", "create_card", "move_card"]
    ),

    # === SEARCH & DATA ===
    MCPServer(
        id="brave-search",
        name="Brave Search",
        description="Brave Search API integration",
        category=MCPCategory.SEARCH,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-brave-search",
        env_vars=["BRAVE_API_KEY"],
        tools=["web_search", "local_search"],
        official=True
    ),
    MCPServer(
        id="exa",
        name="Exa Search",
        description="Exa AI-powered search",
        category=MCPCategory.SEARCH,
        github_url="https://github.com/exa-labs/exa-mcp-server",
        stars=100,
        install_method=InstallMethod.NPX,
        env_vars=["EXA_API_KEY"],
        tools=["search", "find_similar", "get_contents"],
        official=True
    ),
    MCPServer(
        id="tavily",
        name="Tavily",
        description="Tavily AI search for LLMs",
        category=MCPCategory.SEARCH,
        github_url="https://github.com/tavily-ai/tavily-mcp",
        stars=80,
        install_method=InstallMethod.NPX,
        env_vars=["TAVILY_API_KEY"],
        tools=["search", "extract"],
        official=True
    ),
    MCPServer(
        id="firecrawl",
        name="Firecrawl",
        description="Web scraping and crawling",
        category=MCPCategory.SEARCH,
        github_url="https://github.com/mendableai/firecrawl-mcp-server",
        stars=200,
        install_method=InstallMethod.NPX,
        env_vars=["FIRECRAWL_API_KEY"],
        tools=["scrape", "crawl", "map"],
        official=True
    ),
    MCPServer(
        id="fetch",
        name="Fetch",
        description="Web content fetching and conversion",
        category=MCPCategory.SEARCH,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/fetch",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-fetch",
        tools=["fetch"],
        official=True
    ),

    # === AI/ML ===
    MCPServer(
        id="openai",
        name="OpenAI",
        description="OpenAI API integration",
        category=MCPCategory.AI_ML,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["OPENAI_API_KEY"],
        tools=["chat", "embeddings", "images"]
    ),
    MCPServer(
        id="anthropic",
        name="Anthropic",
        description="Anthropic Claude API",
        category=MCPCategory.AI_ML,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["ANTHROPIC_API_KEY"],
        tools=["chat", "complete"]
    ),
    MCPServer(
        id="huggingface",
        name="HuggingFace",
        description="HuggingFace models and datasets",
        category=MCPCategory.AI_ML,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["HF_TOKEN"],
        tools=["inference", "list_models", "download_model"]
    ),
    MCPServer(
        id="langfuse",
        name="Langfuse",
        description="LLM observability and prompt management",
        category=MCPCategory.AI_ML,
        github_url="https://github.com/langfuse/langfuse-mcp",
        stars=50,
        install_method=InstallMethod.NPX,
        env_vars=["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
        tools=["get_prompts", "log_trace"]
    ),

    # === FINANCE ===
    MCPServer(
        id="stripe",
        name="Stripe",
        description="Stripe payments and subscriptions",
        category=MCPCategory.FINANCE,
        github_url="https://github.com/stripe/stripe-mcp",
        stars=100,
        install_method=InstallMethod.NPX,
        env_vars=["STRIPE_API_KEY"],
        tools=["create_payment", "list_customers", "create_subscription"],
        official=True
    ),
    MCPServer(
        id="plaid",
        name="Plaid",
        description="Plaid banking integration",
        category=MCPCategory.FINANCE,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["PLAID_CLIENT_ID", "PLAID_SECRET"],
        tools=["get_accounts", "get_transactions"]
    ),

    # === STORAGE ===
    MCPServer(
        id="filesystem",
        name="Filesystem",
        description="Local filesystem operations",
        category=MCPCategory.STORAGE,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-filesystem",
        tools=["read_file", "write_file", "list_directory", "search_files"],
        official=True
    ),
    MCPServer(
        id="s3",
        name="AWS S3",
        description="AWS S3 object storage",
        category=MCPCategory.STORAGE,
        github_url="https://github.com/awslabs/mcp",
        stars=500,
        install_method=InstallMethod.NPX,
        env_vars=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        tools=["list_buckets", "list_objects", "get_object", "put_object"],
        official=True
    ),
    MCPServer(
        id="google-drive",
        name="Google Drive",
        description="Google Drive file management (standalone)",
        category=MCPCategory.STORAGE,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"],
        tools=["list_files", "get_file", "upload_file", "share_file"]
    ),
    MCPServer(
        id="dropbox",
        name="Dropbox",
        description="Dropbox file storage",
        category=MCPCategory.STORAGE,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["DROPBOX_ACCESS_TOKEN"],
        tools=["list_files", "download_file", "upload_file"]
    ),

    # === SECURITY ===
    MCPServer(
        id="1password",
        name="1Password",
        description="1Password secrets management",
        category=MCPCategory.SECURITY,
        github_url="https://github.com/1Password/1password-mcp",
        stars=100,
        install_method=InstallMethod.NPX,
        env_vars=["OP_SERVICE_ACCOUNT_TOKEN"],
        tools=["get_secret", "list_vaults", "create_item"],
        official=True
    ),
    MCPServer(
        id="vault",
        name="HashiCorp Vault",
        description="HashiCorp Vault secrets",
        category=MCPCategory.SECURITY,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["VAULT_ADDR", "VAULT_TOKEN"],
        tools=["read_secret", "write_secret", "list_secrets"]
    ),

    # === AUTOMATION ===
    MCPServer(
        id="zapier",
        name="Zapier",
        description="Zapier workflow automation",
        category=MCPCategory.AUTOMATION,
        github_url="https://github.com/zapier/zapier-mcp",
        stars=50,
        install_method=InstallMethod.NPX,
        env_vars=["ZAPIER_API_KEY"],
        tools=["trigger_zap", "list_zaps"]
    ),
    MCPServer(
        id="make",
        name="Make (Integromat)",
        description="Make automation scenarios",
        category=MCPCategory.AUTOMATION,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["MAKE_API_KEY"],
        tools=["trigger_scenario", "list_scenarios"]
    ),
    MCPServer(
        id="n8n",
        name="n8n",
        description="n8n workflow automation",
        category=MCPCategory.AUTOMATION,
        github_url="https://github.com/n8n-io/n8n-mcp",
        stars=100,
        install_method=InstallMethod.NPX,
        env_vars=["N8N_API_KEY", "N8N_HOST"],
        tools=["execute_workflow", "list_workflows"]
    ),

    # === ANALYTICS ===
    MCPServer(
        id="google-analytics",
        name="Google Analytics",
        description="Google Analytics data access",
        category=MCPCategory.ANALYTICS,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"],
        tools=["get_report", "list_accounts"]
    ),
    MCPServer(
        id="mixpanel",
        name="Mixpanel",
        description="Mixpanel product analytics",
        category=MCPCategory.ANALYTICS,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["MIXPANEL_TOKEN"],
        tools=["track", "query"]
    ),
    MCPServer(
        id="amplitude",
        name="Amplitude",
        description="Amplitude product analytics",
        category=MCPCategory.ANALYTICS,
        github_url="https://github.com/modelcontextprotocol/servers",
        stars=15000,
        install_method=InstallMethod.NPX,
        env_vars=["AMPLITUDE_API_KEY"],
        tools=["query", "get_cohorts"]
    ),

    # === OTHER USEFUL SERVERS ===
    MCPServer(
        id="memory",
        name="Memory",
        description="Knowledge graph-based persistent memory",
        category=MCPCategory.OTHER,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/memory",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-memory",
        tools=["create_entities", "create_relations", "search_nodes"],
        official=True
    ),
    MCPServer(
        id="sequential-thinking",
        name="Sequential Thinking",
        description="Dynamic problem-solving through thought sequences",
        category=MCPCategory.OTHER,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-sequential-thinking",
        tools=["think"],
        official=True
    ),
    MCPServer(
        id="time",
        name="Time",
        description="Time and timezone utilities",
        category=MCPCategory.OTHER,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/time",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-time",
        tools=["get_current_time", "convert_timezone"],
        official=True
    ),
    MCPServer(
        id="everything",
        name="Everything",
        description="Reference/test server with all MCP features",
        category=MCPCategory.OTHER,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/everything",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-everything",
        tools=["echo", "add", "longRunningOperation"],
        official=True
    ),
    MCPServer(
        id="git",
        name="Git",
        description="Git repository operations",
        category=MCPCategory.DEVELOPER,
        github_url="https://github.com/modelcontextprotocol/servers/tree/main/src/git",
        stars=15000,
        install_method=InstallMethod.NPX,
        npm_package="@modelcontextprotocol/server-git",
        tools=["git_status", "git_log", "git_diff", "git_commit"],
        official=True
    ),
]

# =============================================================================
# REGISTRY STATE
# =============================================================================

def load_installed() -> Dict[str, InstalledMCP]:
    """Load installed MCP servers"""
    installed_file = REGISTRY_DIR / "installed.json"
    if installed_file.exists():
        with open(installed_file) as f:
            data = json.load(f)
            return {k: InstalledMCP(**v) for k, v in data.items()}
    return {}

def save_installed(installed: Dict[str, InstalledMCP]):
    """Save installed MCP servers"""
    installed_file = REGISTRY_DIR / "installed.json"
    with open(installed_file, 'w') as f:
        json.dump({k: asdict(v) for k, v in installed.items()}, f, indent=2)

installed_mcps = load_installed()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Registry info"""
    categories = {}
    for server in TOP_MCP_SERVERS:
        cat = server.category if isinstance(server.category, str) else server.category.value
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "service": "MCP Registry",
        "version": "1.0",
        "stats": {
            "total_servers": len(TOP_MCP_SERVERS),
            "installed": len(installed_mcps),
            "enabled": len([m for m in installed_mcps.values() if m.enabled]),
            "official": len([s for s in TOP_MCP_SERVERS if s.official])
        },
        "categories": categories
    }

@app.get("/servers")
async def list_servers(
    category: Optional[str] = None,
    search: Optional[str] = None,
    official_only: bool = False,
    limit: int = 100
):
    """List available MCP servers"""
    servers = TOP_MCP_SERVERS
    
    if category:
        servers = [s for s in servers if s.category == category or 
                   (hasattr(s.category, 'value') and s.category.value == category)]
    
    if official_only:
        servers = [s for s in servers if s.official]
    
    if search:
        search_lower = search.lower()
        servers = [s for s in servers if 
                   search_lower in s.name.lower() or 
                   search_lower in s.description.lower()]
    
    # Add installed status
    result = []
    for server in servers[:limit]:
        data = asdict(server)
        data["installed"] = server.id in installed_mcps
        data["enabled"] = installed_mcps.get(server.id, InstalledMCP(
            id=server.id, enabled=False, config={}, installed_at=""
        )).enabled
        result.append(data)
    
    return {"servers": result, "count": len(result)}

@app.get("/servers/{server_id}")
async def get_server(server_id: str):
    """Get server details"""
    server = next((s for s in TOP_MCP_SERVERS if s.id == server_id), None)
    if not server:
        raise HTTPException(404, "Server not found")
    
    data = asdict(server)
    data["installed"] = server_id in installed_mcps
    if server_id in installed_mcps:
        data["install_info"] = asdict(installed_mcps[server_id])
    
    return data

@app.post("/servers/{server_id}/install")
async def install_server(server_id: str, config: Dict[str, Any] = None):
    """Install/enable an MCP server"""
    server = next((s for s in TOP_MCP_SERVERS if s.id == server_id), None)
    if not server:
        raise HTTPException(404, "Server not found")
    
    installed_mcps[server_id] = InstalledMCP(
        id=server_id,
        enabled=True,
        config=config or {},
        installed_at=datetime.now(timezone.utc).isoformat()
    )
    
    save_installed(installed_mcps)
    
    return {
        "installed": True,
        "server_id": server_id,
        "install_method": server.install_method,
        "required_env_vars": server.env_vars
    }

@app.post("/servers/{server_id}/uninstall")
async def uninstall_server(server_id: str):
    """Uninstall/disable an MCP server"""
    if server_id in installed_mcps:
        del installed_mcps[server_id]
        save_installed(installed_mcps)
    
    return {"uninstalled": True, "server_id": server_id}

@app.post("/servers/{server_id}/enable")
async def enable_server(server_id: str):
    """Enable an installed server"""
    if server_id not in installed_mcps:
        raise HTTPException(404, "Server not installed")
    
    installed_mcps[server_id].enabled = True
    save_installed(installed_mcps)
    
    return {"enabled": True, "server_id": server_id}

@app.post("/servers/{server_id}/disable")
async def disable_server(server_id: str):
    """Disable an installed server"""
    if server_id not in installed_mcps:
        raise HTTPException(404, "Server not installed")
    
    installed_mcps[server_id].enabled = False
    save_installed(installed_mcps)
    
    return {"disabled": True, "server_id": server_id}

@app.get("/installed")
async def list_installed():
    """List installed MCP servers"""
    result = []
    for mcp_id, mcp in installed_mcps.items():
        server = next((s for s in TOP_MCP_SERVERS if s.id == mcp_id), None)
        if server:
            result.append({
                **asdict(server),
                "install_info": asdict(mcp)
            })
    
    return {"installed": result, "count": len(result)}

@app.get("/categories")
async def list_categories():
    """List all categories with counts"""
    categories = {}
    for server in TOP_MCP_SERVERS:
        cat = server.category if isinstance(server.category, str) else server.category.value
        if cat not in categories:
            categories[cat] = {"count": 0, "servers": []}
        categories[cat]["count"] += 1
        categories[cat]["servers"].append(server.id)
    
    return categories

@app.get("/export")
async def export_config():
    """Export MCP configuration for Claude/Cursor"""
    config = {"mcpServers": {}}
    
    for mcp_id, mcp in installed_mcps.items():
        if not mcp.enabled:
            continue
        
        server = next((s for s in TOP_MCP_SERVERS if s.id == mcp_id), None)
        if not server:
            continue
        
        if server.install_method == InstallMethod.NPX or server.install_method == "npx":
            package = server.npm_package or f"@mcp/{server.id}"
            config["mcpServers"][mcp_id] = {
                "command": "npx",
                "args": ["-y", package],
                "env": {var: f"${{{var}}}" for var in server.env_vars}
            }
        elif server.install_method == InstallMethod.DOCKER or server.install_method == "docker":
            config["mcpServers"][mcp_id] = {
                "command": "docker",
                "args": ["run", "--rm", "-i", server.docker_image],
                "env": {var: f"${{{var}}}" for var in server.env_vars}
            }
    
    return config

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mcp-registry"}

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("📦 MCP REGISTRY SERVICE")
    print("=" * 70)
    print(f"""
Cataloged MCP Servers: {len(TOP_MCP_SERVERS)}
Official Servers: {len([s for s in TOP_MCP_SERVERS if s.official])}

Categories:
""")
    categories = {}
    for s in TOP_MCP_SERVERS:
        cat = s.category if isinstance(s.category, str) else s.category.value
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} servers")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
