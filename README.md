# ⚡ Origin OS

**Origin OS** is a modular AI-powered automation platform for managing Google Tag Manager, web scraping, and container orchestration. It provides a unified interface to interact with multiple LLM providers through OpenRouter and exposes MCP (Model Context Protocol) servers for seamless AI tool integration.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Origin OS UI                           │
│                   (http://localhost:8000)                   │
│           Multi-LLM Chat Interface via OpenRouter           │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│        Codex            │     │       MCP Hub           │
│  (http://localhost:8001)│     │  (http://localhost:8002)│
│   GTM Manager API       │     │  Unified MCP Gateway    │
│   - Tag management      │     │  - Firecrawl scraping   │
│   - Trigger creation    │     │  - Docker management    │
│   - Container publish   │     │  - Filesystem access    │
└─────────────────────────┘     │  - Memory/Knowledge     │
                                │  - SEMrush SEO data     │
                                └─────────────────────────┘
```

## 🚀 Services

| Service | Port | Description |
|---------|------|-------------|
| **UI** | 8000 | Web-based chat interface with multi-LLM support (Claude, GPT-4, Gemini, Llama, Mixtral) |
| **Codex** | 8001 | JWT-authenticated API for Google Tag Manager operations |
| **MCP Hub** | 8002 | Unified gateway exposing MCP servers (Firecrawl, Docker, Filesystem, Memory, SEMrush) |
| **Vault** | Internal | Secure credential storage with encryption |

## 📦 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenRouter API key (for multi-LLM access)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cwalinapj/origin-os.git
   cd origin-os
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start Origin OS:**
   ```bash
   ./origin.sh start
   ```

4. **Access the UI:**
   Open [http://localhost:8000](http://localhost:8000) in your browser.

### Stopping Services

```bash
./origin.sh stop
```

The stop command will optionally encrypt your `.env` file for security.

## ⚙️ Configuration

Create a `.env` file with the following keys:

```env
# OpenRouter (Multi-LLM access) - Required
OPENROUTER_API_KEY=sk-or-v1-xxx

# Direct API keys (optional)
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx

# Google Service Account (for GTM access)
GOOGLE_SERVICE_ACCOUNT_EMAIL=xxx@xxx.iam.gserviceaccount.com
GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY=

# AWS (optional)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# MCP Server APIs
FIRECRAWL_API_KEY=fc-xxx

# Vault encryption
VAULT_MASTER_PASSWORD=
```

## 🔌 MCP Hub Tools

The MCP Hub exposes the following tool categories:

### GTM (via Codex)
- `gtm_status` - Get GTM connection status
- `gtm_list_containers` - List GTM containers
- `gtm_list_tags` - List tags in a container
- `gtm_create_tag` - Create a new tag
- `gtm_create_trigger` - Create a new trigger
- `gtm_publish` - Publish container changes

### Firecrawl
- `scrape_url` - Scrape a single URL
- `crawl_site` - Crawl an entire website
- `extract_data` - Extract structured data
- `screenshot` - Take screenshots

### Docker
- `list_containers` - List running containers
- `run_container` - Start a new container
- `stop_container` - Stop a container
- `inspect_container` - Get container details
- `logs` - View container logs

### Filesystem
- `read_file` - Read file contents
- `write_file` - Write to a file
- `list_directory` - List directory contents
- `create_directory` - Create new directories

### Memory
- `create_entities` - Create knowledge entities
- `search_nodes` - Search the knowledge graph
- `add_observations` - Add observations to entities
- `read_graph` - Read the entire knowledge graph

## 🔒 Security

- **JWT Authentication**: All Codex endpoints require valid JWT tokens with appropriate scopes
- **Encrypted Credentials**: The `origin.sh` script can encrypt/decrypt your `.env` file using AES-256-CBC
- **Scoped Permissions**: Fine-grained access control (e.g., `gtm:read`, `gtm:write`, `gtm:publish`)

## 📂 Project Structure

```
origin-os/
├── docker-compose.yml     # Service orchestration
├── origin.sh             # Secure launcher script
├── .env.example          # Environment template
├── services/
│   ├── ui/               # Web chat interface
│   │   ├── Dockerfile
│   │   └── ui.py
│   ├── codex/            # GTM Manager API
│   │   ├── Dockerfile
│   │   ├── codex.py
│   │   └── jwt_auth.py
│   ├── mcp-hub/          # MCP Server Gateway
│   │   ├── Dockerfile
│   │   ├── mcp_hub.py
│   │   └── jwt_auth.py
│   ├── vault/            # Credential storage
│   │   └── Dockerfile
│   └── auth/             # Shared auth module
│       └── jwt_auth.py
└── schemas/              # Data schemas
    ├── experiment_container.schema.yaml
    └── mcp_auth.schema.yaml
```

## 🛠️ Development

### Running Individual Services

```bash
# Run UI service locally
cd services/ui
pip install fastapi uvicorn httpx
python ui.py

# Run Codex service locally
cd services/codex
pip install fastapi uvicorn pydantic pyjwt google-auth google-api-python-client
python codex.py
```

### Docker Compose Commands

```bash
# View logs
docker compose logs -f

# Rebuild specific service
docker compose build codex
docker compose up -d codex

# View service status
docker compose ps
```

## 📄 License

MIT License - See LICENSE for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
