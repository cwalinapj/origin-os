# Claude Memory Export - December 23, 2025
# Origin OS Development Session
# NOTE: Credentials stored separately in /Users/root1/.env (not in git)

## USER PROFILE
- Name: Root1 
- Location: Gardnerville, Nevada, US
- Primary workspace: /Users/root1
- GitHub: cwalinapj/origin-os

---

## ORIGIN OS ARCHITECTURE

### Running Services (Docker)
| Port | Service | Container | Status |
|------|---------|-----------|--------|
| 8000 | UI | origin-ui | Running |
| 8001 | Codex | origin-codex | Running |
| 8002 | MCP Hub | origin-mcp-hub | Running (43 servers, 288 tools) |
| 8003 | CAD | origin-cad | Running |
| 8004 | Vault | origin-vault | Running |
| 8005 | Orchestrator | origin-orchestrator | Running |
| 8006 | Backup | origin-backup | Running |
| 8007 | Memory | origin-memory | Running |
| 8008 | MCP Registry | origin-mcp-registry | Running (65 servers cataloged) |

### Other Services
- 3000: Gitea (local Git)
- 8080: SupremeX Landing Page

### Docker Volumes
- origin-mcp-data
- origin-cad-data
- origin-vault-data
- origin-orchestrator-data
- origin-backup-data
- origin-memory-data
- origin-registry-data

---

## SERVICES DETAILS

### 1. UI Service (8000)
- FastAPI web interface
- Entry point for Origin OS

### 2. Codex Service (8001)
- GTM (Go-To-Market) API
- Business intelligence

### 3. MCP Hub (8002)
- 43 MCP servers
- 288 tools available
- Central orchestration for AI tools

### 4. CAD Service (8003)
- STL file generation
- 3D model creation
- Uses OpenSCAD

### 5. Vault Service (8004)
- Token envelope encryption
- Secure credential storage
- AES-256-GCM encryption

### 6. Orchestrator (8005)
- Workflow engine
- Multi-step task execution
- DAG-based workflow support

### 7. Backup Service (8006)
- Volume backups
- Multiple providers: Backblaze B2, Dropbox, Google Drive, rsync, external drive, S3
- Retention policies

### 8. Memory Service (8007)
- Conversation persistence
- EBS storage (100GB limit)
- LLM-powered compaction at 85GB
- Semantic search via embeddings
- Backblaze B2 archive

### 9. MCP Registry (8008)
- 65+ curated MCP servers
- Install/enable management
- Categories: communication, database, cloud, developer, productivity, search, ai_ml, finance, security, automation, storage, analytics

---

## MCP SERVERS IN REGISTRY

### Communication (5)
- google-workspace (Gmail, Calendar, Drive)
- slack, discord, telegram, whatsapp

### Database (10)
- postgres, sqlite, mongodb, redis, supabase
- neo4j, qdrant, pinecone, clickhouse, snowflake

### Cloud (7)
- aws, kubernetes, docker, cloudflare
- vercel, azure, pulumi

### Developer (10)
- github, gitlab, linear, jira, sentry
- playwright, puppeteer, circleci, buildkite, git

### Productivity (6)
- notion, obsidian, todoist, google-tasks, airtable, trello

### Search (5)
- brave-search, exa, tavily, firecrawl, fetch

### AI/ML (4)
- openai, anthropic, huggingface, langfuse

### Finance (2)
- stripe, plaid

### Storage (4)
- filesystem, s3, google-drive, dropbox

### Security (2)
- 1password, vault

### Automation (3)
- zapier, make, n8n

### Analytics (3)
- google-analytics, mixpanel, amplitude

### Other (4)
- memory, sequential-thinking, time, everything

---

## CURSOR MCP CONFIG

Location: ~/.cursor/mcp.json

Configured servers:
- google-workspace (Docker)
- github, memory, filesystem, sequential-thinking
- brave-search, postgres, fetch, sqlite
- puppeteer, git, time

---

## GOOGLE CLOUD SETUP

Project: titanium-atlas-391205 (My Maps Project)

### APIs Enabled
- Gmail API ✓
- Google Calendar API ✓
- Google Drive API (pending)
- Maps APIs (various)
- Cloud Storage API
- BigQuery API

### OAuth Status
- Type: External
- Status: Testing

---

## FILE LOCATIONS

### Services
- /Users/root1/services/mcp-registry/
- /Users/root1/services/memory-s3/
- /Users/root1/services/backup/
- /Users/root1/services/vault/
- /Users/root1/services/orchestrator/
- /Users/root1/services/cad/

### Config
- /Users/root1/docker-compose.yml
- /Users/root1/.env (credentials - not in git)
- /Users/root1/.cursor/mcp.json

---

## DESIGN DECISIONS

### Memory Service Architecture
- EBS for real-time writes (~1ms latency)
- Backblaze B2 for archival ($0.005/GB/month)
- LLM compaction using Claude Haiku via OpenRouter
- Embeddings via OpenAI text-embedding-3-small
- 98.4% compression ratio

### Backup Strategy
- Prefer Backblaze B2 over S3 (4.6x cheaper)
- Local/external drive for free backups

### Token Security
- Envelope encryption in Vault service
- AES-256-GCM with random IVs

---

## PENDING TASKS

1. Enable Google Drive API in Google Cloud Console
2. Configure OAuth consent screen with test user
3. Deploy Memory Service to AWS EBS
4. Configure Backblaze B2 for memory archival

---

Last updated: December 23, 2025
