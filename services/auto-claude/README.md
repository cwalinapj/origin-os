# Auto-Claude Integration

Autonomous multi-session AI coding agent from https://github.com/AndyMik90/Auto-Claude

## Setup

The Dockerfile clones the full repo during build. To use:

1. Ensure ANTHROPIC_API_KEY is set in .env
2. Optionally set CLAUDE_CODE_OAUTH_TOKEN for Claude Code CLI
3. Run: docker-compose up -d auto-claude falkordb

## Volumes

- auto-claude-data: Persistent agent data
- auto-claude-worktrees: Git worktrees for isolated development
- auto-claude-specs: Task specifications
- falkordb-data: Graph database for memory layer

## Integration

Auto-Claude connects to:
- Codex: For governance enforcement
- Vault: For secure storage
- Memory: For cross-session context
- MCP Hub: For tool access

## Features

- **Parallel Agents**: Run multiple builds simultaneously
- **Context Engineering**: Agents understand your codebase before writing code
- **Self-Validating**: Built-in QA loop catches issues before you review
- **Isolated Workspaces**: All work happens in git worktrees
- **AI Merge Resolution**: Intelligent conflict resolution
- **Memory Layer**: Agents remember insights across sessions

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| ANTHROPIC_API_KEY | Yes | Claude API key |
| CLAUDE_CODE_OAUTH_TOKEN | No | OAuth token for Claude Code CLI |
| AUTO_BUILD_MODEL | No | Model override (default: claude-sonnet-4-20250514) |
| GRAPHITI_ENABLED | No | Enable memory layer (default: true) |
| OPENAI_API_KEY | For Memory | Required for embeddings |
