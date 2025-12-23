FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @anthropic-ai/claude-code

# Clone Auto-Claude repository
RUN git clone --depth 1 https://github.com/AndyMik90/Auto-Claude.git /app/auto-claude

# Install Python dependencies
WORKDIR /app/auto-claude/apps/backend
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data persistence
RUN mkdir -p /data/auto-claude /data/worktrees /data/specs

# Environment
ENV PYTHONUNBUFFERED=1
ENV AUTO_CLAUDE_DATA_DIR=/data/auto-claude
ENV WORKTREE_DIR=/data/worktrees
ENV SPECS_DIR=/data/specs

# Run as non-root
RUN useradd -m autoclaude
RUN chown -R autoclaude:autoclaude /app /data
USER autoclaude

WORKDIR /app/auto-claude/apps/backend

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "run.py"]
