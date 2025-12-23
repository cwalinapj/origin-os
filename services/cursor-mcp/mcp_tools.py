#!/usr/bin/env python3
"""
MCP Tools for Cursor Integration
=================================
Defines available MCP tools that can be called by AI agents
"""

from typing import Dict, Any, List
from pydantic import BaseModel

# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

MCP_TOOLS = {
    "read_file": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }
    },
    
    "write_file": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "create": {
                    "type": "boolean",
                    "description": "Create file if it doesn't exist",
                    "default": True
                }
            },
            "required": ["path", "content"]
        }
    },
    
    "edit_file": {
        "name": "edit_file",
        "description": "Edit a file by replacing text",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace with"
                }
            },
            "required": ["path", "old_text", "new_text"]
        }
    },
    
    "list_files": {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list"
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files",
                    "default": "*"
                }
            },
            "required": ["directory"]
        }
    },
    
    "search": {
        "name": "search",
        "description": "Search for text in files",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to search in"
                },
                "query": {
                    "type": "string",
                    "description": "Text to search for"
                }
            },
            "required": ["directory", "query"]
        }
    },
    
    "generate": {
        "name": "generate",
        "description": "Generate code using AI",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Description of code to generate"
                },
                "context_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to use as context"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language"
                },
                "model": {
                    "type": "string",
                    "description": "Model to use: claude | gpt-4 | openrouter",
                    "default": "claude"
                }
            },
            "required": ["prompt"]
        }
    },
    
    "run_command": {
        "name": "run_command",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to run"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory"
                }
            },
            "required": ["command"]
        }
    },
    
    "git_status": {
        "name": "git_status",
        "description": "Get git status of a repository",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to git repository"
                }
            },
            "required": ["path"]
        }
    },
    
    "git_commit": {
        "name": "git_commit",
        "description": "Commit changes to git",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to git repository"
                },
                "message": {
                    "type": "string",
                    "description": "Commit message"
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to commit (empty for all)"
                }
            },
            "required": ["path", "message"]
        }
    }
}


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get all tool definitions for MCP registration"""
    return list(MCP_TOOLS.values())


def get_tool(name: str) -> Dict[str, Any]:
    """Get a specific tool definition"""
    return MCP_TOOLS.get(name)


# =============================================================================
# ORIGIN OS INTEGRATION TOOLS
# =============================================================================

ORIGIN_TOOLS = {
    "codex_enforce": {
        "name": "codex_enforce",
        "description": "Check action against Codex governance",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to check"
                },
                "params": {
                    "type": "object",
                    "description": "Action parameters"
                }
            },
            "required": ["action"]
        }
    },
    
    "vault_get": {
        "name": "vault_get",
        "description": "Get secret from Vault",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Secret key"
                }
            },
            "required": ["key"]
        }
    },
    
    "vault_set": {
        "name": "vault_set",
        "description": "Store secret in Vault",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Secret key"
                },
                "value": {
                    "type": "string",
                    "description": "Secret value"
                }
            },
            "required": ["key", "value"]
        }
    },
    
    "memory_store": {
        "name": "memory_store",
        "description": "Store data in Origin OS Memory",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Memory key"
                },
                "value": {
                    "type": "object",
                    "description": "Data to store"
                }
            },
            "required": ["key", "value"]
        }
    },
    
    "memory_recall": {
        "name": "memory_recall",
        "description": "Recall data from Origin OS Memory",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    },
    
    "auto_claude_task": {
        "name": "auto_claude_task",
        "description": "Send task to Auto-Claude for autonomous coding",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task description"
                },
                "project_path": {
                    "type": "string",
                    "description": "Project directory"
                }
            },
            "required": ["task", "project_path"]
        }
    }
}


def get_all_tools() -> List[Dict[str, Any]]:
    """Get all tools including Origin OS integrations"""
    all_tools = list(MCP_TOOLS.values())
    all_tools.extend(ORIGIN_TOOLS.values())
    return all_tools
