#!/usr/bin/env python3
"""
Unified Memory Store - Read/Write operations for Claude and Origin OS agents.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

STORE_PATH = Path("/Users/root1/origin-os-git/memory/store.json")

MemoryType = Literal["Preference", "Decision", "Constraint", "Goal", "Procedure", "Lesson", "Observation", "Hypothesis"]
Authority = Literal["very_low", "low", "medium", "high", "absolute"]
Source = Literal["human", "claude", "agent", "system"]

TYPE_PREFIXES = {
    "Preference": "pref",
    "Decision": "dec",
    "Constraint": "const",
    "Goal": "goal",
    "Procedure": "proc",
    "Lesson": "less",
    "Observation": "obs",
    "Hypothesis": "hyp",
}

TYPE_AUTHORITY = {
    "Preference": "low",
    "Decision": "high",
    "Constraint": "absolute",
    "Goal": "medium",
    "Procedure": "high",
    "Lesson": "medium",
    "Observation": "low",
    "Hypothesis": "very_low",
}


def load_store() -> dict:
    """Load the memory store."""
    if STORE_PATH.exists():
        return json.loads(STORE_PATH.read_text())
    return {
        "version": "1.0",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "owner": "paul",
        "memories": [],
        "indexes": {"by_type": {t: [] for t in TYPE_PREFIXES}, "by_tag": {}},
    }


def save_store(store: dict) -> None:
    """Save the memory store."""
    store["last_updated"] = datetime.now(timezone.utc).isoformat()
    STORE_PATH.write_text(json.dumps(store, indent=2))


def generate_id(memory_type: MemoryType, store: dict) -> str:
    """Generate next ID for memory type."""
    prefix = TYPE_PREFIXES[memory_type]
    existing = [m["id"] for m in store["memories"] if m["id"].startswith(prefix)]
    if not existing:
        return f"{prefix}-001"
    max_num = max(int(m.split("-")[1]) for m in existing)
    return f"{prefix}-{max_num + 1:03d}"


def add_memory(
    memory_type: MemoryType,
    content: str,
    source: Source,
    tags: list[str] = None,
    rationale: str = None,
    expires: str = None,
    agent_id: str = None,
    llm_model: str = None,
    interaction_id: str = None,
) -> dict:
    """Add a new memory to the store."""
    store = load_store()
    memory_id = generate_id(memory_type, store)
    
    memory = {
        "id": memory_id,
        "type": memory_type,
        "content": content,
        "authority": TYPE_AUTHORITY[memory_type],
        "tags": tags or [],
        "provenance": {
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "expires": expires,
        "promoted_from": None,
    }
    
    if rationale and memory_type in ("Decision", "Constraint"):
        memory["rationale"] = rationale
    
    if llm_model:
        memory["provenance"]["llm_model"] = llm_model
    if agent_id:
        memory["provenance"]["agent_id"] = agent_id
    if interaction_id:
        memory["provenance"]["interaction_id"] = interaction_id
    
    store["memories"].append(memory)
    
    # Update indexes
    if memory_type not in store["indexes"]["by_type"]:
        store["indexes"]["by_type"][memory_type] = []
    store["indexes"]["by_type"][memory_type].append(memory_id)
    
    for tag in (tags or []):
        if tag not in store["indexes"]["by_tag"]:
            store["indexes"]["by_tag"][tag] = []
        store["indexes"]["by_tag"][tag].append(memory_id)
    
    save_store(store)
    return memory


def get_memories(
    memory_type: MemoryType = None,
    tag: str = None,
    min_authority: Authority = None,
) -> list[dict]:
    """Query memories with optional filters."""
    store = load_store()
    memories = store["memories"]
    
    if memory_type:
        memories = [m for m in memories if m["type"] == memory_type]
    
    if tag:
        memories = [m for m in memories if tag in m.get("tags", [])]
    
    if min_authority:
        authority_order = ["very_low", "low", "medium", "high", "absolute"]
        min_idx = authority_order.index(min_authority)
        memories = [m for m in memories if authority_order.index(m["authority"]) >= min_idx]
    
    return memories


def get_constraints() -> list[dict]:
    """Get all active constraints (critical for agent behavior)."""
    return get_memories(memory_type="Constraint")


def get_procedures() -> list[dict]:
    """Get all procedures (workflows)."""
    return get_memories(memory_type="Procedure")


def promote_memory(memory_id: str, new_type: MemoryType, rationale: str = None) -> dict:
    """Promote a memory to a higher type."""
    store = load_store()
    
    # Find original
    original = next((m for m in store["memories"] if m["id"] == memory_id), None)
    if not original:
        raise ValueError(f"Memory {memory_id} not found")
    
    # Create promoted version
    new_memory = add_memory(
        memory_type=new_type,
        content=original["content"],
        source=original["provenance"]["source"],
        tags=original.get("tags", []),
        rationale=rationale,
    )
    
    # Mark promoted_from
    store = load_store()
    for m in store["memories"]:
        if m["id"] == new_memory["id"]:
            m["promoted_from"] = memory_id
            break
    save_store(store)
    
    return new_memory


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Memory CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument("--type", choices=list(TYPE_PREFIXES.keys()))
    list_parser.add_argument("--tag")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a memory")
    add_parser.add_argument("type", choices=list(TYPE_PREFIXES.keys()))
    add_parser.add_argument("content")
    add_parser.add_argument("--source", default="human", choices=["human", "claude", "agent", "system"])
    add_parser.add_argument("--tags", nargs="+", default=[])
    add_parser.add_argument("--rationale")
    
    # Constraints command
    subparsers.add_parser("constraints", help="List all constraints")
    
    args = parser.parse_args()
    
    if args.command == "list":
        memories = get_memories(memory_type=args.type, tag=args.tag)
        for m in memories:
            print(f"[{m['id']}] {m['type']}: {m['content'][:60]}...")
    
    elif args.command == "add":
        memory = add_memory(
            memory_type=args.type,
            content=args.content,
            source=args.source,
            tags=args.tags,
            rationale=args.rationale,
        )
        print(f"Created: {memory['id']}")
    
    elif args.command == "constraints":
        for c in get_constraints():
            print(f"[{c['id']}] {c['content']}")
    
    else:
        parser.print_help()
