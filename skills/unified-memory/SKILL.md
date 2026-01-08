---
name: unified-memory
description: Shared memory substrate for Claude and Origin OS agents. Use at conversation start to load persistent memories. Use when creating, reading, updating, or querying typed memories (Preference, Decision, Constraint, Goal, Procedure, Lesson, Observation, Hypothesis). Bridges Claude's built-in memory with external storage for full-fidelity persistence without compression.
---

# Unified Memory Framework

Shared typed memory between Claude and Origin OS agents.

## Memory Location

```
/Users/root1/origin-os-git/memory/store.json   # Primary store (local + GitHub)
/Users/root1/origin-os-git/memory/schema.json  # JSON Schema for validation
```

## Conversation Start Procedure

1. Read `memory/store.json` via Filesystem MCP
2. Load relevant memories into working context
3. Apply Constraints with `absolute` authority immediately

## Memory Types (8 canonical)

| Type | Authority | Use When |
|------|-----------|----------|
| **Preference** | low | Style/efficiency preference. Violations are inefficient. |
| **Decision** | high | Resolved choice. Requires `rationale` field. |
| **Constraint** | absolute | Hard boundary. Violations are failures. |
| **Goal** | medium | Time-bound objective. Set `expires` field. |
| **Procedure** | high | Tested, repeatable workflow. |
| **Lesson** | medium | Empirical conclusion from outcomes. |
| **Observation** | low | Factual statement. Time-sensitive. |
| **Hypothesis** | very_low | Unverified prediction. Exists to be tested. |

## Creating Memories

```json
{
  "id": "type-NNN",
  "type": "Decision",
  "content": "The decision or fact",
  "rationale": "Why (required for Decision/Constraint)",
  "authority": "high",
  "tags": ["category"],
  "provenance": {
    "source": "claude",
    "timestamp": "2026-01-08T12:00:00Z",
    "llm_model": "claude-opus-4-5",
    "interaction_id": "conversation-id"
  },
  "expires": null,
  "promoted_from": null
}
```

### ID Format
- `pref-001` for Preference
- `dec-001` for Decision  
- `const-001` for Constraint
- `goal-001` for Goal
- `proc-001` for Procedure
- `less-001` for Lesson
- `obs-001` for Observation
- `hyp-001` for Hypothesis

## Provenance Sources

| Source | When |
|--------|------|
| `human` | Paul explicitly stated |
| `claude` | Claude inferred/created (this interface) |
| `agent` | Origin OS agent created (set `agent_id`) |
| `system` | Automated system process |

## Writing Memories

After creating/updating memories:
1. Update `last_updated` timestamp
2. Update `indexes.by_type` and `indexes.by_tag`
3. Write to `memory/store.json`
4. Optionally commit to git: `git -C /Users/root1/origin-os-git add memory/ && git commit -m "memory: <summary>"`

## Conflict Resolution

When memories conflict:
1. `absolute` authority wins
2. `exclusive` conflicts require human resolution
3. `overridable` yields to higher authority
4. Newer timestamp wins for same authority level

## Promotion Paths

```
Observation → Hypothesis → Lesson → Procedure
                              ↘ → Constraint
Preference → Decision → Constraint
```

When promoting, set `promoted_from` to original memory ID.

## Security

**NEVER store in memory:**
- API keys, tokens, passwords
- Personal identifiers (SSN, etc.)
- Financial credentials

Use Vault service (port 8004) for secrets.

## Git Sync

The memory store syncs to:
- GitHub: `cwalinapj/origin-os` (origin remote)
- Gitea: `localhost:3000/paul/origin-os` (gitea remote)

Push after significant memory updates.

## Example: Reading at Conversation Start

```python
# For Origin OS agents
import json
from pathlib import Path

store = json.loads(Path("/Users/root1/origin-os-git/memory/store.json").read_text())
constraints = [m for m in store["memories"] if m["type"] == "Constraint"]
# Apply constraints immediately
```

## Example: Claude Reading

Use Filesystem:read_file on `/Users/root1/origin-os-git/memory/store.json` at conversation start when persistent context is needed.
