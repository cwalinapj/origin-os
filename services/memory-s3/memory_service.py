#!/usr/bin/env python3
"""
Memory MCP Server
=================
Intelligent conversation persistence with automatic compaction and semantic encoding.

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EBS (100GB Max, us-west-2)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  /data/memory/                                                              │
│  ├── conversations/          # Active conversations (raw JSON)             │
│  ├── pending_sync/           # Queue for Backblaze sync                    │
│  ├── compacted/              # LLM-summarized conversations                │
│  ├── embeddings/             # Dense vector encodings (FAISS index)        │
│  └── index/                  # Search index + metadata                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (Background Worker)
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Backblaze B2 (Archive, $0.005/GB/month)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  origin-os-memory/                                                          │
│  ├── raw/                    # Original conversations (archived)           │
│  ├── compacted/              # Summaries + embeddings                       │
│  └── snapshots/              # Periodic full backups                       │
└─────────────────────────────────────────────────────────────────────────────┘

Compaction Flow:
1. Monitor EBS usage (target: <90GB, trigger at 85GB)
2. Select oldest non-compacted conversations
3. Generate LLM summary (preserves key facts, decisions, context)
4. Create embedding vector for semantic search
5. Archive raw to Backblaze, delete from EBS
6. Keep only: summary + embedding + metadata locally

Encoding Efficiency:
- Raw conversation: ~500KB (1000 turns)
- LLM Summary: ~2KB (preserves meaning)
- Embedding: ~6KB (1536-dim float32)
- Total: ~8KB = 98.4% compression

Search Capability:
- Semantic search via embeddings (find similar conversations)
- Keyword search via summaries
- Full retrieval from Backblaze if needed
"""

import os
import json
import hashlib
import asyncio
import gzip
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import threading
import time
import shutil
import struct

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import httpx
import uvicorn

app = FastAPI(title="Memory MCP Server", version="1.0")

# =============================================================================
# CONFIGURATION
# =============================================================================

# EBS Storage (us-west-2)
EBS_DATA_DIR = Path(os.getenv("EBS_DATA_DIR", "/data/memory"))
EBS_MAX_SIZE_GB = int(os.getenv("EBS_MAX_SIZE_GB", "100"))
EBS_COMPACT_TRIGGER_GB = int(os.getenv("EBS_COMPACT_TRIGGER_GB", "85"))
EBS_TARGET_SIZE_GB = int(os.getenv("EBS_TARGET_SIZE_GB", "70"))

# Directories
CONVERSATIONS_DIR = EBS_DATA_DIR / "conversations"
PENDING_SYNC_DIR = EBS_DATA_DIR / "pending_sync"  
COMPACTED_DIR = EBS_DATA_DIR / "compacted"
EMBEDDINGS_DIR = EBS_DATA_DIR / "embeddings"
INDEX_DIR = EBS_DATA_DIR / "index"

# Backblaze B2
B2_ENABLED = os.getenv("B2_ENABLED", "true").lower() == "true"
B2_KEY_ID = os.getenv("B2_KEY_ID", "")
B2_APP_KEY = os.getenv("B2_APP_KEY", "")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "origin-os-memory")

# Worker timing
SYNC_INTERVAL_SECONDS = int(os.getenv("SYNC_INTERVAL_SECONDS", "60"))
COMPACTION_CHECK_INTERVAL_SECONDS = int(os.getenv("COMPACTION_CHECK_INTERVAL_SECONDS", "300"))
COMPACTION_MIN_AGE_HOURS = int(os.getenv("COMPACTION_MIN_AGE_HOURS", "24"))
COMPACTION_MIN_TURNS = int(os.getenv("COMPACTION_MIN_TURNS", "10"))

# LLM Configuration
LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_SUMMARY_MODEL = os.getenv("LLM_SUMMARY_MODEL", "anthropic/claude-3-haiku")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://api.openai.com/v1/embeddings")
EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# Create directories
for dir_path in [CONVERSATIONS_DIR, PENDING_SYNC_DIR, COMPACTED_DIR, EMBEDDINGS_DIR, INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STATE TRACKING
# =============================================================================

server_state = {
    "ebs_usage_gb": 0,
    "conversation_count": 0,
    "compacted_count": 0,
    "embedding_count": 0,
    "last_sync": None,
    "last_compaction": None,
    "is_syncing": False,
    "is_compacting": False,
    "errors": []
}

# In-memory embedding index for fast search
embedding_index = {
    "ids": [],           # conversation_ids
    "vectors": None,     # numpy array of embeddings
    "metadata": {}       # id -> {summary, created_at, tags, ...}
}

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ConversationTurn:
    role: str  # user, assistant, system
    content: str
    timestamp: str
    turn_id: str
    metadata: Optional[Dict] = None

@dataclass
class ConversationMeta:
    conversation_id: str
    created_at: str
    updated_at: str
    turn_count: int
    size_bytes: int = 0
    is_compacted: bool = False
    is_synced: bool = False
    tags: List[str] = field(default_factory=list)

@dataclass
class CompactedConversation:
    conversation_id: str
    original_turn_count: int
    original_size_bytes: int
    summary: str
    key_facts: List[str]
    decisions: List[str]
    topics: List[str]
    embedding: Optional[List[float]] = None
    created_at: str = ""
    compacted_at: str = ""

class SaveTurnRequest(BaseModel):
    conversation_id: str
    role: str
    content: str
    metadata: Optional[Dict] = None

class NewConversationRequest(BaseModel):
    conversation_id: Optional[str] = None
    initial_context: Optional[str] = None
    tags: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    search_type: str = "semantic"  # semantic, keyword, hybrid
    min_similarity: float = 0.7

class RecallRequest(BaseModel):
    conversation_id: str
    fetch_from_archive: bool = False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_ebs_usage_gb() -> float:
    """Get current EBS usage in GB"""
    total_size = 0
    for path in EBS_DATA_DIR.rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size / (1024 ** 3)

def generate_id(prefix: str = "conv") -> str:
    """Generate unique ID"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    random_hash = hashlib.md5(f"{timestamp}{time.time()}".encode()).hexdigest()[:8]
    return f"{prefix}_{timestamp[:14]}_{random_hash}"

def get_conversation_path(conversation_id: str) -> Path:
    """Get path to conversation directory"""
    return CONVERSATIONS_DIR / conversation_id

def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file"""
    try:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None

def save_json(path: Path, data: Dict) -> bool:
    """Save JSON file atomically"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        temp_path.rename(path)
        return True
    except Exception as e:
        print(f"Error saving {path}: {e}")
        return False

def compress_data(data: bytes) -> bytes:
    """Compress data using gzip"""
    return gzip.compress(data, compresslevel=9)

def decompress_data(data: bytes) -> bytes:
    """Decompress gzip data"""
    return gzip.decompress(data)

# =============================================================================
# CONVERSATION OPERATIONS (EBS - FAST)
# =============================================================================

def create_conversation(conversation_id: Optional[str] = None, tags: List[str] = None) -> str:
    """Create a new conversation"""
    if not conversation_id:
        conversation_id = generate_id("conv")
    
    conv_dir = get_conversation_path(conversation_id)
    conv_dir.mkdir(parents=True, exist_ok=True)
    
    now = datetime.now(timezone.utc).isoformat()
    meta = ConversationMeta(
        conversation_id=conversation_id,
        created_at=now,
        updated_at=now,
        turn_count=0,
        tags=tags or []
    )
    
    save_json(conv_dir / "meta.json", asdict(meta))
    save_json(conv_dir / "turns.json", {"turns": []})
    
    # Add to pending sync queue
    queue_for_sync(conversation_id, "create")
    
    return conversation_id

def save_turn(conversation_id: str, role: str, content: str, metadata: Dict = None) -> Dict:
    """Save a conversation turn immediately to EBS"""
    conv_dir = get_conversation_path(conversation_id)
    
    # Create conversation if doesn't exist
    if not conv_dir.exists():
        create_conversation(conversation_id)
    
    # Load existing turns
    turns_path = conv_dir / "turns.json"
    turns_data = load_json(turns_path) or {"turns": []}
    
    # Create new turn
    turn = ConversationTurn(
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc).isoformat(),
        turn_id=generate_id("turn"),
        metadata=metadata
    )
    
    turns_data["turns"].append(asdict(turn))
    
    # Save turns
    save_json(turns_path, turns_data)
    
    # Update metadata
    meta_path = conv_dir / "meta.json"
    meta = load_json(meta_path) or {}
    meta["updated_at"] = turn.timestamp
    meta["turn_count"] = len(turns_data["turns"])
    meta["size_bytes"] = turns_path.stat().st_size
    save_json(meta_path, meta)
    
    # Queue for sync
    queue_for_sync(conversation_id, "update")
    
    return asdict(turn)

def get_conversation(conversation_id: str, max_turns: int = None) -> Optional[Dict]:
    """Get conversation from EBS"""
    conv_dir = get_conversation_path(conversation_id)
    
    if not conv_dir.exists():
        # Check compacted
        compacted = get_compacted_conversation(conversation_id)
        if compacted:
            return compacted
        return None
    
    meta = load_json(conv_dir / "meta.json")
    turns_data = load_json(conv_dir / "turns.json") or {"turns": []}
    
    turns = turns_data["turns"]
    if max_turns and len(turns) > max_turns:
        turns = turns[-max_turns:]
    
    return {
        "meta": meta,
        "turns": turns,
        "is_compacted": False
    }

def get_compacted_conversation(conversation_id: str) -> Optional[Dict]:
    """Get compacted conversation"""
    compacted_path = COMPACTED_DIR / f"{conversation_id}.json.gz"
    
    if compacted_path.exists():
        with gzip.open(compacted_path, 'rt') as f:
            data = json.load(f)
        return {
            "meta": {
                "conversation_id": conversation_id,
                "created_at": data.get("created_at"),
                "is_compacted": True
            },
            "summary": data.get("summary"),
            "key_facts": data.get("key_facts", []),
            "decisions": data.get("decisions", []),
            "topics": data.get("topics", []),
            "original_turn_count": data.get("original_turn_count", 0),
            "is_compacted": True
        }
    return None

def queue_for_sync(conversation_id: str, action: str):
    """Add conversation to sync queue"""
    queue_file = PENDING_SYNC_DIR / f"{conversation_id}.json"
    save_json(queue_file, {
        "conversation_id": conversation_id,
        "action": action,
        "queued_at": datetime.now(timezone.utc).isoformat()
    })

def list_conversations(limit: int = 100, include_compacted: bool = True) -> List[Dict]:
    """List all conversations"""
    conversations = []
    
    # Active conversations
    for conv_dir in CONVERSATIONS_DIR.iterdir():
        if conv_dir.is_dir():
            meta = load_json(conv_dir / "meta.json")
            if meta:
                meta["is_compacted"] = False
                conversations.append(meta)
    
    # Compacted conversations
    if include_compacted:
        for compacted_file in COMPACTED_DIR.glob("*.json.gz"):
            conv_id = compacted_file.stem.replace(".json", "")
            with gzip.open(compacted_file, 'rt') as f:
                data = json.load(f)
            conversations.append({
                "conversation_id": conv_id,
                "created_at": data.get("created_at"),
                "compacted_at": data.get("compacted_at"),
                "is_compacted": True,
                "original_turn_count": data.get("original_turn_count", 0)
            })
    
    # Sort by updated_at/created_at
    conversations.sort(
        key=lambda x: x.get("updated_at") or x.get("created_at", ""),
        reverse=True
    )
    
    return conversations[:limit]

# =============================================================================
# LLM SUMMARIZATION
# =============================================================================

async def generate_summary(turns: List[Dict]) -> Dict:
    """Generate LLM summary of conversation"""
    if not LLM_API_KEY:
        return {
            "summary": "No LLM API key configured for summarization",
            "key_facts": [],
            "decisions": [],
            "topics": []
        }
    
    # Build conversation text
    conv_text = "\n".join([
        f"{t['role'].upper()}: {t['content'][:1000]}"  # Truncate long messages
        for t in turns[-100:]  # Last 100 turns max
    ])
    
    prompt = f"""Analyze this conversation and provide a structured summary.

CONVERSATION:
{conv_text}

Respond in JSON format:
{{
    "summary": "A comprehensive 2-3 paragraph summary preserving key context, decisions, and outcomes",
    "key_facts": ["Important facts, names, numbers, dates mentioned"],
    "decisions": ["Decisions made or conclusions reached"],
    "topics": ["Main topics/themes discussed"]
}}

Focus on preserving information that would be useful for continuing this conversation later."""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                LLM_API_URL,
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_SUMMARY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON from response
                try:
                    # Try to extract JSON
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        return json.loads(json_match.group())
                except:
                    pass
                
                return {
                    "summary": content,
                    "key_facts": [],
                    "decisions": [],
                    "topics": []
                }
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    return {
        "summary": f"Conversation with {len(turns)} turns",
        "key_facts": [],
        "decisions": [],
        "topics": []
    }

async def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding vector for text"""
    if not EMBEDDING_API_KEY:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                EMBEDDING_API_URL,
                headers={
                    "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text[:8000]  # Truncate to model limit
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
    
    return None

# =============================================================================
# COMPACTION ENGINE
# =============================================================================

async def compact_conversation(conversation_id: str) -> bool:
    """Compact a single conversation"""
    conv_dir = get_conversation_path(conversation_id)
    
    if not conv_dir.exists():
        return False
    
    # Load conversation
    meta = load_json(conv_dir / "meta.json")
    turns_data = load_json(conv_dir / "turns.json") or {"turns": []}
    turns = turns_data["turns"]
    
    if not turns:
        return False
    
    print(f"Compacting conversation {conversation_id} ({len(turns)} turns)")
    
    # Generate summary
    summary_data = await generate_summary(turns)
    
    # Generate embedding for semantic search
    embedding_text = f"{summary_data['summary']} {' '.join(summary_data['key_facts'])}"
    embedding = await generate_embedding(embedding_text)
    
    # Create compacted record
    compacted = CompactedConversation(
        conversation_id=conversation_id,
        original_turn_count=len(turns),
        original_size_bytes=meta.get("size_bytes", 0),
        summary=summary_data["summary"],
        key_facts=summary_data["key_facts"],
        decisions=summary_data["decisions"],
        topics=summary_data["topics"],
        embedding=embedding,
        created_at=meta.get("created_at", ""),
        compacted_at=datetime.now(timezone.utc).isoformat()
    )
    
    # Save compacted version (compressed)
    compacted_path = COMPACTED_DIR / f"{conversation_id}.json.gz"
    with gzip.open(compacted_path, 'wt') as f:
        json.dump(asdict(compacted), f)
    
    # Save embedding separately for fast loading
    if embedding:
        embedding_path = EMBEDDINGS_DIR / f"{conversation_id}.npy"
        np.save(embedding_path, np.array(embedding, dtype=np.float32))
        
        # Update in-memory index
        update_embedding_index(conversation_id, embedding, {
            "summary": summary_data["summary"][:500],
            "created_at": meta.get("created_at"),
            "topics": summary_data["topics"]
        })
    
    # Queue raw conversation for Backblaze archive before deleting
    archive_to_backblaze(conversation_id, conv_dir)
    
    # Delete raw conversation from EBS
    shutil.rmtree(conv_dir)
    
    print(f"Compacted {conversation_id}: {meta.get('size_bytes', 0)} bytes → {compacted_path.stat().st_size} bytes")
    
    return True

def update_embedding_index(conversation_id: str, embedding: List[float], metadata: Dict):
    """Update in-memory embedding index"""
    global embedding_index
    
    if conversation_id in embedding_index["ids"]:
        idx = embedding_index["ids"].index(conversation_id)
        embedding_index["vectors"][idx] = embedding
        embedding_index["metadata"][conversation_id] = metadata
    else:
        embedding_index["ids"].append(conversation_id)
        embedding_index["metadata"][conversation_id] = metadata
        
        new_vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if embedding_index["vectors"] is None:
            embedding_index["vectors"] = new_vector
        else:
            embedding_index["vectors"] = np.vstack([
                embedding_index["vectors"],
                new_vector
            ])

def load_embedding_index():
    """Load all embeddings into memory index"""
    global embedding_index
    
    embedding_files = list(EMBEDDINGS_DIR.glob("*.npy"))
    if not embedding_files:
        return
    
    ids = []
    vectors = []
    metadata = {}
    
    for emb_file in embedding_files:
        conv_id = emb_file.stem
        vector = np.load(emb_file)
        
        # Load metadata from compacted file
        compacted_path = COMPACTED_DIR / f"{conv_id}.json.gz"
        if compacted_path.exists():
            with gzip.open(compacted_path, 'rt') as f:
                data = json.load(f)
            meta = {
                "summary": data.get("summary", "")[:500],
                "created_at": data.get("created_at"),
                "topics": data.get("topics", [])
            }
        else:
            meta = {}
        
        ids.append(conv_id)
        vectors.append(vector)
        metadata[conv_id] = meta
    
    embedding_index["ids"] = ids
    embedding_index["vectors"] = np.vstack(vectors) if vectors else None
    embedding_index["metadata"] = metadata
    
    print(f"Loaded {len(ids)} embeddings into index")

async def run_compaction_cycle():
    """Run compaction if EBS usage exceeds threshold"""
    global server_state
    
    usage_gb = get_ebs_usage_gb()
    server_state["ebs_usage_gb"] = usage_gb
    
    if usage_gb < EBS_COMPACT_TRIGGER_GB:
        return
    
    if server_state["is_compacting"]:
        return
    
    server_state["is_compacting"] = True
    print(f"EBS usage {usage_gb:.1f}GB exceeds {EBS_COMPACT_TRIGGER_GB}GB, starting compaction...")
    
    try:
        # Get oldest conversations
        conversations = []
        for conv_dir in CONVERSATIONS_DIR.iterdir():
            if conv_dir.is_dir():
                meta = load_json(conv_dir / "meta.json")
                if meta:
                    conversations.append({
                        "id": meta["conversation_id"],
                        "updated_at": meta.get("updated_at", ""),
                        "turn_count": meta.get("turn_count", 0),
                        "size_bytes": meta.get("size_bytes", 0)
                    })
        
        # Sort by age (oldest first)
        conversations.sort(key=lambda x: x["updated_at"])
        
        # Filter: only compact if old enough and has enough turns
        min_age = datetime.now(timezone.utc) - timedelta(hours=COMPACTION_MIN_AGE_HOURS)
        eligible = [
            c for c in conversations
            if c["updated_at"] < min_age.isoformat()
            and c["turn_count"] >= COMPACTION_MIN_TURNS
        ]
        
        # Compact until we're under target
        compacted_count = 0
        for conv in eligible:
            if get_ebs_usage_gb() < EBS_TARGET_SIZE_GB:
                break
            
            try:
                await compact_conversation(conv["id"])
                compacted_count += 1
            except Exception as e:
                print(f"Error compacting {conv['id']}: {e}")
                server_state["errors"].append(str(e))
        
        server_state["compacted_count"] += compacted_count
        server_state["last_compaction"] = datetime.now(timezone.utc).isoformat()
        print(f"Compaction complete: {compacted_count} conversations compacted")
        
    finally:
        server_state["is_compacting"] = False

# =============================================================================
# BACKBLAZE B2 SYNC
# =============================================================================

b2_api = None
b2_bucket = None

def init_b2():
    """Initialize Backblaze B2 client"""
    global b2_api, b2_bucket
    
    if not B2_ENABLED or not B2_KEY_ID or not B2_APP_KEY:
        return False
    
    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
        
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
        print(f"Connected to Backblaze B2 bucket: {B2_BUCKET_NAME}")
        return True
    except Exception as e:
        print(f"Error initializing B2: {e}")
        return False

def archive_to_backblaze(conversation_id: str, conv_dir: Path):
    """Archive raw conversation to Backblaze"""
    if not b2_bucket:
        return
    
    try:
        # Create archive file
        archive_path = PENDING_SYNC_DIR / f"{conversation_id}_archive.json.gz"
        
        meta = load_json(conv_dir / "meta.json")
        turns = load_json(conv_dir / "turns.json")
        
        archive_data = {
            "meta": meta,
            "turns": turns,
            "archived_at": datetime.now(timezone.utc).isoformat()
        }
        
        with gzip.open(archive_path, 'wt') as f:
            json.dump(archive_data, f)
        
        # Upload to B2
        b2_path = f"raw/{conversation_id}/archive.json.gz"
        b2_bucket.upload_local_file(str(archive_path), b2_path)
        
        # Cleanup
        archive_path.unlink()
        
        print(f"Archived {conversation_id} to Backblaze")
        
    except Exception as e:
        print(f"Error archiving to Backblaze: {e}")

async def sync_to_backblaze():
    """Sync pending items to Backblaze"""
    global server_state
    
    if not b2_bucket:
        return
    
    if server_state["is_syncing"]:
        return
    
    server_state["is_syncing"] = True
    
    try:
        pending = list(PENDING_SYNC_DIR.glob("*.json"))
        
        for pending_file in pending:
            if pending_file.name.endswith("_archive.json.gz"):
                continue  # Skip archive files
                
            try:
                data = load_json(pending_file)
                conv_id = data["conversation_id"]
                action = data["action"]
                
                # For create/update, sync the conversation
                conv_dir = get_conversation_path(conv_id)
                if conv_dir.exists():
                    meta = load_json(conv_dir / "meta.json")
                    turns = load_json(conv_dir / "turns.json")
                    
                    sync_data = {"meta": meta, "turns": turns}
                    
                    # Upload to B2
                    b2_path = f"active/{conv_id}/data.json"
                    temp_path = PENDING_SYNC_DIR / f"{conv_id}_temp.json"
                    save_json(temp_path, sync_data)
                    b2_bucket.upload_local_file(str(temp_path), b2_path)
                    temp_path.unlink()
                
                # Remove from queue
                pending_file.unlink()
                server_state["synced_total"] = server_state.get("synced_total", 0) + 1
                
            except Exception as e:
                print(f"Error syncing {pending_file}: {e}")
        
        server_state["last_sync"] = datetime.now(timezone.utc).isoformat()
        
    finally:
        server_state["is_syncing"] = False

def retrieve_from_backblaze(conversation_id: str) -> Optional[Dict]:
    """Retrieve archived conversation from Backblaze"""
    if not b2_bucket:
        return None
    
    try:
        # Try raw archive first
        b2_path = f"raw/{conversation_id}/archive.json.gz"
        
        temp_path = EBS_DATA_DIR / f"temp_{conversation_id}.json.gz"
        
        downloaded = b2_bucket.download_file_by_name(b2_path)
        downloaded.save_to(str(temp_path))
        
        with gzip.open(temp_path, 'rt') as f:
            data = json.load(f)
        
        temp_path.unlink()
        return data
        
    except Exception as e:
        print(f"Error retrieving from Backblaze: {e}")
        return None

# =============================================================================
# SEMANTIC SEARCH
# =============================================================================

async def semantic_search(query: str, limit: int = 10, min_similarity: float = 0.7) -> List[Dict]:
    """Search conversations using semantic similarity"""
    if embedding_index["vectors"] is None or len(embedding_index["ids"]) == 0:
        return []
    
    # Generate query embedding
    query_embedding = await generate_embedding(query)
    if not query_embedding:
        return []
    
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    # Compute cosine similarity
    vectors = embedding_index["vectors"]
    similarities = np.dot(vectors, query_vector) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
    )
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:limit]
    
    results = []
    for idx in top_indices:
        similarity = float(similarities[idx])
        if similarity < min_similarity:
            break
        
        conv_id = embedding_index["ids"][idx]
        metadata = embedding_index["metadata"].get(conv_id, {})
        
        results.append({
            "conversation_id": conv_id,
            "similarity": similarity,
            "summary": metadata.get("summary", ""),
            "created_at": metadata.get("created_at"),
            "topics": metadata.get("topics", [])
        })
    
    return results

def keyword_search(query: str, limit: int = 10) -> List[Dict]:
    """Search conversations using keyword matching"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    results = []
    
    # Search active conversations
    for conv_dir in CONVERSATIONS_DIR.iterdir():
        if conv_dir.is_dir():
            turns_data = load_json(conv_dir / "turns.json")
            if not turns_data:
                continue
            
            # Search in turns
            matches = 0
            for turn in turns_data.get("turns", []):
                content_lower = turn.get("content", "").lower()
                matches += sum(1 for word in query_words if word in content_lower)
            
            if matches > 0:
                meta = load_json(conv_dir / "meta.json") or {}
                results.append({
                    "conversation_id": meta.get("conversation_id"),
                    "matches": matches,
                    "turn_count": meta.get("turn_count", 0),
                    "updated_at": meta.get("updated_at"),
                    "is_compacted": False
                })
    
    # Search compacted conversations
    for compacted_file in COMPACTED_DIR.glob("*.json.gz"):
        with gzip.open(compacted_file, 'rt') as f:
            data = json.load(f)
        
        # Search in summary and key facts
        text = f"{data.get('summary', '')} {' '.join(data.get('key_facts', []))}".lower()
        matches = sum(1 for word in query_words if word in text)
        
        if matches > 0:
            results.append({
                "conversation_id": data.get("conversation_id"),
                "matches": matches,
                "summary": data.get("summary", "")[:200],
                "is_compacted": True
            })
    
    # Sort by matches
    results.sort(key=lambda x: x["matches"], reverse=True)
    
    return results[:limit]

# =============================================================================
# BACKGROUND WORKERS
# =============================================================================

async def background_sync_worker():
    """Background worker for Backblaze sync"""
    while True:
        try:
            await sync_to_backblaze()
        except Exception as e:
            print(f"Sync worker error: {e}")
        await asyncio.sleep(SYNC_INTERVAL_SECONDS)

async def background_compaction_worker():
    """Background worker for compaction"""
    while True:
        try:
            await run_compaction_cycle()
        except Exception as e:
            print(f"Compaction worker error: {e}")
        await asyncio.sleep(COMPACTION_CHECK_INTERVAL_SECONDS)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    # Initialize B2
    init_b2()
    
    # Load embedding index
    load_embedding_index()
    
    # Update state
    server_state["ebs_usage_gb"] = get_ebs_usage_gb()
    server_state["conversation_count"] = len(list(CONVERSATIONS_DIR.iterdir()))
    server_state["compacted_count"] = len(list(COMPACTED_DIR.glob("*.json.gz")))
    server_state["embedding_count"] = len(embedding_index["ids"])
    
    # Start background workers
    asyncio.create_task(background_sync_worker())
    asyncio.create_task(background_compaction_worker())
    
    print(f"Memory MCP Server started")
    print(f"  EBS Usage: {server_state['ebs_usage_gb']:.2f} GB / {EBS_MAX_SIZE_GB} GB")
    print(f"  Conversations: {server_state['conversation_count']} active, {server_state['compacted_count']} compacted")
    print(f"  Embeddings: {server_state['embedding_count']}")

@app.get("/")
async def root():
    """Server info and status"""
    server_state["ebs_usage_gb"] = get_ebs_usage_gb()
    
    return {
        "service": "Memory MCP Server",
        "version": "1.0",
        "status": "healthy",
        "config": {
            "ebs_max_gb": EBS_MAX_SIZE_GB,
            "ebs_compact_trigger_gb": EBS_COMPACT_TRIGGER_GB,
            "ebs_target_gb": EBS_TARGET_SIZE_GB,
            "b2_enabled": B2_ENABLED,
            "b2_bucket": B2_BUCKET_NAME if B2_ENABLED else None,
            "sync_interval_seconds": SYNC_INTERVAL_SECONDS,
            "compaction_min_age_hours": COMPACTION_MIN_AGE_HOURS
        },
        "state": server_state,
        "endpoints": {
            "conversations": {
                "create": "POST /conversations",
                "list": "GET /conversations",
                "get": "GET /conversations/{id}",
                "save_turn": "POST /conversations/{id}/turns"
            },
            "search": {
                "semantic": "POST /search/semantic",
                "keyword": "POST /search/keyword"
            },
            "admin": {
                "compact_now": "POST /admin/compact",
                "sync_now": "POST /admin/sync",
                "stats": "GET /admin/stats"
            }
        }
    }

@app.post("/conversations")
async def create_new_conversation(req: NewConversationRequest):
    """Create a new conversation"""
    conversation_id = create_conversation(req.conversation_id, req.tags)
    
    # Save initial context if provided
    if req.initial_context:
        save_turn(conversation_id, "system", req.initial_context)
    
    return {
        "conversation_id": conversation_id,
        "created": True
    }

@app.get("/conversations")
async def list_all_conversations(limit: int = 100, include_compacted: bool = True):
    """List all conversations"""
    conversations = list_conversations(limit, include_compacted)
    return {
        "conversations": conversations,
        "count": len(conversations)
    }

@app.get("/conversations/{conversation_id}")
async def get_conversation_by_id(conversation_id: str, max_turns: int = None):
    """Get a specific conversation"""
    conv = get_conversation(conversation_id, max_turns)
    
    if not conv:
        raise HTTPException(404, "Conversation not found")
    
    return conv

@app.post("/conversations/{conversation_id}/turns")
async def save_conversation_turn(conversation_id: str, req: SaveTurnRequest):
    """Save a turn to a conversation"""
    turn = save_turn(conversation_id, req.role, req.content, req.metadata)
    return {
        "saved": True,
        "turn": turn
    }

@app.post("/search/semantic")
async def search_semantic(req: SearchRequest):
    """Semantic search across compacted conversations"""
    results = await semantic_search(req.query, req.limit, req.min_similarity)
    return {
        "query": req.query,
        "results": results,
        "count": len(results)
    }

@app.post("/search/keyword")
async def search_keyword(req: SearchRequest):
    """Keyword search across all conversations"""
    results = keyword_search(req.query, req.limit)
    return {
        "query": req.query,
        "results": results,
        "count": len(results)
    }

@app.post("/admin/compact")
async def force_compaction(background_tasks: BackgroundTasks):
    """Force compaction cycle"""
    if server_state["is_compacting"]:
        return {"status": "already_running"}
    
    background_tasks.add_task(run_compaction_cycle)
    return {"status": "started"}

@app.post("/admin/sync")
async def force_sync(background_tasks: BackgroundTasks):
    """Force Backblaze sync"""
    if server_state["is_syncing"]:
        return {"status": "already_running"}
    
    background_tasks.add_task(sync_to_backblaze)
    return {"status": "started"}

@app.get("/admin/stats")
async def get_stats():
    """Get detailed statistics"""
    usage_gb = get_ebs_usage_gb()
    
    # Count files
    active_convs = len(list(CONVERSATIONS_DIR.iterdir()))
    compacted_convs = len(list(COMPACTED_DIR.glob("*.json.gz")))
    pending_sync = len(list(PENDING_SYNC_DIR.glob("*.json")))
    embeddings = len(list(EMBEDDINGS_DIR.glob("*.npy")))
    
    return {
        "storage": {
            "ebs_usage_gb": round(usage_gb, 2),
            "ebs_max_gb": EBS_MAX_SIZE_GB,
            "ebs_usage_percent": round(usage_gb / EBS_MAX_SIZE_GB * 100, 1),
            "will_compact_at_gb": EBS_COMPACT_TRIGGER_GB
        },
        "conversations": {
            "active": active_convs,
            "compacted": compacted_convs,
            "total": active_convs + compacted_convs
        },
        "sync": {
            "pending": pending_sync,
            "last_sync": server_state.get("last_sync"),
            "b2_enabled": B2_ENABLED
        },
        "embeddings": {
            "count": embeddings,
            "in_memory": len(embedding_index["ids"])
        },
        "compaction": {
            "last_compaction": server_state.get("last_compaction"),
            "total_compacted": server_state.get("compacted_count", 0),
            "is_running": server_state.get("is_compacting", False)
        }
    }

@app.get("/recall/{conversation_id}")
async def recall_conversation(conversation_id: str, fetch_archive: bool = False):
    """Recall a conversation - tries local first, then Backblaze"""
    # Try local first
    conv = get_conversation(conversation_id)
    if conv:
        return conv
    
    # Try Backblaze archive
    if fetch_archive:
        archived = retrieve_from_backblaze(conversation_id)
        if archived:
            return {
                "source": "backblaze_archive",
                **archived
            }
    
    raise HTTPException(404, "Conversation not found")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "memory-mcp"}

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 MEMORY MCP SERVER")
    print("=" * 70)
    print(f"""
Configuration:
  EBS Storage: {EBS_DATA_DIR}
  Max Size: {EBS_MAX_SIZE_GB} GB
  Compact Trigger: {EBS_COMPACT_TRIGGER_GB} GB
  Target After Compact: {EBS_TARGET_SIZE_GB} GB
  
Backblaze B2:
  Enabled: {B2_ENABLED}
  Bucket: {B2_BUCKET_NAME}
  
Workers:
  Sync Interval: {SYNC_INTERVAL_SECONDS}s
  Compaction Check: {COMPACTION_CHECK_INTERVAL_SECONDS}s
  Min Age for Compaction: {COMPACTION_MIN_AGE_HOURS}h
    """)
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
