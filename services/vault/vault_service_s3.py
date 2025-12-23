#!/usr/bin/env python3
"""
Origin OS Vault - S3-Backed Secure Token Envelope System
=========================================================
All data stored in Backblaze B2 S3 - fully portable containers.
"""

import os
import sys
import json
import hashlib
import hmac
import secrets
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import boto3
from botocore.config import Config
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Origin OS Vault", version="2.0 - S3 Backend")

# =============================================================================
# CONFIGURATION
# =============================================================================

MASTER_KEY = os.getenv("VAULT_MASTER_KEY", os.getenv("VAULT_MASTER_PASSWORD", ""))

# S3/Backblaze Configuration
B2_ENDPOINT = os.getenv("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET = os.getenv("B2_BUCKET", "origin-os-data")
S3_PREFIX = "vault"

# Local cache for performance
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/vault-cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# S3 STORAGE
# =============================================================================

class S3Backend:
    """S3-compatible storage backend for Backblaze B2"""
    
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=B2_ENDPOINT,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APP_KEY,
            config=Config(signature_version="s3v4", retries={"max_attempts": 3})
        )
        self.bucket = B2_BUCKET
        self.prefix = S3_PREFIX
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Create bucket if needed"""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except:
            try:
                self.client.create_bucket(Bucket=self.bucket)
                print(f"Created bucket: {self.bucket}")
            except Exception as e:
                print(f"Bucket check/create warning: {e}")
    
    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}"
    
    def put(self, name: str, data: str) -> bool:
        """Store data in S3"""
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self._key(name),
                Body=data.encode("utf-8")
            )
            # Update local cache
            cache_path = os.path.join(CACHE_DIR, name)
            with open(cache_path, "w") as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"S3 put error: {e}")
            return False
    
    def get(self, name: str) -> Optional[str]:
        """Get data from S3 (with local cache)"""
        cache_path = os.path.join(CACHE_DIR, name)
        
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(name))
            data = response["Body"].read().decode("utf-8")
            # Update cache
            with open(cache_path, "w") as f:
                f.write(data)
            return data
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            # Try cache on error
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    return f.read()
            print(f"S3 get error: {e}")
            return None
    
    def delete(self, name: str) -> bool:
        """Delete from S3"""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self._key(name))
            cache_path = os.path.join(CACHE_DIR, name)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return True
        except Exception as e:
            print(f"S3 delete error: {e}")
            return False
    
    def append_log(self, name: str, entry: str) -> bool:
        """Append to a log file in S3"""
        existing = self.get(name) or ""
        return self.put(name, existing + entry)


# Initialize S3 backend
s3 = S3Backend()

# =============================================================================
# ENCRYPTION UTILITIES
# =============================================================================

def derive_key(master_password: str, salt: bytes = None) -> tuple[bytes, bytes]:
    """Derive encryption key from master password using PBKDF2"""
    if salt is None:
        salt = secrets.token_bytes(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
    return key, salt


def get_fernet() -> Fernet:
    """Get Fernet cipher initialized with master key"""
    if not MASTER_KEY:
        raise HTTPException(500, "VAULT_MASTER_KEY not configured")
    
    salt = hashlib.sha256(MASTER_KEY.encode()).digest()[:16]
    key, _ = derive_key(MASTER_KEY, salt)
    return Fernet(key)


def encrypt_data(data: str) -> str:
    """Encrypt data and return base64 encoded ciphertext"""
    f = get_fernet()
    return f.encrypt(data.encode()).decode()


def decrypt_data(encrypted: str) -> str:
    """Decrypt base64 encoded ciphertext"""
    f = get_fernet()
    return f.decrypt(encrypted.encode()).decode()


# =============================================================================
# TOKEN ENVELOPE MODELS
# =============================================================================

class TokenType(str, Enum):
    API_KEY = "api_key"
    OAUTH_TOKEN = "oauth_token"
    REFRESH_TOKEN = "refresh_token"
    JWT = "jwt"
    PASSWORD = "password"
    SECRET = "secret"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    WEBHOOK_SECRET = "webhook_secret"
    ENCRYPTION_KEY = "encryption_key"


class AccessPolicy(str, Enum):
    PRIVATE = "private"
    SERVICE = "service"
    SHARED = "shared"
    PUBLIC_READ = "public_read"


@dataclass
class TokenEnvelope:
    """Secure wrapper for sensitive tokens"""
    id: str
    name: str
    token_type: TokenType
    encrypted_value: str
    service: str
    environment: str
    created_at: str
    expires_at: Optional[str]
    rotated_at: Optional[str]
    owner: str
    access_policy: AccessPolicy
    allowed_services: List[str]
    access_count: int
    last_accessed_at: Optional[str]
    last_accessed_by: Optional[str]
    checksum: str
    version: int
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TokenEnvelope':
        data['token_type'] = TokenType(data['token_type'])
        data['access_policy'] = AccessPolicy(data['access_policy'])
        return cls(**data)


# =============================================================================
# ENVELOPE STORAGE (S3-BACKED)
# =============================================================================

class EnvelopeStore:
    """S3-backed storage for token envelopes"""
    
    ENVELOPE_FILE = "envelopes.enc"
    AUDIT_FILE = "audit.log"
    
    def __init__(self):
        self.envelopes: Dict[str, TokenEnvelope] = {}
        self._load()
    
    def _load(self):
        """Load envelopes from S3"""
        try:
            encrypted = s3.get(self.ENVELOPE_FILE)
            if encrypted:
                decrypted = decrypt_data(encrypted)
                data = json.loads(decrypted)
                self.envelopes = {
                    k: TokenEnvelope.from_dict(v) 
                    for k, v in data.items()
                }
                print(f"Loaded {len(self.envelopes)} envelopes from S3")
            else:
                print("No envelopes file in S3, starting fresh")
                self.envelopes = {}
        except Exception as e:
            print(f"Warning: Could not load envelopes from S3: {e}")
            self.envelopes = {}
    
    def _save(self):
        """Save envelopes to S3"""
        data = {k: v.to_dict() for k, v in self.envelopes.items()}
        encrypted = encrypt_data(json.dumps(data))
        if not s3.put(self.ENVELOPE_FILE, encrypted):
            raise HTTPException(500, "Failed to save to S3")
    
    def _audit(self, action: str, envelope_id: str, accessor: str, details: str = ""):
        """Log audit event to S3"""
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = f"{timestamp} | {action} | {envelope_id} | {accessor} | {details}\n"
        s3.append_log(self.AUDIT_FILE, entry)
    
    def _compute_checksum(self, value: str, envelope_id: str) -> str:
        """Compute HMAC checksum for integrity verification"""
        key = hashlib.sha256(MASTER_KEY.encode()).digest()
        message = f"{envelope_id}:{value}".encode()
        return hmac.new(key, message, hashlib.sha256).hexdigest()
    
    def create(self, name: str, value: str, token_type: TokenType,
               service: str, environment: str = "prod",
               owner: str = "system", access_policy: AccessPolicy = AccessPolicy.PRIVATE,
               allowed_services: List[str] = None,
               expires_in_days: Optional[int] = None) -> TokenEnvelope:
        """Create a new token envelope"""
        
        envelope_id = f"env_{secrets.token_hex(12)}"
        now = datetime.now(timezone.utc)
        
        expires_at = None
        if expires_in_days:
            expires_at = (now + timedelta(days=expires_in_days)).isoformat()
        
        encrypted_value = encrypt_data(value)
        checksum = self._compute_checksum(value, envelope_id)
        
        envelope = TokenEnvelope(
            id=envelope_id,
            name=name,
            token_type=token_type,
            encrypted_value=encrypted_value,
            service=service,
            environment=environment,
            created_at=now.isoformat(),
            expires_at=expires_at,
            rotated_at=None,
            owner=owner,
            access_policy=access_policy,
            allowed_services=allowed_services or [],
            access_count=0,
            last_accessed_at=None,
            last_accessed_by=None,
            checksum=checksum,
            version=1
        )
        
        self.envelopes[envelope_id] = envelope
        self._save()
        self._audit("CREATE", envelope_id, owner, f"name={name}, type={token_type.value}")
        
        return envelope
    
    def get(self, envelope_id: str, accessor: str = "system") -> Optional[TokenEnvelope]:
        """Get envelope metadata"""
        envelope = self.envelopes.get(envelope_id)
        if envelope:
            self._audit("GET_META", envelope_id, accessor)
        return envelope
    
    def unwrap(self, envelope_id: str, accessor: str = "system") -> Optional[str]:
        """Unwrap (decrypt) the token value"""
        envelope = self.envelopes.get(envelope_id)
        if not envelope:
            return None
        
        # Check expiration
        if envelope.expires_at:
            expires = datetime.fromisoformat(envelope.expires_at)
            if datetime.now(timezone.utc) > expires:
                self._audit("UNWRAP_DENIED", envelope_id, accessor, "expired")
                raise HTTPException(403, f"Token envelope {envelope_id} has expired")
        
        # Check access policy
        if envelope.access_policy == AccessPolicy.PRIVATE and accessor != envelope.owner:
            self._audit("UNWRAP_DENIED", envelope_id, accessor, "private_access")
            raise HTTPException(403, "Access denied: private envelope")
        
        if envelope.access_policy == AccessPolicy.SERVICE:
            if accessor not in envelope.allowed_services and accessor != envelope.owner:
                self._audit("UNWRAP_DENIED", envelope_id, accessor, "service_not_allowed")
                raise HTTPException(403, f"Service {accessor} not allowed")
        
        # Decrypt
        try:
            value = decrypt_data(envelope.encrypted_value)
        except Exception as e:
            self._audit("UNWRAP_ERROR", envelope_id, accessor, str(e))
            raise HTTPException(500, "Failed to decrypt envelope")
        
        # Verify integrity
        expected_checksum = self._compute_checksum(value, envelope_id)
        if envelope.checksum != expected_checksum:
            self._audit("INTEGRITY_FAILURE", envelope_id, accessor, "checksum_mismatch")
            raise HTTPException(500, "Envelope integrity check failed")
        
        # Update access tracking
        now = datetime.now(timezone.utc).isoformat()
        envelope.access_count += 1
        envelope.last_accessed_at = now
        envelope.last_accessed_by = accessor
        self._save()
        
        self._audit("UNWRAP", envelope_id, accessor)
        return value
    
    def rotate(self, envelope_id: str, new_value: str, rotator: str = "system") -> TokenEnvelope:
        """Rotate a token to a new value"""
        envelope = self.envelopes.get(envelope_id)
        if not envelope:
            raise HTTPException(404, f"Envelope {envelope_id} not found")
        
        if rotator != envelope.owner and rotator != "system":
            self._audit("ROTATE_DENIED", envelope_id, rotator, "not_owner")
            raise HTTPException(403, "Only owner can rotate tokens")
        
        envelope.encrypted_value = encrypt_data(new_value)
        envelope.checksum = self._compute_checksum(new_value, envelope_id)
        envelope.rotated_at = datetime.now(timezone.utc).isoformat()
        envelope.version += 1
        
        self._save()
        self._audit("ROTATE", envelope_id, rotator, f"version={envelope.version}")
        
        return envelope
    
    def delete(self, envelope_id: str, deleter: str = "system") -> bool:
        """Delete an envelope"""
        envelope = self.envelopes.get(envelope_id)
        if not envelope:
            return False
        
        if deleter != envelope.owner and deleter != "system":
            self._audit("DELETE_DENIED", envelope_id, deleter, "not_owner")
            raise HTTPException(403, "Only owner can delete envelopes")
        
        del self.envelopes[envelope_id]
        self._save()
        self._audit("DELETE", envelope_id, deleter)
        
        return True
    
    def list(self, owner: str = None, service: str = None, 
             environment: str = None) -> List[TokenEnvelope]:
        """List envelopes with optional filters"""
        results = list(self.envelopes.values())
        
        if owner:
            results = [e for e in results if e.owner == owner]
        if service:
            results = [e for e in results if e.service == service]
        if environment:
            results = [e for e in results if e.environment == environment]
        
        return results
    
    def find_by_name(self, name: str) -> Optional[TokenEnvelope]:
        """Find envelope by name"""
        for envelope in self.envelopes.values():
            if envelope.name == name:
                return envelope
        return None


# Initialize store
store = EnvelopeStore()


# =============================================================================
# API MODELS
# =============================================================================

class CreateEnvelopeRequest(BaseModel):
    name: str
    value: str
    token_type: TokenType
    service: str
    environment: str = "prod"
    owner: str = "system"
    access_policy: AccessPolicy = AccessPolicy.PRIVATE
    allowed_services: List[str] = []
    expires_in_days: Optional[int] = None


class RotateEnvelopeRequest(BaseModel):
    new_value: str


class EnvelopeResponse(BaseModel):
    id: str
    name: str
    token_type: TokenType
    service: str
    environment: str
    created_at: str
    expires_at: Optional[str]
    rotated_at: Optional[str]
    owner: str
    access_policy: AccessPolicy
    allowed_services: List[str]
    access_count: int
    last_accessed_at: Optional[str]
    version: int


class UnwrapResponse(BaseModel):
    id: str
    name: str
    value: str
    token_type: TokenType


# =============================================================================
# AUTHENTICATION
# =============================================================================

async def get_accessor(x_accessor: str = Header(default="anonymous")) -> str:
    return x_accessor


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "Origin OS Vault",
        "version": "2.0 - S3 Backend",
        "storage": {
            "type": "Backblaze B2 S3",
            "bucket": B2_BUCKET,
            "prefix": S3_PREFIX
        },
        "envelopes_count": len(store.envelopes)
    }


@app.post("/envelopes", response_model=EnvelopeResponse)
async def create_envelope(req: CreateEnvelopeRequest, accessor: str = Depends(get_accessor)):
    envelope = store.create(
        name=req.name,
        value=req.value,
        token_type=req.token_type,
        service=req.service,
        environment=req.environment,
        owner=req.owner or accessor,
        access_policy=req.access_policy,
        allowed_services=req.allowed_services,
        expires_in_days=req.expires_in_days
    )
    
    return EnvelopeResponse(
        id=envelope.id,
        name=envelope.name,
        token_type=envelope.token_type,
        service=envelope.service,
        environment=envelope.environment,
        created_at=envelope.created_at,
        expires_at=envelope.expires_at,
        rotated_at=envelope.rotated_at,
        owner=envelope.owner,
        access_policy=envelope.access_policy,
        allowed_services=envelope.allowed_services,
        access_count=envelope.access_count,
        last_accessed_at=envelope.last_accessed_at,
        version=envelope.version
    )


@app.get("/envelopes", response_model=List[EnvelopeResponse])
async def list_envelopes(
    owner: Optional[str] = None,
    service: Optional[str] = None,
    environment: Optional[str] = None,
    accessor: str = Depends(get_accessor)
):
    envelopes = store.list(owner=owner, service=service, environment=environment)
    return [
        EnvelopeResponse(
            id=e.id, name=e.name, token_type=e.token_type, service=e.service,
            environment=e.environment, created_at=e.created_at, expires_at=e.expires_at,
            rotated_at=e.rotated_at, owner=e.owner, access_policy=e.access_policy,
            allowed_services=e.allowed_services, access_count=e.access_count,
            last_accessed_at=e.last_accessed_at, version=e.version
        )
        for e in envelopes
    ]


@app.get("/envelopes/find/{name}", response_model=EnvelopeResponse)
async def find_envelope(name: str, accessor: str = Depends(get_accessor)):
    envelope = store.find_by_name(name)
    if not envelope:
        raise HTTPException(404, f"Envelope '{name}' not found")
    
    return EnvelopeResponse(
        id=envelope.id, name=envelope.name, token_type=envelope.token_type,
        service=envelope.service, environment=envelope.environment,
        created_at=envelope.created_at, expires_at=envelope.expires_at,
        rotated_at=envelope.rotated_at, owner=envelope.owner,
        access_policy=envelope.access_policy, allowed_services=envelope.allowed_services,
        access_count=envelope.access_count, last_accessed_at=envelope.last_accessed_at,
        version=envelope.version
    )


@app.get("/envelopes/{envelope_id}", response_model=EnvelopeResponse)
async def get_envelope(envelope_id: str, accessor: str = Depends(get_accessor)):
    envelope = store.get(envelope_id, accessor)
    if not envelope:
        raise HTTPException(404, f"Envelope {envelope_id} not found")
    
    return EnvelopeResponse(
        id=envelope.id, name=envelope.name, token_type=envelope.token_type,
        service=envelope.service, environment=envelope.environment,
        created_at=envelope.created_at, expires_at=envelope.expires_at,
        rotated_at=envelope.rotated_at, owner=envelope.owner,
        access_policy=envelope.access_policy, allowed_services=envelope.allowed_services,
        access_count=envelope.access_count, last_accessed_at=envelope.last_accessed_at,
        version=envelope.version
    )


@app.post("/envelopes/{envelope_id}/unwrap", response_model=UnwrapResponse)
async def unwrap_envelope(envelope_id: str, accessor: str = Depends(get_accessor)):
    envelope = store.get(envelope_id)
    if not envelope:
        raise HTTPException(404, f"Envelope {envelope_id} not found")
    
    value = store.unwrap(envelope_id, accessor)
    
    return UnwrapResponse(
        id=envelope.id,
        name=envelope.name,
        value=value,
        token_type=envelope.token_type
    )


@app.post("/envelopes/{envelope_id}/rotate", response_model=EnvelopeResponse)
async def rotate_envelope(
    envelope_id: str, 
    req: RotateEnvelopeRequest,
    accessor: str = Depends(get_accessor)
):
    envelope = store.rotate(envelope_id, req.new_value, accessor)
    
    return EnvelopeResponse(
        id=envelope.id, name=envelope.name, token_type=envelope.token_type,
        service=envelope.service, environment=envelope.environment,
        created_at=envelope.created_at, expires_at=envelope.expires_at,
        rotated_at=envelope.rotated_at, owner=envelope.owner,
        access_policy=envelope.access_policy, allowed_services=envelope.allowed_services,
        access_count=envelope.access_count, last_accessed_at=envelope.last_accessed_at,
        version=envelope.version
    )


@app.delete("/envelopes/{envelope_id}")
async def delete_envelope(envelope_id: str, accessor: str = Depends(get_accessor)):
    if store.delete(envelope_id, accessor):
        return {"deleted": envelope_id}
    raise HTTPException(404, f"Envelope {envelope_id} not found")


@app.get("/audit")
async def get_audit_log(lines: int = 100, accessor: str = Depends(get_accessor)):
    log_content = s3.get(store.AUDIT_FILE)
    if not log_content:
        return {"entries": []}
    
    all_lines = log_content.strip().split("\n")
    recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
    
    entries = []
    for line in recent:
        parts = line.strip().split(" | ")
        if len(parts) >= 4:
            entries.append({
                "timestamp": parts[0],
                "action": parts[1],
                "envelope_id": parts[2],
                "accessor": parts[3],
                "details": parts[4] if len(parts) > 4 else ""
            })
    
    return {"entries": entries}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "vault",
        "storage": "s3",
        "bucket": B2_BUCKET,
        "envelopes_count": len(store.envelopes)
    }


@app.post("/sync/from-local")
async def sync_from_local(local_path: str = "/vault"):
    """One-time migration from local storage to S3"""
    import os
    
    if not os.path.exists(local_path):
        return {"status": "no local data"}
    
    migrated = []
    for filename in os.listdir(local_path):
        filepath = os.path.join(local_path, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                content = f.read()
            if s3.put(filename, content):
                migrated.append(filename)
    
    # Reload from S3
    store._load()
    
    return {"migrated": migrated, "envelopes": len(store.envelopes)}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔐 ORIGIN OS VAULT - S3 BACKEND")
    print("=" * 60)
    print(f"\nStorage: Backblaze B2 S3")
    print(f"Bucket: {B2_BUCKET}")
    print(f"Prefix: {S3_PREFIX}")
    print(f"Envelopes Loaded: {len(store.envelopes)}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
