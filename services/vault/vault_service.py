#!/usr/bin/env python3
"""
Origin OS Vault - Secure Token Envelope System
Implements encrypted token wrapping with metadata, expiration, and access control
"""

import os
import json
import hashlib
import hmac
import secrets
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Origin OS Vault", version="1.0")

# =============================================================================
# CONFIGURATION
# =============================================================================

VAULT_DIR = os.getenv("VAULT_DIR", "/vault")
MASTER_KEY = os.getenv("VAULT_MASTER_KEY", os.getenv("VAULT_MASTER_PASSWORD", ""))
ENVELOPE_FILE = os.path.join(VAULT_DIR, "envelopes.enc")
AUDIT_LOG = os.path.join(VAULT_DIR, "audit.log")

os.makedirs(VAULT_DIR, exist_ok=True)

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
    
    # Use a fixed salt derived from the master key for consistency
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
    PRIVATE = "private"          # Only owner can access
    SERVICE = "service"          # Specific services can access
    SHARED = "shared"            # Multiple users can access
    PUBLIC_READ = "public_read"  # Anyone can read (careful!)


@dataclass
class TokenEnvelope:
    """Secure wrapper for sensitive tokens"""
    id: str                          # Unique envelope ID
    name: str                        # Human-readable name
    token_type: TokenType            # Type of token
    encrypted_value: str             # Encrypted token value
    
    # Metadata
    service: str                     # Service this token belongs to
    environment: str                 # dev, staging, prod
    created_at: str                  # ISO timestamp
    expires_at: Optional[str]        # Optional expiration
    rotated_at: Optional[str]        # Last rotation time
    
    # Access Control
    owner: str                       # Owner identifier
    access_policy: AccessPolicy      # Access control policy
    allowed_services: List[str]      # Services allowed to access
    
    # Audit
    access_count: int                # Number of times accessed
    last_accessed_at: Optional[str]  # Last access time
    last_accessed_by: Optional[str]  # Last accessor
    
    # Security
    checksum: str                    # HMAC checksum for integrity
    version: int                     # Version number for rotation tracking
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TokenEnvelope':
        data['token_type'] = TokenType(data['token_type'])
        data['access_policy'] = AccessPolicy(data['access_policy'])
        return cls(**data)


# =============================================================================
# ENVELOPE STORAGE
# =============================================================================

class EnvelopeStore:
    """Encrypted storage for token envelopes"""
    
    def __init__(self):
        self.envelopes: Dict[str, TokenEnvelope] = {}
        self._load()
    
    def _load(self):
        """Load envelopes from encrypted file"""
        if os.path.exists(ENVELOPE_FILE):
            try:
                with open(ENVELOPE_FILE, 'r') as f:
                    encrypted = f.read()
                decrypted = decrypt_data(encrypted)
                data = json.loads(decrypted)
                self.envelopes = {
                    k: TokenEnvelope.from_dict(v) 
                    for k, v in data.items()
                }
            except Exception as e:
                print(f"Warning: Could not load envelopes: {e}")
                self.envelopes = {}
    
    def _save(self):
        """Save envelopes to encrypted file"""
        data = {k: v.to_dict() for k, v in self.envelopes.items()}
        encrypted = encrypt_data(json.dumps(data))
        with open(ENVELOPE_FILE, 'w') as f:
            f.write(encrypted)
    
    def _audit(self, action: str, envelope_id: str, accessor: str, details: str = ""):
        """Log audit event"""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = f"{timestamp} | {action} | {envelope_id} | {accessor} | {details}\n"
        with open(AUDIT_LOG, 'a') as f:
            f.write(log_entry)
    
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
        
        # Encrypt the token value
        encrypted_value = encrypt_data(value)
        
        # Compute integrity checksum
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
        """Get envelope metadata (not the decrypted value)"""
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
        
        # Only owner can rotate
        if rotator != envelope.owner and rotator != "system":
            self._audit("ROTATE_DENIED", envelope_id, rotator, "not_owner")
            raise HTTPException(403, "Only owner can rotate tokens")
        
        # Update envelope
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
    """Get accessor identity from header"""
    return x_accessor


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "Origin OS Vault",
        "version": "1.0",
        "features": [
            "Token Envelopes - Secure token wrapping",
            "Encryption - AES-256 via Fernet",
            "Access Control - Owner, Service, Shared policies",
            "Expiration - Auto-expire tokens",
            "Rotation - Version-tracked token rotation",
            "Audit Log - Full access audit trail",
            "Integrity - HMAC checksums"
        ],
        "endpoints": {
            "create": "POST /envelopes",
            "list": "GET /envelopes",
            "get": "GET /envelopes/{id}",
            "unwrap": "POST /envelopes/{id}/unwrap",
            "rotate": "POST /envelopes/{id}/rotate",
            "delete": "DELETE /envelopes/{id}",
            "find": "GET /envelopes/find/{name}",
            "audit": "GET /audit"
        }
    }


@app.post("/envelopes", response_model=EnvelopeResponse)
async def create_envelope(req: CreateEnvelopeRequest, accessor: str = Depends(get_accessor)):
    """Create a new token envelope"""
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
    """List token envelopes"""
    envelopes = store.list(owner=owner, service=service, environment=environment)
    
    return [
        EnvelopeResponse(
            id=e.id,
            name=e.name,
            token_type=e.token_type,
            service=e.service,
            environment=e.environment,
            created_at=e.created_at,
            expires_at=e.expires_at,
            rotated_at=e.rotated_at,
            owner=e.owner,
            access_policy=e.access_policy,
            allowed_services=e.allowed_services,
            access_count=e.access_count,
            last_accessed_at=e.last_accessed_at,
            version=e.version
        )
        for e in envelopes
    ]


@app.get("/envelopes/find/{name}", response_model=EnvelopeResponse)
async def find_envelope(name: str, accessor: str = Depends(get_accessor)):
    """Find envelope by name"""
    envelope = store.find_by_name(name)
    if not envelope:
        raise HTTPException(404, f"Envelope '{name}' not found")
    
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


@app.get("/envelopes/{envelope_id}", response_model=EnvelopeResponse)
async def get_envelope(envelope_id: str, accessor: str = Depends(get_accessor)):
    """Get envelope metadata"""
    envelope = store.get(envelope_id, accessor)
    if not envelope:
        raise HTTPException(404, f"Envelope {envelope_id} not found")
    
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


@app.post("/envelopes/{envelope_id}/unwrap", response_model=UnwrapResponse)
async def unwrap_envelope(envelope_id: str, accessor: str = Depends(get_accessor)):
    """Unwrap (decrypt) a token envelope"""
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
    """Rotate a token to a new value"""
    envelope = store.rotate(envelope_id, req.new_value, accessor)
    
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


@app.delete("/envelopes/{envelope_id}")
async def delete_envelope(envelope_id: str, accessor: str = Depends(get_accessor)):
    """Delete an envelope"""
    if store.delete(envelope_id, accessor):
        return {"deleted": envelope_id}
    raise HTTPException(404, f"Envelope {envelope_id} not found")


@app.get("/audit")
async def get_audit_log(lines: int = 100, accessor: str = Depends(get_accessor)):
    """Get recent audit log entries"""
    if not os.path.exists(AUDIT_LOG):
        return {"entries": []}
    
    with open(AUDIT_LOG, 'r') as f:
        all_lines = f.readlines()
    
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
        "envelopes_count": len(store.envelopes)
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔐 ORIGIN OS VAULT")
    print("=" * 60)
    print(f"\nVault Directory: {VAULT_DIR}")
    print(f"Envelopes Loaded: {len(store.envelopes)}")
    print("\nFeatures:")
    print("  • Token Envelopes with AES-256 encryption")
    print("  • Access policies: Private, Service, Shared")
    print("  • Token expiration and rotation")
    print("  • Full audit logging")
    print("  • Integrity verification (HMAC)")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
