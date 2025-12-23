#!/usr/bin/env python3
"""
Origin OS - JWT Auth Module
Shared JWT signing/verification for inter-service communication
"""

import os
import time
import hmac
import hashlib
import base64
import json
from typing import Optional, Dict

# JWT Secret - shared between services via environment
JWT_SECRET = os.getenv("JWT_SECRET", "origin-os-default-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 300  # 5 minutes


def base64url_encode(data: bytes) -> str:
    """Base64 URL-safe encoding without padding"""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def base64url_decode(data: str) -> bytes:
    """Base64 URL-safe decoding with padding fix"""
    padding = 4 - len(data) % 4
    if padding != 4:
        data += '=' * padding
    return base64.urlsafe_b64decode(data)


def create_jwt(payload: Dict, secret: str = None, expiry: int = None) -> str:
    """
    Create a signed JWT token
    
    Args:
        payload: Data to include in token
        secret: Signing secret (defaults to JWT_SECRET env var)
        expiry: Expiry time in seconds (defaults to JWT_EXPIRY_SECONDS)
    
    Returns:
        Signed JWT string
    """
    secret = secret or JWT_SECRET
    expiry = expiry or JWT_EXPIRY_SECONDS
    
    # Header
    header = {
        "alg": JWT_ALGORITHM,
        "typ": "JWT"
    }
    
    # Add standard claims
    now = int(time.time())
    payload = {
        **payload,
        "iat": now,           # Issued at
        "exp": now + expiry,  # Expiry
        "iss": "origin-os"    # Issuer
    }
    
    # Encode header and payload
    header_b64 = base64url_encode(json.dumps(header).encode())
    payload_b64 = base64url_encode(json.dumps(payload).encode())
    
    # Create signature
    message = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).digest()
    signature_b64 = base64url_encode(signature)
    
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def verify_jwt(token: str, secret: str = None) -> Optional[Dict]:
    """
    Verify and decode a JWT token
    
    Args:
        token: JWT string to verify
        secret: Signing secret (defaults to JWT_SECRET env var)
    
    Returns:
        Decoded payload dict if valid, None if invalid
    """
    secret = secret or JWT_SECRET
    
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        
        header_b64, payload_b64, signature_b64 = parts
        
        # Verify signature
        message = f"{header_b64}.{payload_b64}"
        expected_signature = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        actual_signature = base64url_decode(signature_b64)
        
        if not hmac.compare_digest(expected_signature, actual_signature):
            return None
        
        # Decode payload
        payload = json.loads(base64url_decode(payload_b64))
        
        # Check expiry
        if payload.get("exp", 0) < int(time.time()):
            return None
        
        return payload
        
    except Exception as e:
        print(f"JWT verification error: {e}")
        return None


def create_service_token(service_name: str, scopes: list = None) -> str:
    """
    Create a JWT for inter-service communication
    
    Args:
        service_name: Name of the calling service (e.g., "mcp")
        scopes: List of permissions (e.g., ["gtm:read", "gtm:write"])
    
    Returns:
        Signed JWT string
    """
    return create_jwt({
        "service": service_name,
        "scopes": scopes or ["*"],
        "type": "service"
    })


def verify_service_token(token: str, required_scope: str = None) -> Optional[Dict]:
    """
    Verify a service token and optionally check scope
    
    Args:
        token: JWT string
        required_scope: Scope that must be present (e.g., "gtm:write")
    
    Returns:
        Decoded payload if valid and authorized, None otherwise
    """
    payload = verify_jwt(token)
    
    if not payload:
        return None
    
    if payload.get("type") != "service":
        return None
    
    if required_scope:
        scopes = payload.get("scopes", [])
        if "*" not in scopes and required_scope not in scopes:
            return None
    
    return payload


# FastAPI middleware helper
def get_auth_header(token: str) -> Dict[str, str]:
    """Get Authorization header for requests"""
    return {"Authorization": f"Bearer {token}"}


def extract_token(auth_header: str) -> Optional[str]:
    """Extract token from Authorization header"""
    if not auth_header:
        return None
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return auth_header
