#!/usr/bin/env python3
"""
Origin OS Codex - GTM Manager API
JWT-authenticated endpoints for tag management
"""

import os
import sys
from typing import Dict, List, Optional
from functools import wraps

from jwt_auth import verify_service_token, extract_token

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Origin OS Codex")
security = HTTPBearer(auto_error=False)

# JWT settings
JWT_SECRET = os.getenv("JWT_SECRET", "origin-os-default-secret-change-me")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"


# Auth dependency
async def verify_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[Dict]:
    """Verify JWT token from request"""
    
    # Skip auth if disabled (dev mode)
    if not REQUIRE_AUTH:
        return {"service": "anonymous", "scopes": ["*"]}
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    token = credentials.credentials
    payload = verify_service_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload


def require_scope(scope: str):
    """Decorator to require a specific scope"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, auth: Dict = None, **kwargs):
            if auth:
                scopes = auth.get("scopes", [])
                if "*" not in scopes and scope not in scopes:
                    raise HTTPException(status_code=403, detail=f"Missing scope: {scope}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Request models
class TagRequest(BaseModel):
    container: str
    config: Dict


class TriggerRequest(BaseModel):
    container: str
    config: Dict


# GTM Client (placeholder - same as before)
class GTMClient:
    def __init__(self):
        self.connected = False
        self._init_client()
    
    def _init_client(self):
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            
            sa_email = os.getenv("GOOGLE_SERVICE_ACCOUNT_EMAIL")
            sa_key = os.getenv("GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY")
            
            if sa_email and sa_key:
                sa_key = sa_key.replace('\\n', '\n')
                info = {
                    "type": "service_account",
                    "client_email": sa_email,
                    "private_key": sa_key,
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "project_id": sa_email.split('@')[1].split('.')[0]
                }
                credentials = service_account.Credentials.from_service_account_info(
                    info,
                    scopes=[
                        'https://www.googleapis.com/auth/tagmanager.edit.containers',
                        'https://www.googleapis.com/auth/tagmanager.manage.accounts',
                        'https://www.googleapis.com/auth/tagmanager.publish'
                    ]
                )
                self._service = build('tagmanager', 'v2', credentials=credentials)
                self.connected = True
                print("✅ GTM API connected")
        except Exception as e:
            print(f"⚠️ GTM not connected: {e}")
            self._service = None
    
    def list_containers(self) -> List[Dict]:
        return []
    
    def list_tags(self, container: str) -> List[Dict]:
        return []
    
    def create_tag(self, container: str, config: Dict) -> Dict:
        return {"success": False, "error": "GTM not configured"}
    
    def create_trigger(self, container: str, config: Dict) -> Dict:
        return {"success": False, "error": "GTM not configured"}
    
    def publish(self, container: str) -> Dict:
        return {"success": False, "error": "GTM not configured"}


gtm = GTMClient()


# Endpoints with JWT auth
@app.get("/status")
async def status(auth: Dict = Depends(verify_token)):
    """Get Codex status - requires valid JWT"""
    return {
        "service": "codex",
        "gtm_connected": gtm.connected,
        "auth": {
            "service": auth.get("service"),
            "scopes": auth.get("scopes")
        },
        "containers": []
    }


@app.get("/containers")
async def list_containers(auth: Dict = Depends(verify_token)):
    """List containers - requires gtm:read scope"""
    scopes = auth.get("scopes", [])
    if "*" not in scopes and "gtm:read" not in scopes:
        raise HTTPException(status_code=403, detail="Missing scope: gtm:read")
    
    return {"containers": gtm.list_containers()}


@app.get("/tags/{container}")
async def list_tags(container: str, auth: Dict = Depends(verify_token)):
    """List tags - requires gtm:read scope"""
    scopes = auth.get("scopes", [])
    if "*" not in scopes and "gtm:read" not in scopes:
        raise HTTPException(status_code=403, detail="Missing scope: gtm:read")
    
    return {"tags": gtm.list_tags(container)}


@app.post("/tag")
async def create_tag(request: TagRequest, auth: Dict = Depends(verify_token)):
    """Create tag - requires gtm:write scope"""
    scopes = auth.get("scopes", [])
    if "*" not in scopes and "gtm:write" not in scopes:
        raise HTTPException(status_code=403, detail="Missing scope: gtm:write")
    
    print(f"🏷️ Create tag request from {auth.get('service')}: {request.config.get('name')}")
    return gtm.create_tag(request.container, request.config)


@app.post("/trigger")
async def create_trigger(request: TriggerRequest, auth: Dict = Depends(verify_token)):
    """Create trigger - requires gtm:write scope"""
    scopes = auth.get("scopes", [])
    if "*" not in scopes and "gtm:write" not in scopes:
        raise HTTPException(status_code=403, detail="Missing scope: gtm:write")
    
    print(f"⚡ Create trigger request from {auth.get('service')}: {request.config.get('name')}")
    return gtm.create_trigger(request.container, request.config)


@app.post("/publish/{container}")
async def publish_container(container: str, auth: Dict = Depends(verify_token)):
    """Publish container - requires gtm:publish scope"""
    scopes = auth.get("scopes", [])
    if "*" not in scopes and "gtm:publish" not in scopes:
        raise HTTPException(status_code=403, detail="Missing scope: gtm:publish")
    
    print(f"🚀 Publish request from {auth.get('service')}: {container}")
    return gtm.publish(container)


@app.get("/health")
async def health():
    """Health check - no auth required"""
    return {"status": "healthy", "service": "codex", "jwt_enabled": REQUIRE_AUTH}


if __name__ == "__main__":
    print("=" * 50)
    print("🏷️  ORIGIN OS CODEX")
    print("=" * 50)
    print(f"JWT Auth: {'Enabled' if REQUIRE_AUTH else 'Disabled'}")
    print(f"GTM Connected: {gtm.connected}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
