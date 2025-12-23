#!/usr/bin/env python3
"""
Origin OS Codex - GTM Manager API
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import uvicorn

api = FastAPI(title="Origin OS Codex")

class TagRequest(BaseModel):
    container: str
    config: Dict

@api.get("/status")
async def status():
    return {
        "service": "codex",
        "gtm_connected": False,
        "accounts": [],
        "containers": []
    }

@api.get("/containers")
async def list_containers():
    return {"containers": []}

@api.post("/tag")
async def create_tag(r: TagRequest):
    return {"success": False, "error": "GTM not configured yet"}

@api.post("/trigger")
async def create_trigger(r: TagRequest):
    return {"success": False, "error": "GTM not configured yet"}

@api.get("/health")
async def health():
    return {"status": "healthy", "service": "codex"}

if __name__ == "__main__":
    print("=" * 50)
    print("🏷️  ORIGIN OS CODEX")
    print("=" * 50)
    print("Codex online — GTM API ready")
    uvicorn.run(api, host="0.0.0.0", port=8000)
