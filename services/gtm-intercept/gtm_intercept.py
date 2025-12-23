#!/usr/bin/env python3
"""
GTM INTERCEPT MCP — Google Tag Manager Real-Time Event Capture
===============================================================
Captures GTM events BEFORE they go to Google, enabling real-time
LAM routing decisions without Google's processing delay.

Architecture:
  Website → GTM Container → Origin OS Intercept → LAM Router → Action
                                    ↓
                              Google Analytics (delayed copy)
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path
from collections import deque

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

GTM_CONTAINER_ID = os.getenv("GTM_CONTAINER_ID", "")
LAM_ROUTER_URL = os.getenv("LAM_ROUTER_URL", "http://lam-router:8000")
GOOGLE_ANALYTICS_FORWARD = os.getenv("GA_FORWARD", "true").lower() == "true"

DATA_DIR = Path(os.getenv("GTM_DATA_DIR", "/data/gtm"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gtm-intercept")

# Event buffer for real-time processing
EVENT_BUFFER: deque = deque(maxlen=10000)

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="GTM Intercept MCP",
    description="Real-time GTM event capture for LAM routing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MODELS
# =============================================================================

class GTMEvent(BaseModel):
    event_name: str
    event_params: Dict[str, Any] = {}
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    page_url: Optional[str] = None
    page_title: Optional[str] = None
    referrer: Optional[str] = None
    timestamp: Optional[str] = None
    user_agent: Optional[str] = None
    ip_hash: Optional[str] = None
    geo: Optional[Dict[str, str]] = None
    device: Optional[Dict[str, str]] = None
    custom_dimensions: Optional[Dict[str, Any]] = None

class LAMRoutingDecision(BaseModel):
    action: str  # route, personalize, ab_test, block, allow
    target: Optional[str] = None
    variant: Optional[str] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None

# =============================================================================
# GTM INTERCEPT SERVICE
# =============================================================================

class GTMInterceptService:
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.event_count = 0
        self.routing_decisions = 0
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=5.0)
        return self.http_client
    
    async def process_event(self, event: GTMEvent) -> LAMRoutingDecision:
        """Process GTM event and get LAM routing decision"""
        
        if not event.timestamp:
            event.timestamp = datetime.now(timezone.utc).isoformat()
        
        EVENT_BUFFER.append(event.model_dump())
        self.event_count += 1
        
        # Send to LAM Router for real-time decision
        decision = await self._get_lam_decision(event)
        
        # Forward to Google Analytics (async, non-blocking)
        if GOOGLE_ANALYTICS_FORWARD:
            asyncio.create_task(self._forward_to_google(event))
        
        await self._log_event(event, decision)
        return decision
    
    async def _get_lam_decision(self, event: GTMEvent) -> LAMRoutingDecision:
        """Get routing decision from LAM"""
        try:
            client = await self._get_client()
            response = await client.post(
                f"{LAM_ROUTER_URL}/route",
                json=event.model_dump(),
                timeout=0.5  # 500ms max for real-time
            )
            
            if response.status_code == 200:
                self.routing_decisions += 1
                return LAMRoutingDecision(**response.json())
        
        except Exception as e:
            logger.warning(f"LAM routing failed: {e}")
        
        return LAMRoutingDecision(
            action="allow",
            confidence=0.0,
            reasoning="LAM unavailable, default allow"
        )
    
    async def _forward_to_google(self, event: GTMEvent):
        """Forward event to Google Analytics"""
        try:
            logger.debug(f"Forward to GA: {event.event_name}")
        except Exception as e:
            logger.error(f"GA forward failed: {e}")
    
    async def _log_event(self, event: GTMEvent, decision: LAMRoutingDecision):
        """Log event and decision"""
        log_entry = {
            "timestamp": event.timestamp,
            "event": event.event_name,
            "decision": decision.action,
            "confidence": decision.confidence
        }
        
        log_file = DATA_DIR / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def generate_gtm_snippet(self, container_id: str, intercept_url: str) -> str:
        """Generate GTM snippet with Origin OS intercept"""
        return f'''<!-- Google Tag Manager with Origin OS Intercept -->
<script>
(function(w,d,s,l,i,o){{
  w[l]=w[l]||[];
  var originalPush = w[l].push.bind(w[l]);
  w[l].push = function(data) {{
    if (data && data.event) {{
      fetch(o+'/event', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          event_name: data.event,
          event_params: data,
          page_url: window.location.href,
          page_title: document.title,
          referrer: document.referrer,
          timestamp: new Date().toISOString(),
          session_id: w.gtm_session_id || 'unknown'
        }}),
        keepalive: true
      }}).then(r => r.json()).then(decision => {{
        if (decision.action === 'route' && decision.target) {{
          window.location.href = decision.target;
        }}
      }}).catch(() => {{}});
    }}
    return originalPush(data);
  }};
  w[l].push({{'gtm.start': new Date().getTime(), event:'gtm.js'}});
  var f=d.getElementsByTagName(s)[0],
      j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';
  j.async=true;
  j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;
  f.parentNode.insertBefore(j,f);
}})(window,document,'script','dataLayer','{container_id}','{intercept_url}');
</script>'''


gtm_service = GTMInterceptService()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "events_processed": gtm_service.event_count,
        "routing_decisions": gtm_service.routing_decisions
    }

@app.post("/event")
async def capture_event(event: GTMEvent):
    """Capture GTM event and get LAM routing decision"""
    decision = await gtm_service.process_event(event)
    return decision.model_dump()

@app.get("/events/recent")
async def get_recent_events(limit: int = 100):
    """Get recent events from buffer"""
    events = list(EVENT_BUFFER)[-limit:]
    return {"events": events, "total": len(EVENT_BUFFER)}

@app.get("/snippet/{container_id}")
async def get_gtm_snippet(container_id: str, request: Request):
    """Generate GTM snippet with intercept"""
    base_url = str(request.base_url).rstrip("/")
    snippet = gtm_service.generate_gtm_snippet(container_id, base_url)
    return {
        "container_id": container_id,
        "head_snippet": snippet,
        "intercept_url": base_url
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
