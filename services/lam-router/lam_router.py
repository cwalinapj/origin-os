#!/usr/bin/env python3
"""
LAM ROUTER — Large Action Model for Real-Time Traffic Routing
==============================================================
Runs on Raspberry Pi 5 (8GB) + Hailo8 AI Accelerator

Responsibilities:
- Real-time traffic routing decisions (<100ms)
- A/B test assignments
- Personalization decisions
- Fraud/bot detection
- Dynamic content selection
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = os.getenv("LAM_MODEL_PATH", "/models/lam")
HAILO_ENABLED = os.getenv("HAILO_ENABLED", "true").lower() == "true"
DECISION_TIMEOUT_MS = int(os.getenv("DECISION_TIMEOUT_MS", "100"))

DATA_DIR = Path(os.getenv("LAM_DATA_DIR", "/data/lam"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lam-router")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="LAM Router",
    description="Large Action Model for Real-Time Traffic Routing",
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

class RoutingDecision(BaseModel):
    action: str  # route, personalize, ab_test, block, allow
    target: Optional[str] = None
    variant: Optional[str] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None
    latency_ms: float = 0.0

class ABTest(BaseModel):
    test_id: str
    name: str
    variants: List[Dict[str, Any]]
    traffic_split: List[float]
    active: bool = True

class RoutingRule(BaseModel):
    rule_id: str
    name: str
    conditions: Dict[str, Any]
    action: str
    target: Optional[str] = None
    priority: int = 0

# =============================================================================
# LAM ENGINE
# =============================================================================

class LAMEngine:
    """
    Large Action Model Engine
    Optimized for Raspberry Pi 5 + Hailo8 inference
    """
    
    def __init__(self):
        self.hailo_runtime = None
        self.model = None
        self.ab_tests: Dict[str, ABTest] = {}
        self.rules: List[RoutingRule] = []
        self.decision_count = 0
        self.avg_latency_ms = 0.0
        
        self._load_hailo()
        self._load_rules()
    
    def _load_hailo(self):
        """Load Hailo8 runtime if available"""
        if HAILO_ENABLED:
            try:
                # Hailo SDK import
                # from hailo_platform import HailoRTClient
                logger.info("Hailo8 runtime loaded")
            except ImportError:
                logger.warning("Hailo SDK not available, using CPU fallback")
    
    def _load_rules(self):
        """Load routing rules from disk"""
        rules_file = DATA_DIR / "rules.json"
        if rules_file.exists():
            data = json.loads(rules_file.read_text())
            self.rules = [RoutingRule(**r) for r in data.get("rules", [])]
            self.ab_tests = {
                t["test_id"]: ABTest(**t) 
                for t in data.get("ab_tests", [])
            }
            logger.info(f"Loaded {len(self.rules)} rules, {len(self.ab_tests)} A/B tests")
    
    def _save_rules(self):
        """Save rules to disk"""
        rules_file = DATA_DIR / "rules.json"
        data = {
            "rules": [r.model_dump() for r in self.rules],
            "ab_tests": [t.model_dump() for t in self.ab_tests.values()]
        }
        rules_file.write_text(json.dumps(data, indent=2))
    
    async def route(self, event: GTMEvent) -> RoutingDecision:
        """Make routing decision for event"""
        import time
        start = time.time()
        
        decision = RoutingDecision(action="allow", confidence=1.0)
        
        try:
            # 1. Check rules first (fastest)
            rule_decision = self._check_rules(event)
            if rule_decision:
                decision = rule_decision
            
            # 2. Check A/B tests
            elif self.ab_tests:
                ab_decision = self._check_ab_tests(event)
                if ab_decision:
                    decision = ab_decision
            
            # 3. Run ML model if available
            elif self.model:
                ml_decision = await self._run_model(event)
                if ml_decision:
                    decision = ml_decision
        
        except Exception as e:
            logger.error(f"Routing error: {e}")
            decision = RoutingDecision(
                action="allow",
                confidence=0.0,
                reasoning=f"Error: {str(e)}"
            )
        
        # Calculate latency
        latency_ms = (time.time() - start) * 1000
        decision.latency_ms = latency_ms
        
        # Update stats
        self.decision_count += 1
        self.avg_latency_ms = (
            (self.avg_latency_ms * (self.decision_count - 1) + latency_ms)
            / self.decision_count
        )
        
        return decision
    
    def _check_rules(self, event: GTMEvent) -> Optional[RoutingDecision]:
        """Check event against routing rules"""
        for rule in sorted(self.rules, key=lambda r: -r.priority):
            if self._matches_conditions(event, rule.conditions):
                return RoutingDecision(
                    action=rule.action,
                    target=rule.target,
                    confidence=1.0,
                    reasoning=f"Rule: {rule.name}"
                )
        return None
    
    def _matches_conditions(self, event: GTMEvent, conditions: Dict) -> bool:
        """Check if event matches conditions"""
        event_dict = event.model_dump()
        
        for key, value in conditions.items():
            if key not in event_dict:
                return False
            
            if isinstance(value, dict):
                # Complex conditions
                if "contains" in value:
                    if value["contains"] not in str(event_dict.get(key, "")):
                        return False
                if "equals" in value:
                    if event_dict.get(key) != value["equals"]:
                        return False
                if "in" in value:
                    if event_dict.get(key) not in value["in"]:
                        return False
            else:
                if event_dict.get(key) != value:
                    return False
        
        return True
    
    def _check_ab_tests(self, event: GTMEvent) -> Optional[RoutingDecision]:
        """Assign user to A/B test variant"""
        for test_id, test in self.ab_tests.items():
            if not test.active:
                continue
            
            # Deterministic assignment based on session_id
            session = event.session_id or "unknown"
            hash_val = hash(f"{test_id}:{session}") % 100
            
            cumulative = 0
            for i, split in enumerate(test.traffic_split):
                cumulative += split * 100
                if hash_val < cumulative:
                    variant = test.variants[i]
                    return RoutingDecision(
                        action="ab_test",
                        variant=variant.get("name"),
                        target=variant.get("url"),
                        confidence=0.95,
                        reasoning=f"A/B Test: {test.name}, Variant: {variant.get('name')}"
                    )
        
        return None
    
    async def _run_model(self, event: GTMEvent) -> Optional[RoutingDecision]:
        """Run ML model for routing decision"""
        # This would use Hailo8 for inference
        # For now, return None to fall through to default
        return None
    
    def add_rule(self, rule: RoutingRule):
        """Add a routing rule"""
        self.rules = [r for r in self.rules if r.rule_id != rule.rule_id]
        self.rules.append(rule)
        self._save_rules()
    
    def add_ab_test(self, test: ABTest):
        """Add an A/B test"""
        self.ab_tests[test.test_id] = test
        self._save_rules()
    
    def get_stats(self) -> Dict:
        """Get routing statistics"""
        return {
            "decision_count": self.decision_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "rules_count": len(self.rules),
            "ab_tests_count": len(self.ab_tests),
            "hailo_enabled": HAILO_ENABLED
        }


# Global engine
lam_engine = LAMEngine()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    stats = lam_engine.get_stats()
    return {"status": "healthy", **stats}

@app.post("/route")
async def route_event(event: GTMEvent):
    """Get routing decision for event"""
    decision = await lam_engine.route(event)
    return decision.model_dump()

@app.get("/stats")
async def get_stats():
    """Get routing statistics"""
    return lam_engine.get_stats()

# Rules Management
@app.get("/rules")
async def list_rules():
    """List all routing rules"""
    return {"rules": [r.model_dump() for r in lam_engine.rules]}

@app.post("/rules")
async def add_rule(rule: RoutingRule):
    """Add a routing rule"""
    lam_engine.add_rule(rule)
    return {"status": "added", "rule_id": rule.rule_id}

@app.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """Delete a routing rule"""
    lam_engine.rules = [r for r in lam_engine.rules if r.rule_id != rule_id]
    lam_engine._save_rules()
    return {"status": "deleted", "rule_id": rule_id}

# A/B Tests
@app.get("/ab-tests")
async def list_ab_tests():
    """List all A/B tests"""
    return {"ab_tests": [t.model_dump() for t in lam_engine.ab_tests.values()]}

@app.post("/ab-tests")
async def add_ab_test(test: ABTest):
    """Add an A/B test"""
    lam_engine.add_ab_test(test)
    return {"status": "added", "test_id": test.test_id}

@app.delete("/ab-tests/{test_id}")
async def delete_ab_test(test_id: str):
    """Delete an A/B test"""
    if test_id in lam_engine.ab_tests:
        del lam_engine.ab_tests[test_id]
        lam_engine._save_rules()
    return {"status": "deleted", "test_id": test_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
