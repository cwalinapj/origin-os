#!/usr/bin/env python3
"""
Origin OS Orchestrator - Workflow Coordination Engine
Manages multi-step workflows, pipelines, and cross-service orchestration
"""

import os
import json
import asyncio
import uuid
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import httpx
import uvicorn

app = FastAPI(title="Origin OS Orchestrator", version="1.0")

# =============================================================================
# CONFIGURATION
# =============================================================================

SERVICES = {
    "mcp-hub": os.getenv("MCP_HUB_URL", "http://mcp-hub:8000"),
    "codex": os.getenv("CODEX_URL", "http://codex:8000"),
    "vault": os.getenv("VAULT_URL", "http://vault:8000"),
    "cad": os.getenv("CAD_URL", "http://cad:8000"),
    "ui": os.getenv("UI_URL", "http://ui:8000"),
}

ORCHESTRATOR_DIR = os.getenv("ORCHESTRATOR_DIR", "/data/orchestrator")
os.makedirs(ORCHESTRATOR_DIR, exist_ok=True)

# =============================================================================
# ENUMS & MODELS
# =============================================================================

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TriggerType(str, Enum):
    MANUAL = "manual"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    EVENT = "event"
    DEPENDENT = "dependent"


@dataclass
class WorkflowStep:
    id: str
    name: str
    service: str                      # Target service (mcp-hub, vault, etc.)
    action: str                       # Action to perform
    params: Dict[str, Any]            # Parameters for the action
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None   # Condition expression
    on_failure: str = "fail"          # fail, skip, continue
    
    def to_dict(self):
        return asdict(self)


@dataclass 
class Workflow:
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    trigger: TriggerType = TriggerType.MANUAL
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_by: str = "system"
    context: Dict[str, Any] = field(default_factory=dict)  # Shared context between steps
    error: Optional[str] = None
    current_step: int = 0
    
    def to_dict(self):
        d = asdict(self)
        d['steps'] = [s.to_dict() if isinstance(s, WorkflowStep) else s for s in self.steps]
        return d


# =============================================================================
# WORKFLOW STORAGE
# =============================================================================

class WorkflowStore:
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.templates: Dict[str, dict] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load built-in workflow templates"""
        self.templates = {
            "deploy-gtm-tag": {
                "name": "Deploy GTM Tag",
                "description": "Create and publish a GTM tag with trigger",
                "steps": [
                    {"name": "Create Trigger", "service": "mcp-hub", "action": "execute", 
                     "params": {"server": "gtm", "tool": "gtm_create_trigger", "params": {"$trigger"}}},
                    {"name": "Create Tag", "service": "mcp-hub", "action": "execute",
                     "params": {"server": "gtm", "tool": "gtm_create_tag", "params": {"$tag"}},
                     "depends_on": ["Create Trigger"]},
                    {"name": "Publish", "service": "mcp-hub", "action": "execute",
                     "params": {"server": "gtm", "tool": "gtm_publish", "params": {"$container"}},
                     "depends_on": ["Create Tag"]}
                ]
            },
            "scrape-and-analyze": {
                "name": "Scrape and Analyze",
                "description": "Scrape website and analyze with AI",
                "steps": [
                    {"name": "Scrape URL", "service": "mcp-hub", "action": "execute",
                     "params": {"server": "firecrawl", "tool": "scrape_url", "params": {"url": "$url"}}},
                    {"name": "Analyze Content", "service": "mcp-hub", "action": "execute",
                     "params": {"server": "openrouter", "tool": "chat", "params": {
                         "model": "anthropic/claude-3.5-sonnet",
                         "messages": [{"role": "user", "content": "Analyze this: $scrape_result"}]
                     }},
                     "depends_on": ["Scrape URL"]}
                ]
            },
            "backup-secrets": {
                "name": "Backup Secrets to S3",
                "description": "Export vault secrets and upload to S3",
                "steps": [
                    {"name": "List Envelopes", "service": "vault", "action": "list_envelopes", "params": {}},
                    {"name": "Export to JSON", "service": "orchestrator", "action": "transform",
                     "params": {"template": "json_export"}, "depends_on": ["List Envelopes"]},
                    {"name": "Upload to S3", "service": "mcp-hub", "action": "execute",
                     "params": {"server": "s3", "tool": "put_object", "params": {
                         "bucket": "$bucket", "key": "backups/vault_$timestamp.json", "body": "$export"
                     }},
                     "depends_on": ["Export to JSON"]}
                ]
            },
            "seo-audit": {
                "name": "SEO Audit Pipeline",
                "description": "Full SEO audit with crawl, analysis, and report",
                "steps": [
                    {"name": "Crawl Site", "service": "mcp-hub", "action": "execute",
                     "params": {"server": "firecrawl", "tool": "crawl_site", "params": {"url": "$url", "limit": 50}}},
                    {"name": "Get SEMrush Data", "service": "mcp-hub", "action": "execute",
                     "params": {"server": "semrush", "tool": "domain_overview", "params": {"domain": "$domain"}},
                     "depends_on": []},
                    {"name": "Analyze with AI", "service": "mcp-hub", "action": "execute",
                     "params": {"server": "openrouter", "tool": "chat", "params": {
                         "model": "anthropic/claude-3.5-sonnet",
                         "messages": [{"role": "user", "content": "SEO audit for: $crawl_result $semrush_result"}]
                     }},
                     "depends_on": ["Crawl Site", "Get SEMrush Data"]},
                    {"name": "Save Report", "service": "mcp-hub", "action": "execute",
                     "params": {"server": "filesystem", "tool": "write_file", "params": {
                         "path": "reports/seo_$domain_$timestamp.md", "content": "$analysis"
                     }},
                     "depends_on": ["Analyze with AI"]}
                ]
            },
            "generate-3d-model": {
                "name": "Generate 3D Model",
                "description": "Generate STL from description using AI",
                "steps": [
                    {"name": "Generate OpenSCAD", "service": "mcp-hub", "action": "execute",
                     "params": {"server": "openrouter", "tool": "chat", "params": {
                         "model": "anthropic/claude-3.5-sonnet",
                         "messages": [{"role": "user", "content": "Generate OpenSCAD code for: $description. Only output the code, no explanation."}]
                     }}},
                    {"name": "Render STL", "service": "cad", "action": "openscad",
                     "params": {"code": "$openscad_code", "filename": "$name.stl"},
                     "depends_on": ["Generate OpenSCAD"]}
                ]
            }
        }
    
    def save(self, workflow: Workflow):
        self.workflows[workflow.id] = workflow
        # Persist to disk
        filepath = os.path.join(ORCHESTRATOR_DIR, f"{workflow.id}.json")
        with open(filepath, 'w') as f:
            json.dump(workflow.to_dict(), f, indent=2)
    
    def get(self, workflow_id: str) -> Optional[Workflow]:
        return self.workflows.get(workflow_id)
    
    def list_all(self) -> List[Workflow]:
        return list(self.workflows.values())
    
    def delete(self, workflow_id: str) -> bool:
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            filepath = os.path.join(ORCHESTRATOR_DIR, f"{workflow_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        return False


store = WorkflowStore()

# =============================================================================
# WORKFLOW ENGINE
# =============================================================================

class WorkflowEngine:
    def __init__(self):
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.event_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.websockets: List[WebSocket] = []
    
    async def broadcast(self, event: str, data: dict):
        """Broadcast event to all connected websockets"""
        message = json.dumps({"event": event, "data": data})
        for ws in self.websockets[:]:
            try:
                await ws.send_text(message)
            except:
                self.websockets.remove(ws)
    
    def resolve_params(self, params: Dict, context: Dict) -> Dict:
        """Resolve $variables in params from context"""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                resolved[key] = context.get(var_name, value)
            elif isinstance(value, dict):
                resolved[key] = self.resolve_params(value, context)
            elif isinstance(value, list):
                resolved[key] = [
                    self.resolve_params(v, context) if isinstance(v, dict) 
                    else context.get(v[1:], v) if isinstance(v, str) and v.startswith("$")
                    else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved
    
    async def execute_step(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute a single workflow step"""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now(timezone.utc).isoformat()
        
        try:
            # Resolve parameters
            resolved_params = self.resolve_params(step.params, context)
            
            # Route to appropriate service
            if step.service == "mcp-hub":
                result = await self._call_mcp_hub(step.action, resolved_params)
            elif step.service == "vault":
                result = await self._call_vault(step.action, resolved_params)
            elif step.service == "cad":
                result = await self._call_cad(step.action, resolved_params)
            elif step.service == "codex":
                result = await self._call_codex(step.action, resolved_params)
            elif step.service == "orchestrator":
                result = await self._internal_action(step.action, resolved_params, context)
            else:
                raise ValueError(f"Unknown service: {step.service}")
            
            step.status = StepStatus.COMPLETED
            step.result = result
            step.completed_at = datetime.now(timezone.utc).isoformat()
            
            return result
            
        except Exception as e:
            step.error = str(e)
            step.retry_count += 1
            
            if step.retry_count < step.max_retries:
                # Retry
                await asyncio.sleep(2 ** step.retry_count)  # Exponential backoff
                return await self.execute_step(step, context)
            
            if step.on_failure == "skip":
                step.status = StepStatus.SKIPPED
                return None
            elif step.on_failure == "continue":
                step.status = StepStatus.FAILED
                return None
            else:
                step.status = StepStatus.FAILED
                raise
    
    async def _call_mcp_hub(self, action: str, params: Dict) -> Any:
        """Call MCP Hub service"""
        async with httpx.AsyncClient() as client:
            if action == "execute":
                r = await client.post(
                    f"{SERVICES['mcp-hub']}/execute",
                    json=params,
                    timeout=300
                )
                r.raise_for_status()
                return r.json()
            else:
                raise ValueError(f"Unknown MCP Hub action: {action}")
    
    async def _call_vault(self, action: str, params: Dict) -> Any:
        """Call Vault service"""
        async with httpx.AsyncClient() as client:
            if action == "list_envelopes":
                r = await client.get(f"{SERVICES['vault']}/envelopes", timeout=30)
            elif action == "unwrap":
                r = await client.post(
                    f"{SERVICES['vault']}/envelopes/{params['id']}/unwrap",
                    headers={"X-Accessor": params.get("accessor", "orchestrator")},
                    timeout=30
                )
            elif action == "create":
                r = await client.post(
                    f"{SERVICES['vault']}/envelopes",
                    json=params,
                    timeout=30
                )
            else:
                raise ValueError(f"Unknown Vault action: {action}")
            r.raise_for_status()
            return r.json()
    
    async def _call_cad(self, action: str, params: Dict) -> Any:
        """Call CAD service"""
        async with httpx.AsyncClient() as client:
            if action == "primitive":
                r = await client.post(f"{SERVICES['cad']}/primitive", json=params, timeout=60)
            elif action == "openscad":
                r = await client.post(f"{SERVICES['cad']}/openscad", json=params, timeout=300)
            elif action == "combine":
                r = await client.post(f"{SERVICES['cad']}/combine", json=params, timeout=120)
            else:
                raise ValueError(f"Unknown CAD action: {action}")
            r.raise_for_status()
            return r.json()
    
    async def _call_codex(self, action: str, params: Dict) -> Any:
        """Call Codex service"""
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{SERVICES['codex']}/{action}",
                json=params,
                timeout=60
            )
            r.raise_for_status()
            return r.json()
    
    async def _internal_action(self, action: str, params: Dict, context: Dict) -> Any:
        """Execute internal orchestrator actions"""
        if action == "transform":
            # Transform data between steps
            template = params.get("template")
            if template == "json_export":
                return json.dumps(context, indent=2)
            return context
        elif action == "delay":
            await asyncio.sleep(params.get("seconds", 1))
            return {"delayed": params.get("seconds", 1)}
        elif action == "condition":
            # Evaluate condition
            expr = params.get("expression", "true")
            return eval(expr, {"context": context})
        elif action == "parallel":
            # Run multiple sub-workflows in parallel
            return {"parallel": "executed"}
        else:
            raise ValueError(f"Unknown orchestrator action: {action}")
    
    async def run_workflow(self, workflow: Workflow):
        """Execute a complete workflow"""
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now(timezone.utc).isoformat()
        store.save(workflow)
        
        await self.broadcast("workflow_started", {"id": workflow.id, "name": workflow.name})
        
        try:
            # Build dependency graph - map step names to IDs
            step_by_name = {s.name: s.id for s in workflow.steps}
            step_map = {s.id: s for s in workflow.steps}
            completed_steps = set()
            completed_names = set()
            
            while len(completed_steps) < len(workflow.steps):
                # Find steps that can run (dependencies satisfied)
                runnable = []
                for step in workflow.steps:
                    if step.id in completed_steps:
                        continue
                    if step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED]:
                        completed_steps.add(step.id)
                        completed_names.add(step.name)
                        continue
                    
                    # Check dependencies (by name)
                    deps_satisfied = all(
                        dep_name in completed_names 
                        for dep_name in step.depends_on
                    )
                    
                    if deps_satisfied:
                        runnable.append(step)
                
                if not runnable:
                    if len(completed_steps) < len(workflow.steps):
                        raise RuntimeError("Workflow deadlock - circular dependencies?")
                    break
                
                # Execute runnable steps (could parallelize here)
                for step in runnable:
                    await self.broadcast("step_started", {
                        "workflow_id": workflow.id,
                        "step_id": step.id,
                        "step_name": step.name
                    })
                    
                    try:
                        result = await self.execute_step(step, workflow.context)
                        
                        # Store result in context for downstream steps
                        result_key = step.name.lower().replace(" ", "_") + "_result"
                        workflow.context[result_key] = result
                        
                        await self.broadcast("step_completed", {
                            "workflow_id": workflow.id,
                            "step_id": step.id,
                            "step_name": step.name,
                            "status": step.status.value
                        })
                        
                    except Exception as e:
                        await self.broadcast("step_failed", {
                            "workflow_id": workflow.id,
                            "step_id": step.id,
                            "error": str(e)
                        })
                        
                        if step.on_failure == "fail":
                            raise
                    
                    completed_steps.add(step.id)
                    completed_names.add(step.name)
                    workflow.current_step = len(completed_steps)
                    store.save(workflow)
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now(timezone.utc).isoformat()
            
            await self.broadcast("workflow_completed", {
                "id": workflow.id,
                "name": workflow.name,
                "duration": workflow.completed_at
            })
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(e)
            workflow.completed_at = datetime.now(timezone.utc).isoformat()
            
            await self.broadcast("workflow_failed", {
                "id": workflow.id,
                "error": str(e)
            })
        
        finally:
            store.save(workflow)
            if workflow.id in self.running_workflows:
                del self.running_workflows[workflow.id]
    
    def start_workflow(self, workflow: Workflow, background_tasks: BackgroundTasks):
        """Start workflow in background"""
        task = asyncio.create_task(self.run_workflow(workflow))
        self.running_workflows[workflow.id] = task
        return workflow.id
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id in self.running_workflows:
            self.running_workflows[workflow_id].cancel()
            workflow = store.get(workflow_id)
            if workflow:
                workflow.status = WorkflowStatus.CANCELLED
                store.save(workflow)
            return True
        return False


engine = WorkflowEngine()

# =============================================================================
# API MODELS
# =============================================================================

class StepInput(BaseModel):
    name: str
    service: str
    action: str
    params: Dict[str, Any] = {}
    depends_on: List[str] = []
    on_failure: str = "fail"
    timeout_seconds: int = 300
    max_retries: int = 3


class CreateWorkflowRequest(BaseModel):
    name: str
    description: str = ""
    steps: List[StepInput]
    context: Dict[str, Any] = {}
    trigger: TriggerType = TriggerType.MANUAL


class RunTemplateRequest(BaseModel):
    template: str
    context: Dict[str, Any] = {}


class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: str
    status: WorkflowStatus
    trigger: TriggerType
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    current_step: int
    total_steps: int
    error: Optional[str]


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "Origin OS Orchestrator",
        "version": "1.0",
        "features": [
            "Multi-step workflow execution",
            "Cross-service orchestration",
            "Dependency management",
            "Parallel execution",
            "Retry with backoff",
            "Real-time WebSocket updates",
            "Built-in workflow templates"
        ],
        "services": list(SERVICES.keys()),
        "templates": list(store.templates.keys()),
        "endpoints": {
            "create": "POST /workflows",
            "list": "GET /workflows",
            "get": "GET /workflows/{id}",
            "run": "POST /workflows/{id}/run",
            "cancel": "POST /workflows/{id}/cancel",
            "templates": "GET /templates",
            "run_template": "POST /templates/{name}/run",
            "websocket": "WS /ws"
        }
    }


@app.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(req: CreateWorkflowRequest):
    """Create a new workflow"""
    workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
    
    steps = [
        WorkflowStep(
            id=f"step_{i}_{uuid.uuid4().hex[:6]}",
            name=s.name,
            service=s.service,
            action=s.action,
            params=s.params,
            depends_on=s.depends_on,
            on_failure=s.on_failure,
            timeout_seconds=s.timeout_seconds,
            max_retries=s.max_retries
        )
        for i, s in enumerate(req.steps)
    ]
    
    workflow = Workflow(
        id=workflow_id,
        name=req.name,
        description=req.description,
        steps=steps,
        trigger=req.trigger,
        created_at=datetime.now(timezone.utc).isoformat(),
        context=req.context
    )
    
    store.save(workflow)
    
    return WorkflowResponse(
        id=workflow.id,
        name=workflow.name,
        description=workflow.description,
        status=workflow.status,
        trigger=workflow.trigger,
        created_at=workflow.created_at,
        started_at=workflow.started_at,
        completed_at=workflow.completed_at,
        current_step=workflow.current_step,
        total_steps=len(workflow.steps),
        error=workflow.error
    )


@app.get("/workflows", response_model=List[WorkflowResponse])
async def list_workflows():
    """List all workflows"""
    return [
        WorkflowResponse(
            id=w.id,
            name=w.name,
            description=w.description,
            status=w.status,
            trigger=w.trigger,
            created_at=w.created_at,
            started_at=w.started_at,
            completed_at=w.completed_at,
            current_step=w.current_step,
            total_steps=len(w.steps),
            error=w.error
        )
        for w in store.list_all()
    ]


@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow details including steps"""
    workflow = store.get(workflow_id)
    if not workflow:
        raise HTTPException(404, f"Workflow {workflow_id} not found")
    
    return workflow.to_dict()


@app.post("/workflows/{workflow_id}/run")
async def run_workflow(workflow_id: str, background_tasks: BackgroundTasks):
    """Start running a workflow"""
    workflow = store.get(workflow_id)
    if not workflow:
        raise HTTPException(404, f"Workflow {workflow_id} not found")
    
    if workflow.status == WorkflowStatus.RUNNING:
        raise HTTPException(400, "Workflow is already running")
    
    # Reset workflow state
    workflow.status = WorkflowStatus.PENDING
    workflow.current_step = 0
    workflow.error = None
    for step in workflow.steps:
        step.status = StepStatus.PENDING
        step.result = None
        step.error = None
        step.retry_count = 0
    
    # Start in background
    background_tasks.add_task(engine.run_workflow, workflow)
    
    return {"started": workflow_id, "status": "running"}


@app.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow"""
    if await engine.cancel_workflow(workflow_id):
        return {"cancelled": workflow_id}
    raise HTTPException(400, "Workflow not running or not found")


@app.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow"""
    if store.delete(workflow_id):
        return {"deleted": workflow_id}
    raise HTTPException(404, f"Workflow {workflow_id} not found")


@app.get("/templates")
async def list_templates():
    """List available workflow templates"""
    return {
        "templates": [
            {"name": k, "description": v["description"]}
            for k, v in store.templates.items()
        ]
    }


@app.get("/templates/{name}")
async def get_template(name: str):
    """Get template details"""
    if name not in store.templates:
        raise HTTPException(404, f"Template {name} not found")
    return store.templates[name]


@app.post("/templates/{name}/run")
async def run_template(name: str, req: RunTemplateRequest, background_tasks: BackgroundTasks):
    """Create and run a workflow from template"""
    if name not in store.templates:
        raise HTTPException(404, f"Template {name} not found")
    
    template = store.templates[name]
    workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
    
    steps = []
    for i, s in enumerate(template["steps"]):
        step = WorkflowStep(
            id=f"step_{i}_{uuid.uuid4().hex[:6]}",
            name=s["name"],
            service=s["service"],
            action=s["action"],
            params=s["params"],
            depends_on=s.get("depends_on", [])
        )
        steps.append(step)
    
    workflow = Workflow(
        id=workflow_id,
        name=f"{template['name']} - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description=template["description"],
        steps=steps,
        trigger=TriggerType.MANUAL,
        created_at=datetime.now(timezone.utc).isoformat(),
        context=req.context
    )
    
    store.save(workflow)
    background_tasks.add_task(engine.run_workflow, workflow)
    
    return {"workflow_id": workflow_id, "template": name, "status": "started"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time workflow updates"""
    await websocket.accept()
    engine.websockets.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming commands if needed
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        engine.websockets.remove(websocket)


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "orchestrator",
        "workflows_count": len(store.workflows),
        "running_workflows": len(engine.running_workflows),
        "templates_count": len(store.templates)
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎭 ORIGIN OS ORCHESTRATOR")
    print("=" * 60)
    print(f"\nConnected Services:")
    for name, url in SERVICES.items():
        print(f"  • {name}: {url}")
    print(f"\nBuilt-in Templates:")
    for name in store.templates:
        print(f"  • {name}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
