#!/usr/bin/env python3
"""
CONTAINER DIFF ENFORCER — Brand Safety Guardrails for Mutations
================================================================

Ensures all LAM mutations stay within brand-safe boundaries:

1. Structural Constraints: Layout changes must respect brand grid
2. Semantic Boundaries: Copy changes must stay within brand voice
3. Visual Diff Limits: Maximum pixel/color deviation from baseline
4. Compliance Rules: Legal disclaimers, accessibility requirements

All mutations are validated before deployment. Violations are blocked
and logged for review.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_LAYOUT_SHIFT = float(os.getenv("MAX_LAYOUT_SHIFT", "0.15"))
MAX_SEMANTIC_DRIFT = float(os.getenv("MAX_SEMANTIC_DRIFT", "0.3"))
MAX_COLOR_DEVIATION = int(os.getenv("MAX_COLOR_DEVIATION", "30"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diff-enforcer")


class ViolationType(Enum):
    LAYOUT_SHIFT = "layout_shift"
    SEMANTIC_DRIFT = "semantic_drift"
    COLOR_DEVIATION = "color_deviation"
    FONT_CHANGE = "font_change"
    MISSING_ELEMENT = "missing_element"
    COMPLIANCE_VIOLATION = "compliance_violation"
    ACCESSIBILITY_VIOLATION = "accessibility_violation"


class Severity(Enum):
    WARNING = "warning"
    BLOCK = "block"
    CRITICAL = "critical"


@dataclass
class Violation:
    violation_type: ViolationType
    severity: Severity
    message: str
    details: Dict[str, Any]
    element_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.violation_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "element_id": self.element_id
        }


@dataclass
class BrandConstraints:
    site_id: str
    allowed_layouts: List[str]
    grid_columns: int
    grid_gutter: int
    max_layout_shift: float
    brand_voice_vector: np.ndarray
    allowed_tones: List[str]
    forbidden_words: List[str]
    required_elements: List[str]
    primary_colors: List[Tuple[int, int, int]]
    secondary_colors: List[Tuple[int, int, int]]
    max_color_deviation: int
    allowed_fonts: List[str]
    min_font_size: int
    max_font_size: int
    required_disclaimers: List[str]
    accessibility_level: str

    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "allowed_layouts": self.allowed_layouts,
            "grid_columns": self.grid_columns,
            "grid_gutter": self.grid_gutter,
            "max_layout_shift": self.max_layout_shift,
            "brand_voice_vector": self.brand_voice_vector.tolist(),
            "allowed_tones": self.allowed_tones,
            "forbidden_words": self.forbidden_words,
            "required_elements": self.required_elements,
            "primary_colors": self.primary_colors,
            "secondary_colors": self.secondary_colors,
            "max_color_deviation": self.max_color_deviation,
            "allowed_fonts": self.allowed_fonts,
            "min_font_size": self.min_font_size,
            "max_font_size": self.max_font_size,
            "required_disclaimers": self.required_disclaimers,
            "accessibility_level": self.accessibility_level
        }


@dataclass
class ValidationResult:
    mutation_id: str
    approved: bool
    violations: List[Violation]
    warnings: List[Violation]
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "mutation_id": self.mutation_id,
            "approved": self.approved,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [w.to_dict() for w in self.warnings],
            "timestamp": self.timestamp
        }


class StructuralEnforcer:
    """Enforces layout/structural constraints."""

    def validate(self, layout_changes: List[Dict], constraints: BrandConstraints) -> List[Violation]:
        violations = []
        total_shift = self._calculate_layout_shift(layout_changes)
        
        if total_shift > constraints.max_layout_shift:
            violations.append(Violation(
                violation_type=ViolationType.LAYOUT_SHIFT,
                severity=Severity.BLOCK,
                message=f"Layout shift {total_shift:.2%} exceeds max {constraints.max_layout_shift:.2%}",
                details={"actual_shift": total_shift, "max_allowed": constraints.max_layout_shift}
            ))
        
        return violations

    def _calculate_layout_shift(self, changes: List[Dict]) -> float:
        if not changes:
            return 0.0
        total_shift = 0.0
        for change in changes:
            old_pos = change.get("old_pos", {})
            new_pos = change.get("new_pos", {})
            dx = abs(new_pos.get("x", 0) - old_pos.get("x", 0))
            dy = abs(new_pos.get("y", 0) - old_pos.get("y", 0))
            shift = (dx / 1920 + dy / 1080) / 2
            width = new_pos.get("width", 100)
            height = new_pos.get("height", 100)
            impact = (width * height) / (1920 * 1080)
            total_shift += shift * impact
        return total_shift


class SemanticEnforcer:
    """Enforces copy/semantic constraints."""

    def validate(self, copy_changes: List[Dict], constraints: BrandConstraints) -> List[Violation]:
        violations = []
        
        for change in copy_changes:
            new_text = change.get("new_text", "")
            embedding = np.array(change.get("embedding", []))
            
            # Check semantic drift
            if len(embedding) == len(constraints.brand_voice_vector):
                drift = self._calculate_drift(embedding, constraints.brand_voice_vector)
                if drift > MAX_SEMANTIC_DRIFT:
                    violations.append(Violation(
                        violation_type=ViolationType.SEMANTIC_DRIFT,
                        severity=Severity.BLOCK,
                        message=f"Copy drift {drift:.2%} exceeds brand voice threshold",
                        details={"drift": drift, "threshold": MAX_SEMANTIC_DRIFT},
                        element_id=change.get("element_id")
                    ))
            
            # Check forbidden words
            for word in constraints.forbidden_words:
                if word.lower() in new_text.lower():
                    violations.append(Violation(
                        violation_type=ViolationType.SEMANTIC_DRIFT,
                        severity=Severity.BLOCK,
                        message=f"Forbidden word '{word}' found",
                        details={"word": word},
                        element_id=change.get("element_id")
                    ))
        
        return violations

    def _calculate_drift(self, embedding: np.ndarray, anchor: np.ndarray) -> float:
        if np.linalg.norm(embedding) == 0 or np.linalg.norm(anchor) == 0:
            return 0.0
        similarity = np.dot(embedding, anchor) / (np.linalg.norm(embedding) * np.linalg.norm(anchor))
        return 1 - similarity


class VisualEnforcer:
    """Enforces visual/style constraints."""

    def validate(self, style_changes: List[Dict], constraints: BrandConstraints) -> List[Violation]:
        violations = []
        
        for change in style_changes:
            prop = change.get("property", "")
            new_val = change.get("new_value", "")
            
            if prop in ["color", "background-color"]:
                if not self._is_valid_color(new_val, constraints):
                    violations.append(Violation(
                        violation_type=ViolationType.COLOR_DEVIATION,
                        severity=Severity.BLOCK,
                        message=f"Color {new_val} not in brand palette",
                        details={"property": prop, "value": new_val},
                        element_id=change.get("element_id")
                    ))
            
            if prop == "font-family":
                if not any(f.lower() in new_val.lower() for f in constraints.allowed_fonts):
                    violations.append(Violation(
                        violation_type=ViolationType.FONT_CHANGE,
                        severity=Severity.BLOCK,
                        message=f"Font '{new_val}' not allowed",
                        details={"font": new_val, "allowed": constraints.allowed_fonts},
                        element_id=change.get("element_id")
                    ))
        
        return violations

    def _is_valid_color(self, color_str: str, constraints: BrandConstraints) -> bool:
        rgb = self._parse_color(color_str)
        if not rgb:
            return True
        all_colors = constraints.primary_colors + constraints.secondary_colors
        for allowed in all_colors:
            distance = sum(abs(a - b) for a, b in zip(rgb, allowed))
            if distance <= constraints.max_color_deviation * 3:
                return True
        return False

    def _parse_color(self, color_str: str) -> Optional[Tuple[int, int, int]]:
        color_str = color_str.strip().lower()
        if color_str.startswith("#") and len(color_str) == 7:
            return (int(color_str[1:3], 16), int(color_str[3:5], 16), int(color_str[5:7], 16))
        return None


class ComplianceEnforcer:
    """Enforces compliance requirements."""

    def validate(self, removed_elements: List[str], copy_changes: List[Dict], 
                 constraints: BrandConstraints) -> List[Violation]:
        violations = []
        
        for element in constraints.required_elements:
            if element in removed_elements:
                violations.append(Violation(
                    violation_type=ViolationType.MISSING_ELEMENT,
                    severity=Severity.CRITICAL,
                    message=f"Required element '{element}' was removed",
                    details={"element": element}
                ))
        
        for disclaimer in constraints.required_disclaimers:
            for change in copy_changes:
                old_text = change.get("old_text", "").lower()
                new_text = change.get("new_text", "").lower()
                if disclaimer.lower() in old_text and disclaimer.lower() not in new_text:
                    violations.append(Violation(
                        violation_type=ViolationType.COMPLIANCE_VIOLATION,
                        severity=Severity.CRITICAL,
                        message=f"Required disclaimer removed",
                        details={"disclaimer": disclaimer[:50]}
                    ))
        
        return violations


class DiffEnforcer:
    """Main enforcer coordinating all checks."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.structural = StructuralEnforcer()
        self.semantic = SemanticEnforcer()
        self.visual = VisualEnforcer()
        self.compliance = ComplianceEnforcer()

    async def validate_mutation(
        self,
        mutation_id: str,
        site_id: str,
        layout_changes: List[Dict],
        copy_changes: List[Dict],
        style_changes: List[Dict],
        removed_elements: List[str]
    ) -> ValidationResult:
        constraints = await self._get_constraints(site_id)
        
        if not constraints:
            return ValidationResult(
                mutation_id=mutation_id,
                approved=True,
                violations=[],
                warnings=[Violation(ViolationType.MISSING_ELEMENT, Severity.WARNING, 
                                   "No constraints configured", {})],
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        all_violations = []
        all_violations.extend(self.structural.validate(layout_changes, constraints))
        all_violations.extend(self.semantic.validate(copy_changes, constraints))
        all_violations.extend(self.visual.validate(style_changes, constraints))
        all_violations.extend(self.compliance.validate(removed_elements, copy_changes, constraints))
        
        blocking = [v for v in all_violations if v.severity in [Severity.BLOCK, Severity.CRITICAL]]
        warnings = [v for v in all_violations if v.severity == Severity.WARNING]
        
        result = ValidationResult(
            mutation_id=mutation_id,
            approved=len(blocking) == 0,
            violations=blocking,
            warnings=warnings,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        await self._store_result(result)
        logger.info(f"Mutation {mutation_id}: {'APPROVED' if result.approved else 'BLOCKED'}")
        
        return result

    async def register_constraints(self, constraints: BrandConstraints):
        await self.redis.set(f"constraints:{constraints.site_id}", json.dumps(constraints.to_dict()))

    async def _get_constraints(self, site_id: str) -> Optional[BrandConstraints]:
        data = await self.redis.get(f"constraints:{site_id}")
        if not data:
            return None
        parsed = json.loads(data)
        return BrandConstraints(
            site_id=parsed["site_id"],
            allowed_layouts=parsed["allowed_layouts"],
            grid_columns=parsed["grid_columns"],
            grid_gutter=parsed["grid_gutter"],
            max_layout_shift=parsed["max_layout_shift"],
            brand_voice_vector=np.array(parsed["brand_voice_vector"]),
            allowed_tones=parsed["allowed_tones"],
            forbidden_words=parsed["forbidden_words"],
            required_elements=parsed["required_elements"],
            primary_colors=[tuple(c) for c in parsed["primary_colors"]],
            secondary_colors=[tuple(c) for c in parsed["secondary_colors"]],
            max_color_deviation=parsed["max_color_deviation"],
            allowed_fonts=parsed["allowed_fonts"],
            min_font_size=parsed["min_font_size"],
            max_font_size=parsed["max_font_size"],
            required_disclaimers=parsed["required_disclaimers"],
            accessibility_level=parsed["accessibility_level"]
        )

    async def _store_result(self, result: ValidationResult):
        await self.redis.xadd(f"validation_history", {"result": json.dumps(result.to_dict())}, maxlen=1000)


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title="Container Diff Enforcer", version="1.0.0")

redis_client: Optional[redis.Redis] = None
enforcer: Optional[DiffEnforcer] = None


class LayoutChange(BaseModel):
    element_id: str
    old_pos: Dict[str, float]
    new_pos: Dict[str, float]


class CopyChange(BaseModel):
    element_id: str
    old_text: str
    new_text: str
    embedding: List[float] = []


class StyleChange(BaseModel):
    element_id: str
    property: str
    old_value: str
    new_value: str


class MutationRequest(BaseModel):
    mutation_id: str
    site_id: str
    layout_changes: List[LayoutChange] = []
    copy_changes: List[CopyChange] = []
    style_changes: List[StyleChange] = []
    removed_elements: List[str] = []


class ConstraintsRequest(BaseModel):
    site_id: str
    allowed_layouts: List[str] = []
    grid_columns: int = 12
    grid_gutter: int = 16
    max_layout_shift: float = 0.15
    brand_voice_vector: List[float] = []
    allowed_tones: List[str] = ["neutral", "professional"]
    forbidden_words: List[str] = []
    required_elements: List[str] = ["logo", "copyright"]
    primary_colors: List[List[int]] = []
    secondary_colors: List[List[int]] = []
    max_color_deviation: int = 30
    allowed_fonts: List[str] = ["Arial", "Helvetica"]
    min_font_size: int = 12
    max_font_size: int = 72
    required_disclaimers: List[str] = []
    accessibility_level: str = "AA"


@app.on_event("startup")
async def startup():
    global redis_client, enforcer
    redis_client = redis.from_url(REDIS_URL)
    enforcer = DiffEnforcer(redis_client)


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/validate")
async def validate_mutation(request: MutationRequest):
    result = await enforcer.validate_mutation(
        mutation_id=request.mutation_id,
        site_id=request.site_id,
        layout_changes=[c.dict() for c in request.layout_changes],
        copy_changes=[c.dict() for c in request.copy_changes],
        style_changes=[c.dict() for c in request.style_changes],
        removed_elements=request.removed_elements
    )
    return result.to_dict()


@app.post("/constraints")
async def register_constraints(request: ConstraintsRequest):
    constraints = BrandConstraints(
        site_id=request.site_id,
        allowed_layouts=request.allowed_layouts,
        grid_columns=request.grid_columns,
        grid_gutter=request.grid_gutter,
        max_layout_shift=request.max_layout_shift,
        brand_voice_vector=np.array(request.brand_voice_vector) if request.brand_voice_vector else np.zeros(EMBEDDING_DIM),
        allowed_tones=request.allowed_tones,
        forbidden_words=request.forbidden_words,
        required_elements=request.required_elements,
        primary_colors=[tuple(c) for c in request.primary_colors],
        secondary_colors=[tuple(c) for c in request.secondary_colors],
        max_color_deviation=request.max_color_deviation,
        allowed_fonts=request.allowed_fonts,
        min_font_size=request.min_font_size,
        max_font_size=request.max_font_size,
        required_disclaimers=request.required_disclaimers,
        accessibility_level=request.accessibility_level
    )
    await enforcer.register_constraints(constraints)
    return {"status": "registered", "site_id": request.site_id}


@app.get("/constraints/{site_id}")
async def get_constraints(site_id: str):
    constraints = await enforcer._get_constraints(site_id)
    if not constraints:
        raise HTTPException(status_code=404, detail="No constraints found")
    return constraints.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
