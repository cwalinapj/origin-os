# codex/enforce.py
from codex.state import get_mode
import yaml

with open("/codex/enforcement_matrix.yaml") as f:
    MATRIX = yaml.safe_load(f)

class EnforcementResult:
    def __init__(self, allowed, requires_override=False, reason=None):
        self.allowed = allowed
        self.requires_override = requires_override
        self.reason = reason

def enforce(action_type):
    mode = get_mode()
    rule = MATRIX[mode].get(action_type)
    if not rule:
        return EnforcementResult(False, reason="Unknown action")
    action = rule["action"]
    if action == "allow":
        return EnforcementResult(True)
    if action == "allow_with_override":
        return EnforcementResult(True, requires_override=True)
    if action == "forbid":
        return EnforcementResult(False, reason=f"Forbidden in {mode}")
    return EnforcementResult(False, reason="Invalid policy")
