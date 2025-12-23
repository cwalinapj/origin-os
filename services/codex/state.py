# codex/state.py
import yaml
from pathlib import Path

SYSTEM_STATE_PATH = Path("/system/system_state.yaml")

def load_system_state():
    with open(SYSTEM_STATE_PATH, "r") as f:
        return yaml.safe_load(f)

def get_mode():
    state = load_system_state()
    return state["state"]["mode"]
