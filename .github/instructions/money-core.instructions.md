---
applyTo:
  - "programs/session_escrow/**"
  - "programs/collateral_vault/**"
  - "programs/mode_registry/**"
---

# Money-core security rules (must-follow)

⚠️ IMPORTANT: Read and follow all security rules in `/AGENTS.md` before making any code changes.

DO NOT modify these security-sensitive programs without explicit human approval:
- `programs/session_escrow/` — Escrow, vault interactions, claims, permit verification
- `programs/collateral_vault/` — Collateral custody, slashing
- `programs/mode_registry/` — Mode config, verifier allowlist

For any changes to these programs, you must:
1. STOP and explain what change is needed
2. Provide code as a suggestion only (no direct edits unless approved)
3. Request human review via the Security PR process
