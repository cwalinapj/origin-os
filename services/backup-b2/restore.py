#!/usr/bin/env python3
"""
ORIGIN OS RESTORE — v1.0 (LOCKED)
=================================
Immutable restore system with strict constraints:
- NO production restores
- NO baseline mutation
- NO overwrites
- Checksum verification required
- Codex + Sentinel approval required
- Full audit trail
"""

import os
import sys
import json
import uuid
import hashlib
import tarfile
import boto3
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
S3_BUCKET = os.environ.get("S3_BUCKET", "origin-os-backups")

VAULT_URL = os.environ.get("VAULT_URL", "http://origin-vault:8000")
CODEX_URL = os.environ.get("CODEX_URL", "http://origin-codex:8000")

OUTPUT_DIR = "/restore"
AUDIT_DIR = "/audit"

# =============================================================================
# ENUMS
# =============================================================================

class Environment(Enum):
    SANDBOX = "sandbox"
    STAGING = "staging"
    FORENSICS = "forensics"
    # PRODUCTION is forbidden


class RestoreMode(Enum):
    READ_ONLY = "read_only"
    CLONE = "clone"
    FORENSIC = "forensic"
    # OVERWRITE is forbidden


# =============================================================================
# RESTORE REQUEST
# =============================================================================

class RestoreRequest:
    def __init__(
        self,
        source_bucket: str,
        object_key: str,
        expected_checksum: str,
        target_environment: Environment,
        target_container: str,
        restore_mode: RestoreMode,
        requested_by: str,
    ):
        self.restore_id = str(uuid.uuid4())
        self.requested_at = datetime.now(timezone.utc)
        self.requested_by = requested_by
        
        # Source
        self.source_bucket = source_bucket
        self.object_key = object_key
        self.expected_checksum = expected_checksum
        
        # Target
        self.target_environment = target_environment
        self.target_container = target_container
        self.restore_mode = restore_mode
        
        # Validation state
        self.retention_verified = False
        self.checksum_verified = False
        self.codex_approved = False
        self.sentinel_approved = False
        
    def to_dict(self) -> dict:
        return {
            "restore_id": self.restore_id,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "source": {
                "bucket": self.source_bucket,
                "object_key": self.object_key,
                "checksum": self.expected_checksum,
                "retention_verified": self.retention_verified,
            },
            "target": {
                "environment": self.target_environment.value,
                "container": self.target_container,
                "restore_mode": self.restore_mode.value,
            },
            "validation": {
                "checksum_verified": self.checksum_verified,
                "codex_approved": self.codex_approved,
                "sentinel_approved": self.sentinel_approved,
            },
        }


# =============================================================================
# S3 CLIENT
# =============================================================================

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def verify_retention(s3_client, bucket: str, key: str) -> tuple[bool, Optional[str]]:
    """Verify object has valid retention (Object Lock)"""
    try:
        response = s3_client.get_object_retention(Bucket=bucket, Key=key)
        retention = response.get("Retention", {})
        mode = retention.get("Mode")
        retain_until = retention.get("RetainUntilDate")
        
        if mode in ["COMPLIANCE", "GOVERNANCE"] and retain_until:
            return True, f"{mode} until {retain_until}"
        return False, "No valid retention found"
    except Exception as e:
        return False, str(e)


def sha256_file(filepath: str) -> str:
    """Calculate SHA256 hash of a file"""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(filepath: str, expected: str) -> bool:
    """Verify file checksum matches expected"""
    actual = sha256_file(filepath)
    return actual == expected


def request_codex_approval(request: RestoreRequest) -> tuple[bool, str]:
    """
    Request approval from Codex (GTM API).
    In production, this would call the Codex API.
    For now, we require manual approval via environment variable.
    """
    approval_token = os.environ.get("CODEX_APPROVAL_TOKEN")
    if approval_token == request.restore_id:
        return True, "Codex approval granted"
    return False, "Codex approval required. Set CODEX_APPROVAL_TOKEN={restore_id}"


def request_sentinel_approval(request: RestoreRequest) -> tuple[bool, str]:
    """
    Request approval from Sentinel (security monitor).
    In production, this would call the Sentinel API.
    For now, we require manual approval via environment variable.
    """
    approval_token = os.environ.get("SENTINEL_APPROVAL_TOKEN")
    if approval_token == request.restore_id:
        return True, "Sentinel approval granted"
    return False, "Sentinel approval required. Set SENTINEL_APPROVAL_TOKEN={restore_id}"


# =============================================================================
# AUDIT FUNCTIONS
# =============================================================================

def log_audit(request: RestoreRequest, action: str, result: str, details: str = ""):
    """Log restore action to audit trail"""
    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "restore_id": request.restore_id,
        "action": action,
        "result": result,
        "details": details,
        "request": request.to_dict(),
    }
    
    # Write to local audit log
    os.makedirs(AUDIT_DIR, exist_ok=True)
    audit_file = os.path.join(AUDIT_DIR, f"restore_{request.restore_id}.json")
    
    with open(audit_file, 'a') as f:
        f.write(json.dumps(audit_entry) + "\n")
    
    print(f"[AUDIT] {action}: {result}")
    
    # TODO: In production, also log to Vault immutable storage


# =============================================================================
# RESTORE FUNCTIONS
# =============================================================================

def download_backup(s3_client, request: RestoreRequest) -> Optional[str]:
    """Download backup file from S3"""
    local_path = os.path.join(OUTPUT_DIR, "downloads", os.path.basename(request.object_key))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        s3_client.download_file(request.source_bucket, request.object_key, local_path)
        return local_path
    except Exception as e:
        log_audit(request, "download", "FAILED", str(e))
        return None


def extract_backup(request: RestoreRequest, local_path: str) -> Optional[str]:
    """Extract backup to restore directory"""
    extract_dir = os.path.join(
        OUTPUT_DIR,
        request.target_environment.value,
        request.target_container,
        request.restore_id
    )
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with tarfile.open(local_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        return extract_dir
    except Exception as e:
        log_audit(request, "extract", "FAILED", str(e))
        return None


# =============================================================================
# MAIN RESTORE FUNCTION
# =============================================================================

def execute_restore(request: RestoreRequest) -> bool:
    """
    Execute a restore with full validation.
    
    Constraints enforced:
    - overwrite_existing: false
    - production_restore: forbidden
    - baseline_mutation: forbidden
    """
    
    print("=" * 60)
    print("ORIGIN OS RESTORE — v1.0")
    print("=" * 60)
    print(f"Restore ID:  {request.restore_id}")
    print(f"Requested:   {request.requested_at}")
    print(f"Source:      s3://{request.source_bucket}/{request.object_key}")
    print(f"Target:      {request.target_environment.value}/{request.target_container}")
    print(f"Mode:        {request.restore_mode.value}")
    print("=" * 60)
    print()
    
    log_audit(request, "restore_initiated", "STARTED")
    
    s3_client = get_s3_client()
    
    # =========================================================================
    # STEP 1: Verify Object Lock retention
    # =========================================================================
    print("🔒 Step 1: Verifying Object Lock retention...")
    
    retention_ok, retention_msg = verify_retention(
        s3_client, request.source_bucket, request.object_key
    )
    
    if not retention_ok:
        log_audit(request, "retention_check", "FAILED", retention_msg)
        print(f"   ❌ FAILED: {retention_msg}")
        print("   Restore from non-immutable sources is FORBIDDEN")
        return False
    
    request.retention_verified = True
    log_audit(request, "retention_check", "PASSED", retention_msg)
    print(f"   ✅ {retention_msg}")
    print()
    
    # =========================================================================
    # STEP 2: Download backup
    # =========================================================================
    print("📥 Step 2: Downloading backup...")
    
    local_path = download_backup(s3_client, request)
    if not local_path:
        print("   ❌ Download failed")
        return False
    
    log_audit(request, "download", "COMPLETED", local_path)
    print(f"   ✅ Downloaded to {local_path}")
    print()
    
    # =========================================================================
    # STEP 3: Verify checksum
    # =========================================================================
    print("🔐 Step 3: Verifying checksum...")
    
    if not verify_checksum(local_path, request.expected_checksum):
        log_audit(request, "checksum_check", "FAILED", "Checksum mismatch")
        print("   ❌ CHECKSUM MISMATCH - backup may be corrupted or tampered")
        print("   Restore ABORTED")
        return False
    
    request.checksum_verified = True
    log_audit(request, "checksum_check", "PASSED")
    print(f"   ✅ Checksum verified: {request.expected_checksum[:16]}...")
    print()
    
    # =========================================================================
    # STEP 4: Request Codex approval
    # =========================================================================
    print("📋 Step 4: Requesting Codex approval...")
    
    codex_ok, codex_msg = request_codex_approval(request)
    if not codex_ok:
        log_audit(request, "codex_approval", "PENDING", codex_msg)
        print(f"   ⏳ {codex_msg}")
        return False
    
    request.codex_approved = True
    log_audit(request, "codex_approval", "GRANTED")
    print(f"   ✅ {codex_msg}")
    print()
    
    # =========================================================================
    # STEP 5: Request Sentinel approval
    # =========================================================================
    print("🛡️  Step 5: Requesting Sentinel approval...")
    
    sentinel_ok, sentinel_msg = request_sentinel_approval(request)
    if not sentinel_ok:
        log_audit(request, "sentinel_approval", "PENDING", sentinel_msg)
        print(f"   ⏳ {sentinel_msg}")
        return False
    
    request.sentinel_approved = True
    log_audit(request, "sentinel_approval", "GRANTED")
    print(f"   ✅ {sentinel_msg}")
    print()
    
    # =========================================================================
    # STEP 6: Extract to target
    # =========================================================================
    print("📦 Step 6: Extracting to target...")
    
    extract_dir = extract_backup(request, local_path)
    if not extract_dir:
        print("   ❌ Extraction failed")
        return False
    
    log_audit(request, "extract", "COMPLETED", extract_dir)
    print(f"   ✅ Extracted to {extract_dir}")
    print()
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    log_audit(request, "restore_completed", "SUCCESS", extract_dir)
    
    print("=" * 60)
    print("✅ RESTORE COMPLETED")
    print("=" * 60)
    print(f"Restore ID:   {request.restore_id}")
    print(f"Location:     {extract_dir}")
    print(f"Mode:         {request.restore_mode.value}")
    print()
    print("⚠️  CONSTRAINTS ENFORCED:")
    print("   - Target is NOT production")
    print("   - Existing data NOT overwritten")
    print("   - Baseline NOT mutated")
    print("=" * 60)
    
    return True


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Origin OS Restore")
    parser.add_argument("--bucket", required=True, help="Source S3 bucket")
    parser.add_argument("--key", required=True, help="Object key (path)")
    parser.add_argument("--checksum", required=True, help="Expected SHA256 checksum")
    parser.add_argument("--environment", required=True, 
                        choices=["sandbox", "staging", "forensics"],
                        help="Target environment")
    parser.add_argument("--container", required=True, help="Target container name")
    parser.add_argument("--mode", required=True,
                        choices=["read_only", "clone", "forensic"],
                        help="Restore mode")
    parser.add_argument("--requested-by", required=True, help="Requester ID")
    
    args = parser.parse_args()
    
    request = RestoreRequest(
        source_bucket=args.bucket,
        object_key=args.key,
        expected_checksum=args.checksum,
        target_environment=Environment(args.environment),
        target_container=args.container,
        restore_mode=RestoreMode(args.mode),
        requested_by=args.requested_by,
    )
    
    # Print restore ID for approval workflow
    print(f"\n🔑 RESTORE ID: {request.restore_id}")
    print("   Use this ID for Codex and Sentinel approval tokens\n")
    
    success = execute_restore(request)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
