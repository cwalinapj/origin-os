#!/usr/bin/env python3
"""
SENTINEL — Origin OS Security Guardian
=======================================
Always-on verification service that:
- Monitors backup integrity
- Verifies Object Lock compliance
- Detects tampering
- Validates checksums
- Reports violations to Codex

This service does NOT backup — it VERIFIES.
"""

import boto3
import hashlib
import json
import os
import sys
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
S3_KEY = os.getenv("S3_KEY")
S3_SECRET = os.getenv("S3_SECRET")
BACKUP_BUCKET = os.getenv("BACKUP_BUCKET", "origin-os-backups")

CODEX_URL = os.getenv("CODEX_URL", "http://origin-codex:8000")
VAULT_URL = os.getenv("VAULT_URL", "http://origin-vault:8000")

# Verification interval (seconds)
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "3600"))  # 1 hour default

# Required settings
REQUIRED_LOCK_MODE = "COMPLIANCE"
MIN_RETENTION_DAYS = 365

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SENTINEL] %(levelname)s: %(message)s'
)
logger = logging.getLogger("sentinel")

# =============================================================================
# S3 CLIENT
# =============================================================================

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET,
    )

# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

class Violation:
    """Represents a security violation"""
    def __init__(self, severity: str, object_key: str, violation_type: str, details: str):
        self.severity = severity  # P0, P1, P2
        self.object_key = object_key
        self.violation_type = violation_type
        self.details = details
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "object_key": self.object_key,
            "violation_type": self.violation_type,
            "details": self.details,
            "timestamp": self.timestamp,
        }


def verify_object_lock(s3_client, bucket: str, key: str) -> Optional[Violation]:
    """Verify object has valid Object Lock retention"""
    try:
        response = s3_client.get_object_retention(Bucket=bucket, Key=key)
        retention = response.get("Retention", {})
        mode = retention.get("Mode")
        retain_until = retention.get("RetainUntilDate")
        
        if not mode:
            return Violation("P0", key, "missing_retention", "Object has no Object Lock")
        
        if mode != REQUIRED_LOCK_MODE:
            return Violation("P0", key, "invalid_lock_mode", 
                           f"Expected {REQUIRED_LOCK_MODE}, got {mode}")
        
        if retain_until:
            # Check retention period
            now = datetime.now(timezone.utc)
            days_remaining = (retain_until.replace(tzinfo=timezone.utc) - now).days
            if days_remaining < 0:
                return Violation("P0", key, "expired_retention", 
                               f"Retention expired {-days_remaining} days ago")
        
        return None  # No violation
        
    except s3_client.exceptions.NoSuchKey:
        return Violation("P0", key, "object_not_found", "Object does not exist")
    except Exception as e:
        if "ObjectLockConfigurationNotFoundError" in str(e):
            return Violation("P0", key, "no_object_lock", "Bucket does not have Object Lock enabled")
        return Violation("P1", key, "verification_error", str(e))


def verify_manifest_integrity(s3_client, bucket: str, manifest_key: str) -> List[Violation]:
    """Verify manifest and all referenced objects"""
    violations = []
    
    try:
        # Download manifest
        response = s3_client.get_object(Bucket=bucket, Key=manifest_key)
        manifest = json.loads(response['Body'].read().decode('utf-8'))
        
        # Verify schema version
        if manifest.get("schema") != "origin.backup.schema.v1":
            violations.append(Violation("P1", manifest_key, "invalid_schema",
                                       f"Unknown schema: {manifest.get('schema')}"))
        
        # Verify immutability settings in manifest
        immutability = manifest.get("immutability", {})
        if not immutability.get("object_lock"):
            violations.append(Violation("P0", manifest_key, "manifest_no_lock",
                                       "Manifest claims no Object Lock"))
        
        if immutability.get("mode") != REQUIRED_LOCK_MODE:
            violations.append(Violation("P0", manifest_key, "manifest_wrong_mode",
                                       f"Manifest claims mode: {immutability.get('mode')}"))
        
        if immutability.get("retention_days", 0) < MIN_RETENTION_DAYS:
            violations.append(Violation("P1", manifest_key, "insufficient_retention",
                                       f"Retention {immutability.get('retention_days')} < {MIN_RETENTION_DAYS}"))
        
        # Verify each volume's checksum
        backup_prefix = manifest_key.rsplit('/', 1)[0]
        for vol in manifest.get("volumes", []):
            if vol.get("missing"):
                continue
            
            vol_key = f"{backup_prefix}/{vol['filename']}"
            expected_sha256 = vol.get("sha256")
            
            if expected_sha256:
                # Download and verify checksum
                try:
                    vol_response = s3_client.get_object(Bucket=bucket, Key=vol_key)
                    content = vol_response['Body'].read()
                    actual_sha256 = hashlib.sha256(content).hexdigest()
                    
                    if actual_sha256 != expected_sha256:
                        violations.append(Violation("P0", vol_key, "checksum_mismatch",
                                                   f"Expected {expected_sha256[:16]}..., got {actual_sha256[:16]}..."))
                except Exception as e:
                    violations.append(Violation("P1", vol_key, "checksum_error", str(e)))
        
        return violations
        
    except Exception as e:
        return [Violation("P0", manifest_key, "manifest_error", str(e))]


def verify_bucket_policy(s3_client, bucket: str) -> List[Violation]:
    """Verify bucket has correct security policy"""
    violations = []
    
    try:
        # Check Object Lock configuration
        lock_config = s3_client.get_object_lock_configuration(Bucket=bucket)
        if lock_config.get("ObjectLockConfiguration", {}).get("ObjectLockEnabled") != "Enabled":
            violations.append(Violation("P0", bucket, "bucket_no_lock",
                                       "Bucket does not have Object Lock enabled"))
    except Exception as e:
        if "ObjectLockConfigurationNotFoundError" in str(e):
            violations.append(Violation("P0", bucket, "bucket_no_lock",
                                       "Bucket does not have Object Lock enabled"))
    
    return violations


# =============================================================================
# REPORTING
# =============================================================================

def alert_codex(violations: List[Violation]):
    """Send violations to Codex for action"""
    if not violations:
        return
    
    p0_violations = [v for v in violations if v.severity == "P0"]
    
    if p0_violations:
        logger.critical(f"🚨 {len(p0_violations)} P0 VIOLATIONS DETECTED")
        
        try:
            requests.post(
                f"{CODEX_URL}/api/alerts",
                json={
                    "source": "sentinel",
                    "severity": "P0",
                    "violations": [v.to_dict() for v in p0_violations],
                    "action_required": "abort_and_investigate"
                },
                timeout=10
            )
        except Exception as e:
            logger.error(f"Failed to alert Codex: {e}")


def write_audit_log(report: dict):
    """Write verification report to Vault audit log"""
    try:
        requests.post(
            f"{VAULT_URL}/api/audit",
            json={
                "source": "sentinel",
                "event": "backup_verification",
                "report": report,
            },
            timeout=10
        )
    except Exception as e:
        logger.warning(f"Failed to write audit log: {e}")


# =============================================================================
# MAIN VERIFICATION LOOP
# =============================================================================

def run_verification() -> dict:
    """Run full verification cycle"""
    logger.info("=" * 60)
    logger.info("SENTINEL VERIFICATION STARTED")
    logger.info("=" * 60)
    
    s3_client = get_s3_client()
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bucket": BACKUP_BUCKET,
        "objects_checked": 0,
        "manifests_verified": 0,
        "violations": [],
        "status": "PASS"
    }
    
    # Step 1: Verify bucket configuration
    logger.info("🔐 Verifying bucket configuration...")
    bucket_violations = verify_bucket_policy(s3_client, BACKUP_BUCKET)
    report["violations"].extend([v.to_dict() for v in bucket_violations])
    
    if bucket_violations:
        logger.error(f"   ❌ {len(bucket_violations)} bucket violations")
        report["status"] = "FAIL"
    else:
        logger.info("   ✅ Bucket configuration OK")
    
    # Step 2: List all objects
    logger.info("📋 Listing backup objects...")
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        all_objects = []
        for page in paginator.paginate(Bucket=BACKUP_BUCKET):
            all_objects.extend(page.get('Contents', []))
        
        logger.info(f"   Found {len(all_objects)} objects")
        report["objects_checked"] = len(all_objects)
    except Exception as e:
        logger.error(f"   Failed to list objects: {e}")
        report["violations"].append({
            "severity": "P0",
            "violation_type": "list_failed",
            "details": str(e)
        })
        report["status"] = "FAIL"
        return report
    
    # Step 3: Verify Object Lock on each object
    logger.info("🔒 Verifying Object Lock on all objects...")
    lock_violations = 0
    for obj in all_objects:
        violation = verify_object_lock(s3_client, BACKUP_BUCKET, obj['Key'])
        if violation:
            report["violations"].append(violation.to_dict())
            lock_violations += 1
            report["status"] = "FAIL"
    
    if lock_violations:
        logger.error(f"   ❌ {lock_violations} objects missing proper Object Lock")
    else:
        logger.info(f"   ✅ All {len(all_objects)} objects have valid Object Lock")
    
    # Step 4: Verify manifests and checksums
    logger.info("📦 Verifying backup manifests and checksums...")
    manifests = [obj['Key'] for obj in all_objects if obj['Key'].endswith('manifest.json')]
    
    for manifest_key in manifests:
        logger.info(f"   Checking {manifest_key}...")
        manifest_violations = verify_manifest_integrity(s3_client, BACKUP_BUCKET, manifest_key)
        report["violations"].extend([v.to_dict() for v in manifest_violations])
        report["manifests_verified"] += 1
        
        if manifest_violations:
            logger.error(f"      ❌ {len(manifest_violations)} violations")
            report["status"] = "FAIL"
        else:
            logger.info(f"      ✅ Manifest OK")
    
    # Step 5: Generate summary
    logger.info("=" * 60)
    p0_count = len([v for v in report["violations"] if v.get("severity") == "P0"])
    p1_count = len([v for v in report["violations"] if v.get("severity") == "P1"])
    
    if report["status"] == "PASS":
        logger.info("✅ VERIFICATION PASSED")
    else:
        logger.error(f"❌ VERIFICATION FAILED: {p0_count} P0, {p1_count} P1 violations")
    
    logger.info(f"   Objects checked: {report['objects_checked']}")
    logger.info(f"   Manifests verified: {report['manifests_verified']}")
    logger.info("=" * 60)
    
    return report


def main():
    """Main entry point - runs continuous verification loop"""
    logger.info("🛡️  SENTINEL STARTING")
    logger.info(f"   Bucket: {BACKUP_BUCKET}")
    logger.info(f"   Endpoint: {S3_ENDPOINT}")
    logger.info(f"   Check interval: {CHECK_INTERVAL}s")
    
    if not S3_KEY or not S3_SECRET:
        logger.error("S3_KEY and S3_SECRET must be set")
        sys.exit(1)
    
    while True:
        try:
            report = run_verification()
            
            # Alert on violations
            violations = [Violation(**v) if isinstance(v, dict) else v 
                         for v in report.get("violations", [])]
            alert_codex([v for v in violations if isinstance(v, Violation)])
            
            # Write audit log
            write_audit_log(report)
            
            # Print report
            print(json.dumps(report, indent=2))
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
        
        # Sleep until next check
        logger.info(f"Next verification in {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    # Allow single run mode for testing
    if os.getenv("SINGLE_RUN"):
        report = run_verification()
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["status"] == "PASS" else 1)
    else:
        main()
