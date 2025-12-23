#!/usr/bin/env python3
"""
SENTINEL — Origin OS Security Guardian v2
==========================================
Always-on verification service with one-way alert flow:
- Monitors backup integrity
- Verifies Object Lock compliance
- Writes structured alerts to Vault
- CANNOT read schemas or policies
- CANNOT receive callbacks

Information flows: Sentinel → Vault → Codex
Control flows: NONE (by design)
"""

import boto3
import hashlib
import json
import os
import sys
import time
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

# =============================================================================
# CONFIGURATION
# =============================================================================

SENTINEL_ID = "sentinel-v1"

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
S3_KEY = os.getenv("S3_KEY")
S3_SECRET = os.getenv("S3_SECRET")
BACKUP_BUCKET = os.getenv("BACKUP_BUCKET", "origin-os-backups")

# Alert output directory (mounted from Vault)
ALERT_OUTPUT_DIR = os.getenv("ALERT_OUTPUT_DIR", "/vault/sentinel/alerts")

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "3600"))

REQUIRED_LOCK_MODE = "COMPLIANCE"
MIN_RETENTION_DAYS = 365

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SENTINEL] %(levelname)s: %(message)s'
)
logger = logging.getLogger("sentinel")

# =============================================================================
# ALERT OUTPUT (Immutable Contract)
# =============================================================================

class SentinelAlert:
    """
    Immutable alert structure matching sentinel_alert.schema.yaml
    Sentinel can only WRITE these. Cannot read responses.
    """
    def __init__(
        self,
        severity: str,
        domain: str,
        finding: str,
        obj: str,
        details: dict = None,
        checksum_verified: bool = False,
        correlation_id: str = None,
        source_backup_id: str = None,
    ):
        self.sentinel_id = SENTINEL_ID
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.severity = severity  # P0, P1, P2
        self.domain = domain
        self.finding = finding
        self.object = obj
        self.details = details or {}
        self.checksum_verified = checksum_verified
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.source_backup_id = source_backup_id
        
    def to_dict(self) -> dict:
        return {
            "sentinel_id": self.sentinel_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "domain": self.domain,
            "finding": self.finding,
            "object": self.object,
            "details": self.details,
            "checksum_verified": self.checksum_verified,
            "correlation_id": self.correlation_id,
            "source_backup_id": self.source_backup_id,
        }
    
    def write_to_vault(self):
        """
        Write alert to Vault alert directory.
        This is ONE-WAY. Sentinel cannot read from Vault.
        """
        os.makedirs(ALERT_OUTPUT_DIR, exist_ok=True)
        
        # Filename includes timestamp and correlation ID for uniqueness
        filename = f"{self.timestamp.replace(':', '-')}_{self.correlation_id[:8]}.json"
        filepath = os.path.join(ALERT_OUTPUT_DIR, filename)
        
        alert_data = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        logger.info(f"Alert written: {filepath}")
        return filepath


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

def verify_object_lock(s3_client, bucket: str, key: str) -> Optional[SentinelAlert]:
    """Verify object has valid Object Lock retention"""
    try:
        response = s3_client.get_object_retention(Bucket=bucket, Key=key)
        retention = response.get("Retention", {})
        mode = retention.get("Mode")
        retain_until = retention.get("RetainUntilDate")
        
        if not mode:
            return SentinelAlert(
                severity="P0",
                domain="object_lock",
                finding="object_lock_missing",
                obj=key,
                details={"expected_lock": REQUIRED_LOCK_MODE, "actual_lock": None}
            )
        
        if mode != REQUIRED_LOCK_MODE:
            return SentinelAlert(
                severity="P0",
                domain="object_lock",
                finding="object_lock_wrong_mode",
                obj=key,
                details={"expected_lock": REQUIRED_LOCK_MODE, "actual_lock": mode}
            )
        
        if retain_until:
            now = datetime.now(timezone.utc)
            days_remaining = (retain_until.replace(tzinfo=timezone.utc) - now).days
            if days_remaining < 0:
                return SentinelAlert(
                    severity="P0",
                    domain="retention",
                    finding="retention_expired",
                    obj=key,
                    details={"retention_days": days_remaining}
                )
        
        return None
        
    except Exception as e:
        if "ObjectLockConfigurationNotFoundError" in str(e):
            return SentinelAlert(
                severity="P0",
                domain="bucket_policy",
                finding="bucket_no_lock",
                obj=key,
                details={"error_message": str(e)}
            )
        return SentinelAlert(
            severity="P1",
            domain="backup_integrity",
            finding="policy_violation",
            obj=key,
            details={"error_message": str(e)}
        )


def verify_manifest_integrity(s3_client, bucket: str, manifest_key: str) -> List[SentinelAlert]:
    """Verify manifest and all referenced objects"""
    alerts = []
    correlation_id = str(uuid.uuid4())
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=manifest_key)
        manifest = json.loads(response['Body'].read().decode('utf-8'))
        backup_id = manifest.get("backup_id")
        
        # Verify schema version
        if manifest.get("schema") != "origin.backup.schema.v1":
            alerts.append(SentinelAlert(
                severity="P1",
                domain="schema_validation",
                finding="schema_invalid",
                obj=manifest_key,
                details={"expected_schema": "origin.backup.schema.v1", "actual_schema": manifest.get("schema")},
                correlation_id=correlation_id,
                source_backup_id=backup_id
            ))
        
        # Verify immutability settings
        immutability = manifest.get("immutability", {})
        if not immutability.get("object_lock"):
            alerts.append(SentinelAlert(
                severity="P0",
                domain="object_lock",
                finding="object_lock_missing",
                obj=manifest_key,
                details={"expected_lock": True, "actual_lock": False},
                correlation_id=correlation_id,
                source_backup_id=backup_id
            ))
        
        if immutability.get("mode") != REQUIRED_LOCK_MODE:
            alerts.append(SentinelAlert(
                severity="P0",
                domain="object_lock",
                finding="object_lock_wrong_mode",
                obj=manifest_key,
                details={"expected_lock": REQUIRED_LOCK_MODE, "actual_lock": immutability.get("mode")},
                correlation_id=correlation_id,
                source_backup_id=backup_id
            ))
        
        if immutability.get("retention_days", 0) < MIN_RETENTION_DAYS:
            alerts.append(SentinelAlert(
                severity="P1",
                domain="retention",
                finding="retention_insufficient",
                obj=manifest_key,
                details={"retention_days": immutability.get("retention_days"), "minimum_required": MIN_RETENTION_DAYS},
                correlation_id=correlation_id,
                source_backup_id=backup_id
            ))
        
        # Verify each volume's checksum
        backup_prefix = manifest_key.rsplit('/', 1)[0]
        for vol in manifest.get("volumes", []):
            if vol.get("missing"):
                continue
            
            vol_key = f"{backup_prefix}/{vol['filename']}"
            expected_sha256 = vol.get("sha256")
            
            if expected_sha256:
                try:
                    vol_response = s3_client.get_object(Bucket=bucket, Key=vol_key)
                    content = vol_response['Body'].read()
                    actual_sha256 = hashlib.sha256(content).hexdigest()
                    
                    if actual_sha256 != expected_sha256:
                        alerts.append(SentinelAlert(
                            severity="P0",
                            domain="checksum",
                            finding="checksum_mismatch",
                            obj=vol_key,
                            details={
                                "expected_checksum": expected_sha256,
                                "actual_checksum": actual_sha256
                            },
                            checksum_verified=False,
                            correlation_id=correlation_id,
                            source_backup_id=backup_id
                        ))
                except Exception as e:
                    alerts.append(SentinelAlert(
                        severity="P1",
                        domain="checksum",
                        finding="policy_violation",
                        obj=vol_key,
                        details={"error_message": str(e)},
                        correlation_id=correlation_id,
                        source_backup_id=backup_id
                    ))
        
        return alerts
        
    except Exception as e:
        return [SentinelAlert(
            severity="P0",
            domain="backup_integrity",
            finding="manifest_corrupted",
            obj=manifest_key,
            details={"error_message": str(e)},
            correlation_id=correlation_id
        )]


def verify_bucket_policy(s3_client, bucket: str) -> List[SentinelAlert]:
    """Verify bucket has correct security policy"""
    alerts = []
    
    try:
        lock_config = s3_client.get_object_lock_configuration(Bucket=bucket)
        if lock_config.get("ObjectLockConfiguration", {}).get("ObjectLockEnabled") != "Enabled":
            alerts.append(SentinelAlert(
                severity="P0",
                domain="bucket_policy",
                finding="bucket_no_lock",
                obj=bucket,
                details={"error_message": "Object Lock not enabled on bucket"}
            ))
    except Exception as e:
        if "ObjectLockConfigurationNotFoundError" in str(e):
            alerts.append(SentinelAlert(
                severity="P0",
                domain="bucket_policy",
                finding="bucket_no_lock",
                obj=bucket,
                details={"error_message": str(e)}
            ))
    
    return alerts


# =============================================================================
# MAIN VERIFICATION LOOP
# =============================================================================

def run_verification() -> dict:
    """Run full verification cycle"""
    logger.info("=" * 60)
    logger.info("SENTINEL VERIFICATION STARTED")
    logger.info("=" * 60)
    
    s3_client = get_s3_client()
    all_alerts = []
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentinel_id": SENTINEL_ID,
        "bucket": BACKUP_BUCKET,
        "objects_checked": 0,
        "manifests_verified": 0,
        "alerts_generated": 0,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        "status": "PASS"
    }
    
    # Step 1: Verify bucket configuration
    logger.info("🔐 Verifying bucket configuration...")
    bucket_alerts = verify_bucket_policy(s3_client, BACKUP_BUCKET)
    all_alerts.extend(bucket_alerts)
    
    if bucket_alerts:
        logger.error(f"   ❌ {len(bucket_alerts)} bucket violations")
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
        all_alerts.append(SentinelAlert(
            severity="P0",
            domain="backup_integrity",
            finding="policy_violation",
            obj=BACKUP_BUCKET,
            details={"error_message": str(e)}
        ))
        report["status"] = "FAIL"
    
    # Step 3: Verify Object Lock on each object
    if all_objects:
        logger.info("🔒 Verifying Object Lock on all objects...")
        for obj in all_objects:
            alert = verify_object_lock(s3_client, BACKUP_BUCKET, obj['Key'])
            if alert:
                all_alerts.append(alert)
                report["status"] = "FAIL"
        
        lock_violations = len([a for a in all_alerts if a.domain == "object_lock"])
        if lock_violations:
            logger.error(f"   ❌ {lock_violations} objects missing proper Object Lock")
        else:
            logger.info(f"   ✅ All {len(all_objects)} objects have valid Object Lock")
    
    # Step 4: Verify manifests and checksums
    logger.info("📦 Verifying backup manifests and checksums...")
    manifests = [obj['Key'] for obj in all_objects if obj['Key'].endswith('manifest.json')]
    
    for manifest_key in manifests:
        logger.info(f"   Checking {manifest_key}...")
        manifest_alerts = verify_manifest_integrity(s3_client, BACKUP_BUCKET, manifest_key)
        all_alerts.extend(manifest_alerts)
        report["manifests_verified"] += 1
        
        if manifest_alerts:
            logger.error(f"      ❌ {len(manifest_alerts)} violations")
            report["status"] = "FAIL"
        else:
            logger.info(f"      ✅ Manifest OK")
    
    # Step 5: Write all alerts to Vault
    logger.info("📝 Writing alerts to Vault...")
    for alert in all_alerts:
        alert.write_to_vault()
        
        if alert.severity == "P0":
            report["p0_count"] += 1
        elif alert.severity == "P1":
            report["p1_count"] += 1
        else:
            report["p2_count"] += 1
    
    report["alerts_generated"] = len(all_alerts)
    
    # Step 6: Summary
    logger.info("=" * 60)
    if report["status"] == "PASS":
        logger.info("✅ VERIFICATION PASSED")
    else:
        logger.error(f"❌ VERIFICATION FAILED: {report['p0_count']} P0, {report['p1_count']} P1")
    
    logger.info(f"   Objects checked: {report['objects_checked']}")
    logger.info(f"   Manifests verified: {report['manifests_verified']}")
    logger.info(f"   Alerts generated: {report['alerts_generated']}")
    logger.info("=" * 60)
    
    return report


def main():
    """Main entry point"""
    logger.info("🛡️  SENTINEL STARTING")
    logger.info(f"   ID: {SENTINEL_ID}")
    logger.info(f"   Bucket: {BACKUP_BUCKET}")
    logger.info(f"   Alert output: {ALERT_OUTPUT_DIR}")
    logger.info(f"   Check interval: {CHECK_INTERVAL}s")
    
    if not S3_KEY or not S3_SECRET:
        logger.error("S3_KEY and S3_SECRET must be set")
        sys.exit(1)
    
    while True:
        try:
            report = run_verification()
            print(json.dumps(report, indent=2))
        except Exception as e:
            logger.error(f"Verification failed: {e}")
        
        logger.info(f"Next verification in {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    if os.getenv("SINGLE_RUN"):
        report = run_verification()
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["status"] == "PASS" else 1)
    else:
        main()
