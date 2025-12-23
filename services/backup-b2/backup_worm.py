#!/usr/bin/env python3
"""
ORIGIN OS BACKUP — v1.1 (LOCKED)
================================
Immutable backup with Object Lock via S3 API (boto3)
"""

import os
import sys
import json
import tarfile
import hashlib
import uuid
import boto3
from datetime import datetime, timezone, timedelta
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
S3_BUCKET = os.environ.get("S3_BUCKET", "origin-os-backups")

RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "365"))
OBJECT_LOCK_MODE = os.environ.get("OBJECT_LOCK_MODE", "COMPLIANCE")

RUNNER = os.environ.get("RUNNER", "backup-container")
HOSTNAME = os.environ.get("HOSTNAME", "unknown")

VOLUME_MOUNTS = {
    "origin-vault-data": "/volumes/vault",
    "origin-memory-data": "/volumes/memory",
    "origin-registry-data": "/volumes/registry",
    "origin-orchestrator-data": "/volumes/orchestrator",
    "origin-cad-data": "/volumes/cad",
    "origin-mcp-data": "/volumes/mcp",
    "origin-backup-data": "/volumes/backup",
}

CONFIG_MOUNTS = {
    "memory.jsonl": "/configs/memory.jsonl",
    "claude_desktop_config.json": "/configs/claude_desktop_config.json",
    "env.txt": "/configs/env.txt",
    "docker-compose.yml": "/configs/docker-compose.yml",
    "cursor_mcp.json": "/configs/cursor_mcp.json",
}

OUTPUT_DIR = "/output"

# =============================================================================
# S3 CLIENT
# =============================================================================

def get_s3_client():
    """Create S3 client for Backblaze"""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


# =============================================================================
# UTILITIES
# =============================================================================

def sha256_file(filepath: str) -> str:
    """Calculate SHA256 hash of a file"""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def get_file_size(filepath: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(filepath)


def upload_with_object_lock(s3_client, local_path: str, s3_key: str, retention_date: datetime) -> bool:
    """Upload file to S3 with Object Lock retention"""
    try:
        with open(local_path, 'rb') as f:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=f,
                ObjectLockMode=OBJECT_LOCK_MODE,
                ObjectLockRetainUntilDate=retention_date,
            )
        return True
    except Exception as e:
        print(f"    UPLOAD FAILED: {e}")
        return False


# =============================================================================
# BACKUP FUNCTIONS
# =============================================================================

def backup_volume(name: str, mount_path: str, output_dir: str) -> dict:
    """Backup a mounted volume to a tarball with integrity check"""
    result = {"name": name}
    
    if not os.path.exists(mount_path):
        result["missing"] = True
        return result
    
    try:
        contents = os.listdir(mount_path)
        if not contents:
            result["missing"] = True
            return result
    except PermissionError:
        result["missing"] = True
        return result
    
    tar_path = os.path.join(output_dir, f"{name}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(mount_path, arcname=name)
    
    size = get_file_size(tar_path)
    sha256 = sha256_file(tar_path)
    
    print(f"  {name}: {size} bytes, sha256:{sha256[:16]}...")
    
    result["size_bytes"] = size
    result["sha256"] = sha256
    result["filename"] = f"{name}.tar.gz"
    
    return result


def backup_config(name: str, mount_path: str, output_dir: str) -> dict:
    """Backup a config file with integrity check"""
    result = {"path": name}
    
    if not os.path.exists(mount_path):
        print(f"  {name}: not found")
        result["missing"] = True
        return result
    
    import shutil
    dest_path = os.path.join(output_dir, "configs", name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(mount_path, dest_path)
    
    sha256 = sha256_file(dest_path)
    print(f"  {name}: sha256:{sha256[:16]}...")
    
    result["sha256"] = sha256
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    retention_date = datetime.now(timezone.utc) + timedelta(days=RETENTION_DAYS)
    
    print("=" * 60)
    print("ORIGIN OS BACKUP — v1.1 (WORM)")
    print("=" * 60)
    print(f"Backup ID:  {backup_id}")
    print(f"Timestamp:  {timestamp}")
    print(f"Retention:  {RETENTION_DAYS} days ({OBJECT_LOCK_MODE})")
    print(f"Bucket:     {S3_BUCKET}")
    print("=" * 60)
    print()
    
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("ERROR: AWS credentials required")
        return 1
    
    # Create S3 client
    s3_client = get_s3_client()
    
    # Create output directory
    backup_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(backup_dir, "configs"), exist_ok=True)
    
    # =========================================================================
    # BACKUP VOLUMES
    # =========================================================================
    print("📦 Backing up volumes...")
    volume_results = []
    
    for name, mount_path in VOLUME_MOUNTS.items():
        result = backup_volume(name, mount_path, backup_dir)
        volume_results.append(result)
    
    print()
    
    # =========================================================================
    # BACKUP CONFIGS
    # =========================================================================
    print("📄 Backing up configs...")
    config_results = []
    
    for name, mount_path in CONFIG_MOUNTS.items():
        result = backup_config(name, mount_path, backup_dir)
        config_results.append(result)
    
    print()
    
    # =========================================================================
    # CREATE MANIFEST
    # =========================================================================
    print("📋 Creating manifest...")
    
    manifest = {
        "schema": "origin.backup.schema.v1",
        "backup_id": backup_id,
        "timestamp": timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage": {
            "type": "s3",
            "location": f"s3://{S3_BUCKET}/backups/{timestamp}/",
            "endpoint": S3_ENDPOINT,
        },
        "immutability": {
            "object_lock": True,
            "mode": OBJECT_LOCK_MODE,
            "retention_days": RETENTION_DAYS,
            "retain_until": retention_date.isoformat(),
        },
        "source": {
            "container": RUNNER,
            "host": HOSTNAME,
            "runner": "backup-container",
        },
        "volumes": volume_results,
        "configs": config_results,
        "verification": {"verified": False},
    }
    
    manifest_path = os.path.join(backup_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    manifest_sha256 = sha256_file(manifest_path)
    print(f"  manifest.json: sha256:{manifest_sha256[:16]}...")
    print()
    
    # =========================================================================
    # UPLOAD TO S3 WITH OBJECT LOCK
    # =========================================================================
    print("☁️  Uploading to S3 with Object Lock...")
    
    uploaded = 0
    total = 0
    
    # Upload volumes
    for vol in volume_results:
        if "filename" in vol:
            total += 1
            local_path = os.path.join(backup_dir, vol["filename"])
            s3_key = f"backups/{timestamp}/{vol['filename']}"
            print(f"  → {vol['filename']}")
            if upload_with_object_lock(s3_client, local_path, s3_key, retention_date):
                uploaded += 1
    
    # Upload configs tarball
    configs_tar = os.path.join(backup_dir, "configs.tar.gz")
    with tarfile.open(configs_tar, "w:gz") as tar:
        tar.add(os.path.join(backup_dir, "configs"), arcname="configs")
    
    total += 1
    print(f"  → configs.tar.gz")
    if upload_with_object_lock(s3_client, configs_tar, f"backups/{timestamp}/configs.tar.gz", retention_date):
        uploaded += 1
    
    # Upload manifest
    total += 1
    print(f"  → manifest.json")
    if upload_with_object_lock(s3_client, manifest_path, f"backups/{timestamp}/manifest.json", retention_date):
        uploaded += 1
    
    print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 60)
    if uploaded == total:
        print(f"✅ BACKUP COMPLETE: {uploaded}/{total} files uploaded")
    else:
        print(f"⚠️  BACKUP PARTIAL: {uploaded}/{total} files uploaded")
    print(f"   Backup ID: {backup_id}")
    print(f"   Location:  s3://{S3_BUCKET}/backups/{timestamp}/")
    print(f"   Locked until: {retention_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    return 0 if uploaded == total else 1


if __name__ == "__main__":
    sys.exit(main())
