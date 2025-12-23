#!/usr/bin/env python3
"""
Origin OS Secure Backup to Backblaze B2
Runs inside a container with mounted volumes
"""

import os
import sys
import json
import tarfile
from datetime import datetime
from pathlib import Path
import subprocess

# Backblaze B2 credentials (passed via env)
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET = os.environ.get("B2_BUCKET", "origin-os-backup")

# Volumes mounted into this container at /volumes/<name>
VOLUME_MOUNTS = {
    "vault": "/volumes/vault",
    "memory": "/volumes/memory",
    "registry": "/volumes/registry",
    "orchestrator": "/volumes/orchestrator",
    "cad": "/volumes/cad",
    "mcp": "/volumes/mcp",
    "backup": "/volumes/backup",
}

# Config files mounted at /configs
CONFIG_MOUNT = "/configs"

# Output directory for tarballs
OUTPUT_DIR = "/output"


def install_b2():
    """Install B2 CLI"""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "b2"], check=True)


def authorize_b2():
    """Authorize with Backblaze"""
    result = subprocess.run(
        ["b2", "authorize-account", B2_KEY_ID, B2_APP_KEY],
        capture_output=True, text=True
    )
    return result.returncode == 0


def backup_volume(name: str, mount_path: str, timestamp: str) -> str:
    """Backup a mounted volume to a tarball"""
    if not os.path.exists(mount_path):
        print(f"  {name}: mount not found at {mount_path}")
        return None
    
    # Check if there's any data
    try:
        contents = os.listdir(mount_path)
        if not contents:
            print(f"  {name}: empty")
            return None
    except PermissionError:
        print(f"  {name}: permission denied")
        return None
    
    tar_path = f"{OUTPUT_DIR}/volume_{name}_{timestamp}.tar.gz"
    
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(mount_path, arcname=name)
    
    size = os.path.getsize(tar_path)
    print(f"  {name}: {size / 1024:.1f} KB")
    return tar_path


def backup_configs(timestamp: str) -> str:
    """Backup config files to a tarball"""
    if not os.path.exists(CONFIG_MOUNT):
        print("  configs: mount not found")
        return None
    
    tar_path = f"{OUTPUT_DIR}/configs_{timestamp}.tar.gz"
    
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(CONFIG_MOUNT, arcname="configs")
    
    size = os.path.getsize(tar_path)
    print(f"  configs: {size / 1024:.1f} KB")
    return tar_path


def upload_to_b2(local_path: str, b2_path: str) -> bool:
    """Upload file to B2"""
    result = subprocess.run(
        ["b2", "file", "upload", B2_BUCKET, local_path, b2_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"    Upload failed: {result.stderr}")
    return result.returncode == 0


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("ORIGIN OS BACKUP TO BACKBLAZE B2")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)
    print()
    
    # Validate credentials
    if not B2_KEY_ID or not B2_APP_KEY:
        print("ERROR: B2_KEY_ID and B2_APP_KEY must be set")
        return 1
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Install and authorize B2
    print("Setting up B2 CLI...")
    install_b2()
    if not authorize_b2():
        print("ERROR: Failed to authorize with B2")
        return 1
    print("  Authorized with Backblaze B2")
    print()
    
    # Backup volumes
    print("Backing up volumes...")
    uploaded = []
    
    for name, mount_path in VOLUME_MOUNTS.items():
        tar_path = backup_volume(name, mount_path, timestamp)
        if tar_path:
            b2_path = f"volumes/{timestamp}/volume_{name}.tar.gz"
            print(f"    Uploading {name}...")
            if upload_to_b2(tar_path, b2_path):
                uploaded.append(b2_path)
    print()
    
    # Backup configs
    print("Backing up configs...")
    tar_path = backup_configs(timestamp)
    if tar_path:
        b2_path = f"configs/{timestamp}/configs.tar.gz"
        print("    Uploading configs...")
        if upload_to_b2(tar_path, b2_path):
            uploaded.append(b2_path)
    print()
    
    # Create manifest
    print("Creating manifest...")
    manifest = {
        "timestamp": timestamp,
        "files": uploaded,
        "bucket": B2_BUCKET
    }
    manifest_path = f"{OUTPUT_DIR}/manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    upload_to_b2(manifest_path, f"manifests/{timestamp}.json")
    
    print()
    print("=" * 60)
    print(f"BACKUP COMPLETE: {len(uploaded)} files uploaded")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
