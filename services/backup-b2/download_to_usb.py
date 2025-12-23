#!/usr/bin/env python3
"""
Download backups from Backblaze B2 and copy to USB drive
Runs inside a container
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET = os.environ.get("B2_BUCKET", "origin-os-backup")

OUTPUT_DIR = "/output"  # USB drive mounted here


def run_cmd(cmd):
    """Run command and return output"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


def main():
    print("=" * 60)
    print("DOWNLOAD B2 BACKUPS TO USB")
    print("=" * 60)
    print()
    
    # Authorize B2
    print("Authorizing with B2...")
    ok, _, err = run_cmd(["b2", "authorize-account", B2_KEY_ID, B2_APP_KEY])
    if not ok:
        print(f"Auth failed: {err}")
        return 1
    print("  Authorized")
    print()
    
    # List all files in bucket
    print("Listing files in bucket...")
    ok, stdout, _ = run_cmd(["b2", "ls", "--recursive", f"b2://{B2_BUCKET}"])
    if not ok:
        print("Failed to list files")
        return 1
    
    files = [f.strip() for f in stdout.strip().split('\n') if f.strip()]
    print(f"  Found {len(files)} files")
    print()
    
    # Create backup directory on USB
    backup_dir = os.path.join(OUTPUT_DIR, f"origin_backup_{datetime.now().strftime('%Y%m%d')}")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Download each file
    print("Downloading files...")
    downloaded = 0
    for f in files:
        # Create subdirectories
        local_path = os.path.join(backup_dir, f)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        print(f"  {f}")
        ok, _, err = run_cmd([
            "b2", "file", "download",
            f"b2://{B2_BUCKET}/{f}",
            local_path
        ])
        if ok:
            downloaded += 1
        else:
            print(f"    FAILED: {err}")
    
    print()
    print("=" * 60)
    print(f"COMPLETE: {downloaded}/{len(files)} files downloaded")
    print(f"Location: {backup_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
