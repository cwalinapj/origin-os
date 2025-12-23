#!/usr/bin/env python3
"""
Origin OS Backup Service
Automated backup of Docker volumes to local storage and optional S3
"""

import os
import json
import subprocess
import tarfile
import hashlib
import shutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import asyncio
import schedule
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx
import uvicorn

app = FastAPI(title="Origin OS Backup Service", version="1.0")

# =============================================================================
# CONFIGURATION
# =============================================================================

BACKUP_DIR = os.getenv("BACKUP_DIR", "/backups")
RETENTION_DAYS = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
BACKUP_SCHEDULE = os.getenv("BACKUP_SCHEDULE", "daily")  # hourly, daily, weekly

# Remote Backup Configuration (optional - choose one)
# Options: local, gdrive, dropbox, backblaze, rsync
REMOTE_BACKUP_TYPE = os.getenv("REMOTE_BACKUP_TYPE", "local")  # local = no remote

# Google Drive (FREE 15GB)
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID", "")

# Dropbox (FREE 2GB, cheap pro)
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN", "")
DROPBOX_PATH = os.getenv("DROPBOX_PATH", "/origin-os-backups")

# Backblaze B2 (CHEAPEST - $0.005/GB/month vs S3 $0.023)
B2_KEY_ID = os.getenv("B2_KEY_ID", "")
B2_APP_KEY = os.getenv("B2_APP_KEY", "")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "")

# Rsync to remote server (FREE if you have a server)
RSYNC_DEST = os.getenv("RSYNC_DEST", "")  # user@host:/path/to/backups

# Local external drive
EXTERNAL_BACKUP_PATH = os.getenv("EXTERNAL_BACKUP_PATH", "")  # /mnt/external/backups

# Legacy S3 (expensive but supported)
S3_ENABLED = os.getenv("S3_BACKUP_ENABLED", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BACKUP_BUCKET", "")
S3_PREFIX = os.getenv("S3_BACKUP_PREFIX", "origin-os-backups")

# Volumes to backup
VOLUMES = [
    {"name": "origin-mcp-data", "description": "MCP Hub data, memory graph"},
    {"name": "origin-cad-data", "description": "Generated STL files"},
    {"name": "origin-vault-data", "description": "Encrypted token envelopes"},
    {"name": "origin-orchestrator-data", "description": "Workflow definitions"},
]

os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(os.path.join(BACKUP_DIR, "daily"), exist_ok=True)
os.makedirs(os.path.join(BACKUP_DIR, "weekly"), exist_ok=True)
os.makedirs(os.path.join(BACKUP_DIR, "monthly"), exist_ok=True)

# Backup state
backup_state = {
    "last_backup": None,
    "last_backup_size": 0,
    "backups_today": 0,
    "total_backups": 0,
    "last_error": None,
    "is_running": False
}

# =============================================================================
# BACKUP FUNCTIONS
# =============================================================================

def get_volume_path(volume_name: str) -> Optional[str]:
    """Get the mount path for a Docker volume"""
    try:
        result = subprocess.run(
            ["docker", "volume", "inspect", volume_name, "--format", "{{.Mountpoint}}"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"Error inspecting volume {volume_name}: {e}")
    return None


def backup_volume(volume_name: str, backup_path: str) -> Dict:
    """Backup a single Docker volume using docker cp"""
    try:
        # Create a temporary container to access the volume
        container_name = f"backup-{volume_name}-{int(time.time())}"
        
        # Run a container with the volume mounted
        subprocess.run([
            "docker", "run", "-d", "--name", container_name,
            "-v", f"{volume_name}:/data:ro",
            "alpine", "sleep", "300"
        ], capture_output=True, check=True, timeout=60)
        
        try:
            # Copy data from container
            temp_dir = os.path.join(backup_path, f"temp_{volume_name}")
            os.makedirs(temp_dir, exist_ok=True)
            
            subprocess.run([
                "docker", "cp", f"{container_name}:/data/.", temp_dir
            ], capture_output=True, check=True, timeout=300)
            
            # Create tarball
            tar_path = os.path.join(backup_path, f"{volume_name}.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(temp_dir, arcname=volume_name)
            
            # Calculate checksum
            with open(tar_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            
            # Get size
            size = os.path.getsize(tar_path)
            
            # Cleanup temp dir
            shutil.rmtree(temp_dir)
            
            return {
                "volume": volume_name,
                "success": True,
                "path": tar_path,
                "size": size,
                "checksum": checksum
            }
            
        finally:
            # Cleanup container
            subprocess.run(["docker", "rm", "-f", container_name], 
                         capture_output=True, timeout=30)
        
    except subprocess.TimeoutExpired:
        return {"volume": volume_name, "success": False, "error": "Timeout"}
    except subprocess.CalledProcessError as e:
        return {"volume": volume_name, "success": False, "error": str(e)}
    except Exception as e:
        return {"volume": volume_name, "success": False, "error": str(e)}


def run_backup(backup_type: str = "daily") -> Dict:
    """Run a full backup of all volumes"""
    global backup_state
    
    if backup_state["is_running"]:
        return {"error": "Backup already in progress"}
    
    backup_state["is_running"] = True
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, backup_type, timestamp)
    os.makedirs(backup_path, exist_ok=True)
    
    results = {
        "timestamp": timestamp,
        "type": backup_type,
        "path": backup_path,
        "volumes": [],
        "total_size": 0,
        "success": True
    }
    
    try:
        for vol in VOLUMES:
            print(f"Backing up {vol['name']}...")
            result = backup_volume(vol["name"], backup_path)
            results["volumes"].append(result)
            
            if result.get("success"):
                results["total_size"] += result.get("size", 0)
            else:
                results["success"] = False
        
        # Write manifest
        manifest = {
            "timestamp": timestamp,
            "type": backup_type,
            "volumes": results["volumes"],
            "total_size": results["total_size"],
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        with open(os.path.join(backup_path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Upload to remote storage if configured
        remote_result = upload_remote(backup_path, backup_type, timestamp)
        results["remote"] = remote_result
        
        # Update state
        backup_state["last_backup"] = timestamp
        backup_state["last_backup_size"] = results["total_size"]
        backup_state["backups_today"] += 1
        backup_state["total_backups"] += 1
        backup_state["last_error"] = None
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        backup_state["last_error"] = str(e)
    
    finally:
        backup_state["is_running"] = False
    
    return results


def upload_to_s3(backup_path: str, backup_type: str, timestamp: str) -> Dict:
    """Upload backup to S3 (expensive - consider alternatives)"""
    try:
        import boto3
        s3 = boto3.client('s3')
        
        uploaded = []
        for filename in os.listdir(backup_path):
            filepath = os.path.join(backup_path, filename)
            if os.path.isfile(filepath):
                s3_key = f"{S3_PREFIX}/{backup_type}/{timestamp}/{filename}"
                s3.upload_file(filepath, S3_BUCKET, s3_key)
                uploaded.append(s3_key)
        
        return {"success": True, "uploaded": uploaded, "bucket": S3_BUCKET, "provider": "s3"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def upload_to_backblaze(backup_path: str, backup_type: str, timestamp: str) -> Dict:
    """Upload to Backblaze B2 - CHEAPEST cloud storage ($0.005/GB/month)"""
    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
        
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        
        bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
        
        uploaded = []
        for filename in os.listdir(backup_path):
            filepath = os.path.join(backup_path, filename)
            if os.path.isfile(filepath):
                b2_path = f"origin-os-backups/{backup_type}/{timestamp}/{filename}"
                bucket.upload_local_file(filepath, b2_path)
                uploaded.append(b2_path)
        
        return {"success": True, "uploaded": uploaded, "bucket": B2_BUCKET_NAME, "provider": "backblaze"}
    
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "backblaze"}


def upload_to_dropbox(backup_path: str, backup_type: str, timestamp: str) -> Dict:
    """Upload to Dropbox - FREE 2GB, cheap pro plans"""
    try:
        import dropbox
        
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
        
        uploaded = []
        for filename in os.listdir(backup_path):
            filepath = os.path.join(backup_path, filename)
            if os.path.isfile(filepath):
                dbx_path = f"{DROPBOX_PATH}/{backup_type}/{timestamp}/{filename}"
                with open(filepath, 'rb') as f:
                    dbx.files_upload(f.read(), dbx_path)
                uploaded.append(dbx_path)
        
        return {"success": True, "uploaded": uploaded, "provider": "dropbox"}
    
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "dropbox"}


def upload_to_gdrive(backup_path: str, backup_type: str, timestamp: str) -> Dict:
    """Upload to Google Drive - FREE 15GB"""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        
        # Requires service account credentials in GOOGLE_APPLICATION_CREDENTIALS env var
        creds = service_account.Credentials.from_service_account_file(
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '/credentials/gdrive.json')
        )
        service = build('drive', 'v3', credentials=creds)
        
        # Create folder for this backup
        folder_metadata = {
            'name': f"{backup_type}_{timestamp}",
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [GDRIVE_FOLDER_ID] if GDRIVE_FOLDER_ID else []
        }
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        folder_id = folder.get('id')
        
        uploaded = []
        for filename in os.listdir(backup_path):
            filepath = os.path.join(backup_path, filename)
            if os.path.isfile(filepath):
                file_metadata = {'name': filename, 'parents': [folder_id]}
                media = MediaFileUpload(filepath, resumable=True)
                file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                uploaded.append(f"gdrive://{file.get('id')}/{filename}")
        
        return {"success": True, "uploaded": uploaded, "folder_id": folder_id, "provider": "gdrive"}
    
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "gdrive"}


def rsync_backup(backup_path: str, backup_type: str, timestamp: str) -> Dict:
    """Rsync to remote server - FREE if you have a server"""
    try:
        dest = f"{RSYNC_DEST}/{backup_type}/{timestamp}/"
        
        # Create remote directory
        subprocess.run(["ssh", RSYNC_DEST.split(':')[0].split('@')[1] if '@' in RSYNC_DEST else RSYNC_DEST.split(':')[0], 
                       f"mkdir -p {RSYNC_DEST.split(':')[1]}/{backup_type}/{timestamp}"], 
                      check=True, timeout=30)
        
        # Rsync files
        result = subprocess.run(
            ["rsync", "-avz", "--progress", f"{backup_path}/", dest],
            capture_output=True, text=True, timeout=600
        )
        
        if result.returncode == 0:
            return {"success": True, "destination": dest, "provider": "rsync"}
        else:
            return {"success": False, "error": result.stderr, "provider": "rsync"}
    
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "rsync"}


def copy_to_external(backup_path: str, backup_type: str, timestamp: str) -> Dict:
    """Copy to external/mounted drive - FREE, most reliable"""
    try:
        if not EXTERNAL_BACKUP_PATH or not os.path.exists(EXTERNAL_BACKUP_PATH):
            return {"success": False, "error": "External path not mounted", "provider": "external"}
        
        dest_path = os.path.join(EXTERNAL_BACKUP_PATH, backup_type, timestamp)
        os.makedirs(dest_path, exist_ok=True)
        
        copied = []
        for filename in os.listdir(backup_path):
            src = os.path.join(backup_path, filename)
            dst = os.path.join(dest_path, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied.append(dst)
        
        return {"success": True, "copied": copied, "destination": dest_path, "provider": "external"}
    
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "external"}


def upload_remote(backup_path: str, backup_type: str, timestamp: str) -> Dict:
    """Upload to configured remote storage"""
    if REMOTE_BACKUP_TYPE == "local" or not REMOTE_BACKUP_TYPE:
        return {"success": True, "provider": "local", "message": "Local only, no remote configured"}
    
    elif REMOTE_BACKUP_TYPE == "backblaze" and B2_KEY_ID:
        return upload_to_backblaze(backup_path, backup_type, timestamp)
    
    elif REMOTE_BACKUP_TYPE == "dropbox" and DROPBOX_ACCESS_TOKEN:
        return upload_to_dropbox(backup_path, backup_type, timestamp)
    
    elif REMOTE_BACKUP_TYPE == "gdrive" and GDRIVE_FOLDER_ID:
        return upload_to_gdrive(backup_path, backup_type, timestamp)
    
    elif REMOTE_BACKUP_TYPE == "rsync" and RSYNC_DEST:
        return rsync_backup(backup_path, backup_type, timestamp)
    
    elif REMOTE_BACKUP_TYPE == "external" and EXTERNAL_BACKUP_PATH:
        return copy_to_external(backup_path, backup_type, timestamp)
    
    elif REMOTE_BACKUP_TYPE == "s3" and S3_ENABLED and S3_BUCKET:
        return upload_to_s3(backup_path, backup_type, timestamp)
    
    else:
        return {"success": False, "error": f"Remote type '{REMOTE_BACKUP_TYPE}' not configured properly"}


def cleanup_old_backups():
    """Remove backups older than retention period"""
    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
    removed = []
    
    for backup_type in ["daily", "weekly", "monthly"]:
        type_path = os.path.join(BACKUP_DIR, backup_type)
        if not os.path.exists(type_path):
            continue
        
        for backup_dir in os.listdir(type_path):
            backup_path = os.path.join(type_path, backup_dir)
            if not os.path.isdir(backup_path):
                continue
            
            try:
                # Parse timestamp from directory name
                backup_time = datetime.strptime(backup_dir, "%Y%m%d_%H%M%S")
                
                # Apply different retention for different types
                if backup_type == "monthly":
                    keep_days = RETENTION_DAYS * 12  # Keep monthly for a year
                elif backup_type == "weekly":
                    keep_days = RETENTION_DAYS * 4   # Keep weekly for 4 months
                else:
                    keep_days = RETENTION_DAYS       # Daily per config
                
                type_cutoff = datetime.now() - timedelta(days=keep_days)
                
                if backup_time < type_cutoff:
                    shutil.rmtree(backup_path)
                    removed.append(backup_path)
                    
            except ValueError:
                continue
    
    return removed


def list_backups() -> List[Dict]:
    """List all available backups"""
    backups = []
    
    for backup_type in ["daily", "weekly", "monthly"]:
        type_path = os.path.join(BACKUP_DIR, backup_type)
        if not os.path.exists(type_path):
            continue
        
        for backup_dir in sorted(os.listdir(type_path), reverse=True):
            backup_path = os.path.join(type_path, backup_dir)
            manifest_path = os.path.join(backup_path, "manifest.json")
            
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    manifest["type"] = backup_type
                    manifest["path"] = backup_path
                    backups.append(manifest)
            else:
                # Calculate size manually
                total_size = sum(
                    os.path.getsize(os.path.join(backup_path, f))
                    for f in os.listdir(backup_path)
                    if os.path.isfile(os.path.join(backup_path, f))
                )
                backups.append({
                    "timestamp": backup_dir,
                    "type": backup_type,
                    "path": backup_path,
                    "total_size": total_size
                })
    
    return backups


def restore_volume(backup_path: str, volume_name: str) -> Dict:
    """Restore a volume from backup"""
    tar_path = os.path.join(backup_path, f"{volume_name}.tar.gz")
    
    if not os.path.exists(tar_path):
        return {"success": False, "error": f"Backup not found: {tar_path}"}
    
    try:
        container_name = f"restore-{volume_name}-{int(time.time())}"
        
        # Create container with volume mounted
        subprocess.run([
            "docker", "run", "-d", "--name", container_name,
            "-v", f"{volume_name}:/data",
            "alpine", "sleep", "300"
        ], capture_output=True, check=True, timeout=60)
        
        try:
            # Extract tarball to temp location
            temp_dir = f"/tmp/restore_{volume_name}"
            os.makedirs(temp_dir, exist_ok=True)
            
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(temp_dir)
            
            # Copy to container
            extracted_path = os.path.join(temp_dir, volume_name)
            subprocess.run([
                "docker", "cp", f"{extracted_path}/.", f"{container_name}:/data/"
            ], capture_output=True, check=True, timeout=300)
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            return {"success": True, "volume": volume_name, "restored_from": backup_path}
            
        finally:
            subprocess.run(["docker", "rm", "-f", container_name],
                         capture_output=True, timeout=30)
    
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# SCHEDULER
# =============================================================================

def schedule_backups():
    """Set up backup schedule"""
    if BACKUP_SCHEDULE == "hourly":
        schedule.every().hour.do(lambda: run_backup("daily"))
    elif BACKUP_SCHEDULE == "daily":
        schedule.every().day.at("02:00").do(lambda: run_backup("daily"))
        schedule.every().sunday.at("03:00").do(lambda: run_backup("weekly"))
        schedule.every().day.at("04:00").do(cleanup_old_backups)
    elif BACKUP_SCHEDULE == "weekly":
        schedule.every().sunday.at("02:00").do(lambda: run_backup("weekly"))
    
    # Monthly backup on 1st of each month
    schedule.every().day.at("05:00").do(
        lambda: run_backup("monthly") if datetime.now().day == 1 else None
    )


async def run_scheduler():
    """Run the backup scheduler in background"""
    schedule_backups()
    while True:
        schedule.run_pending()
        await asyncio.sleep(60)


# =============================================================================
# API MODELS
# =============================================================================

class RestoreRequest(BaseModel):
    backup_path: str
    volumes: Optional[List[str]] = None  # None = all volumes


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Start the backup scheduler"""
    asyncio.create_task(run_scheduler())


@app.get("/")
async def root():
    return {
        "service": "Origin OS Backup Service",
        "version": "1.0",
        "config": {
            "backup_dir": BACKUP_DIR,
            "retention_days": RETENTION_DAYS,
            "schedule": BACKUP_SCHEDULE,
            "s3_enabled": S3_ENABLED,
            "s3_bucket": S3_BUCKET if S3_ENABLED else None
        },
        "volumes": [v["name"] for v in VOLUMES],
        "state": backup_state,
        "endpoints": {
            "backup_now": "POST /backup",
            "list_backups": "GET /backups",
            "restore": "POST /restore",
            "status": "GET /status",
            "cleanup": "POST /cleanup"
        }
    }


@app.post("/backup")
async def trigger_backup(backup_type: str = "daily", background_tasks: BackgroundTasks = None):
    """Trigger a manual backup"""
    if backup_state["is_running"]:
        raise HTTPException(400, "Backup already in progress")
    
    if background_tasks:
        background_tasks.add_task(run_backup, backup_type)
        return {"status": "started", "type": backup_type}
    else:
        result = run_backup(backup_type)
        return result


@app.get("/backups")
async def get_backups():
    """List all available backups"""
    backups = list_backups()
    
    total_size = sum(b.get("total_size", 0) for b in backups)
    
    return {
        "backups": backups,
        "count": len(backups),
        "total_size_bytes": total_size,
        "total_size_human": f"{total_size / 1024 / 1024:.2f} MB"
    }


@app.get("/backups/{backup_type}/{timestamp}")
async def get_backup_details(backup_type: str, timestamp: str):
    """Get details of a specific backup"""
    backup_path = os.path.join(BACKUP_DIR, backup_type, timestamp)
    manifest_path = os.path.join(backup_path, "manifest.json")
    
    if not os.path.exists(backup_path):
        raise HTTPException(404, "Backup not found")
    
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    
    # Build info manually
    files = []
    for f in os.listdir(backup_path):
        filepath = os.path.join(backup_path, f)
        if os.path.isfile(filepath):
            files.append({
                "name": f,
                "size": os.path.getsize(filepath)
            })
    
    return {"timestamp": timestamp, "type": backup_type, "files": files}


@app.post("/restore")
async def restore_backup(req: RestoreRequest):
    """Restore volumes from a backup"""
    if not os.path.exists(req.backup_path):
        raise HTTPException(404, "Backup path not found")
    
    volumes_to_restore = req.volumes or [v["name"] for v in VOLUMES]
    results = []
    
    for vol_name in volumes_to_restore:
        result = restore_volume(req.backup_path, vol_name)
        results.append(result)
    
    success = all(r.get("success") for r in results)
    
    return {
        "success": success,
        "results": results
    }


@app.post("/cleanup")
async def trigger_cleanup():
    """Manually trigger cleanup of old backups"""
    removed = cleanup_old_backups()
    return {
        "removed": removed,
        "count": len(removed)
    }


@app.get("/status")
async def get_status():
    """Get backup service status"""
    # Calculate disk usage
    total_size = 0
    backup_count = 0
    
    for backup_type in ["daily", "weekly", "monthly"]:
        type_path = os.path.join(BACKUP_DIR, backup_type)
        if os.path.exists(type_path):
            for backup_dir in os.listdir(type_path):
                backup_path = os.path.join(type_path, backup_dir)
                if os.path.isdir(backup_path):
                    backup_count += 1
                    for f in os.listdir(backup_path):
                        filepath = os.path.join(backup_path, f)
                        if os.path.isfile(filepath):
                            total_size += os.path.getsize(filepath)
    
    return {
        "state": backup_state,
        "disk_usage_bytes": total_size,
        "disk_usage_human": f"{total_size / 1024 / 1024:.2f} MB",
        "backup_count": backup_count,
        "schedule": BACKUP_SCHEDULE,
        "next_scheduled": str(schedule.next_run()) if schedule.jobs else None
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "backup"}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("💾 ORIGIN OS BACKUP SERVICE")
    print("=" * 60)
    print(f"\nBackup Directory: {BACKUP_DIR}")
    print(f"Retention: {RETENTION_DAYS} days")
    print(f"Schedule: {BACKUP_SCHEDULE}")
    print(f"S3 Enabled: {S3_ENABLED}")
    print(f"\nVolumes to backup:")
    for v in VOLUMES:
        print(f"  • {v['name']}: {v['description']}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
