#!/usr/bin/env python3
"""
Vault CLI - Command line interface for Origin OS Vault
"""

import argparse
import requests
import json
import sys
import os
from typing import Optional

VAULT_URL = os.getenv("VAULT_URL", "http://localhost:8004")


def create_envelope(args):
    """Create a new token envelope"""
    data = {
        "name": args.name,
        "value": args.value,
        "token_type": args.type,
        "service": args.service,
        "environment": args.env,
        "owner": args.owner,
        "access_policy": args.policy,
        "allowed_services": args.allow.split(",") if args.allow else [],
        "expires_in_days": args.expires
    }
    
    headers = {"X-Accessor": args.owner}
    r = requests.post(f"{VAULT_URL}/envelopes", json=data, headers=headers)
    
    if r.status_code == 200:
        envelope = r.json()
        print(f"✅ Created envelope: {envelope['id']}")
        print(f"   Name: {envelope['name']}")
        print(f"   Type: {envelope['token_type']}")
        print(f"   Service: {envelope['service']}")
        return envelope['id']
    else:
        print(f"❌ Error: {r.text}")
        sys.exit(1)


def unwrap_envelope(args):
    """Unwrap (decrypt) a token envelope"""
    headers = {"X-Accessor": args.accessor}
    r = requests.post(f"{VAULT_URL}/envelopes/{args.id}/unwrap", headers=headers)
    
    if r.status_code == 200:
        result = r.json()
        if args.quiet:
            print(result['value'])
        else:
            print(f"🔓 Unwrapped: {result['name']}")
            print(f"   Type: {result['token_type']}")
            print(f"   Value: {result['value'][:20]}..." if len(result['value']) > 20 else f"   Value: {result['value']}")
    else:
        print(f"❌ Error: {r.text}")
        sys.exit(1)


def list_envelopes(args):
    """List token envelopes"""
    params = {}
    if args.service:
        params['service'] = args.service
    if args.env:
        params['environment'] = args.env
    if args.owner:
        params['owner'] = args.owner
    
    r = requests.get(f"{VAULT_URL}/envelopes", params=params)
    
    if r.status_code == 200:
        envelopes = r.json()
        print(f"📦 Found {len(envelopes)} envelope(s):\n")
        for e in envelopes:
            status = "🔴 expired" if e.get('expires_at') else "🟢 active"
            print(f"  {e['id']}")
            print(f"    Name: {e['name']}")
            print(f"    Type: {e['token_type']}")
            print(f"    Service: {e['service']} ({e['environment']})")
            print(f"    Owner: {e['owner']}")
            print(f"    Version: {e['version']}, Accessed: {e['access_count']}x")
            print()
    else:
        print(f"❌ Error: {r.text}")


def rotate_envelope(args):
    """Rotate a token to new value"""
    headers = {"X-Accessor": args.accessor}
    data = {"new_value": args.value}
    r = requests.post(f"{VAULT_URL}/envelopes/{args.id}/rotate", json=data, headers=headers)
    
    if r.status_code == 200:
        envelope = r.json()
        print(f"🔄 Rotated envelope: {envelope['id']}")
        print(f"   New version: {envelope['version']}")
        print(f"   Rotated at: {envelope['rotated_at']}")
    else:
        print(f"❌ Error: {r.text}")
        sys.exit(1)


def delete_envelope(args):
    """Delete an envelope"""
    headers = {"X-Accessor": args.accessor}
    r = requests.delete(f"{VAULT_URL}/envelopes/{args.id}", headers=headers)
    
    if r.status_code == 200:
        print(f"🗑️  Deleted envelope: {args.id}")
    else:
        print(f"❌ Error: {r.text}")
        sys.exit(1)


def show_audit(args):
    """Show audit log"""
    r = requests.get(f"{VAULT_URL}/audit", params={"lines": args.lines})
    
    if r.status_code == 200:
        entries = r.json()['entries']
        print(f"📋 Last {len(entries)} audit entries:\n")
        for e in entries:
            print(f"  {e['timestamp']} | {e['action']:15} | {e['envelope_id'][:20]:20} | {e['accessor']}")
            if e.get('details'):
                print(f"    └─ {e['details']}")
    else:
        print(f"❌ Error: {r.text}")


def import_env(args):
    """Import secrets from .env file"""
    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)
    
    imported = 0
    with open(args.file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            
            # Skip if too short
            if len(value) < 5:
                continue
            
            # Determine token type
            token_type = "secret"
            if "API_KEY" in key or "APIKEY" in key:
                token_type = "api_key"
            elif "TOKEN" in key:
                token_type = "oauth_token"
            elif "PASSWORD" in key or "SECRET" in key:
                token_type = "password"
            elif "KEY" in key and "PRIVATE" in key:
                token_type = "ssh_key"
            
            # Determine service
            service = key.lower().replace("_api_key", "").replace("_token", "").replace("_", "-")
            
            data = {
                "name": key,
                "value": value,
                "token_type": token_type,
                "service": service,
                "environment": args.env,
                "owner": "system",
                "access_policy": "service",
                "allowed_services": ["mcp-hub", "codex", "ui"]
            }
            
            headers = {"X-Accessor": "system"}
            r = requests.post(f"{VAULT_URL}/envelopes", json=data, headers=headers)
            
            if r.status_code == 200:
                print(f"  ✅ Imported: {key}")
                imported += 1
            else:
                print(f"  ❌ Failed: {key} - {r.text}")
    
    print(f"\n📦 Imported {imported} secrets to vault")


def main():
    parser = argparse.ArgumentParser(description="Origin OS Vault CLI")
    parser.add_argument("--url", default=VAULT_URL, help="Vault URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create
    create_p = subparsers.add_parser("create", help="Create envelope")
    create_p.add_argument("name", help="Envelope name")
    create_p.add_argument("value", help="Token value")
    create_p.add_argument("--type", default="api_key", choices=["api_key", "oauth_token", "refresh_token", "jwt", "password", "secret", "certificate", "ssh_key", "webhook_secret", "encryption_key"])
    create_p.add_argument("--service", default="default", help="Service name")
    create_p.add_argument("--env", default="prod", help="Environment")
    create_p.add_argument("--owner", default="system", help="Owner")
    create_p.add_argument("--policy", default="private", choices=["private", "service", "shared", "public_read"])
    create_p.add_argument("--allow", help="Comma-separated allowed services")
    create_p.add_argument("--expires", type=int, help="Expires in N days")
    create_p.set_defaults(func=create_envelope)
    
    # Unwrap
    unwrap_p = subparsers.add_parser("unwrap", help="Unwrap envelope")
    unwrap_p.add_argument("id", help="Envelope ID")
    unwrap_p.add_argument("--accessor", default="cli", help="Accessor identity")
    unwrap_p.add_argument("-q", "--quiet", action="store_true", help="Output value only")
    unwrap_p.set_defaults(func=unwrap_envelope)
    
    # List
    list_p = subparsers.add_parser("list", help="List envelopes")
    list_p.add_argument("--service", help="Filter by service")
    list_p.add_argument("--env", help="Filter by environment")
    list_p.add_argument("--owner", help="Filter by owner")
    list_p.set_defaults(func=list_envelopes)
    
    # Rotate
    rotate_p = subparsers.add_parser("rotate", help="Rotate token")
    rotate_p.add_argument("id", help="Envelope ID")
    rotate_p.add_argument("value", help="New token value")
    rotate_p.add_argument("--accessor", default="cli", help="Accessor identity")
    rotate_p.set_defaults(func=rotate_envelope)
    
    # Delete
    delete_p = subparsers.add_parser("delete", help="Delete envelope")
    delete_p.add_argument("id", help="Envelope ID")
    delete_p.add_argument("--accessor", default="cli", help="Accessor identity")
    delete_p.set_defaults(func=delete_envelope)
    
    # Audit
    audit_p = subparsers.add_parser("audit", help="Show audit log")
    audit_p.add_argument("--lines", type=int, default=20, help="Number of lines")
    audit_p.set_defaults(func=show_audit)
    
    # Import
    import_p = subparsers.add_parser("import", help="Import from .env file")
    import_p.add_argument("file", help="Path to .env file")
    import_p.add_argument("--env", default="prod", help="Environment")
    import_p.set_defaults(func=import_env)
    
    args = parser.parse_args()
    
    if args.command:
        global VAULT_URL
        VAULT_URL = args.url
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
