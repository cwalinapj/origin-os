#!/usr/bin/env python3
"""
LLM AGENT PROVISIONER — Container-Based Infrastructure Setup
=============================================================

Runs inside a container to provision all non-global resources for 5 LLM agents:
- S3 Buckets (one per LLM)
- Pinecone Vector DB namespaces (one per LLM)
- Generates agent-specific env vars

Expects global env vars to be passed in:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION
- PINECONE_API_KEY

Outputs agent-specific env vars to /output/.env.agents
"""

import os
import json
import boto3
import httpx
import asyncio
from typing import Dict, List
from dataclasses import dataclass, asdict

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/output")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "origin-os-lam")

@dataclass
class LLMAgent:
    name: str
    model_id: str
    gtag_id: str
    s3_bucket: str
    vector_namespace: str
    description: str

LLM_AGENTS: List[LLMAgent] = [
    LLMAgent(
        name="claude",
        model_id="anthropic/claude-sonnet-4-20250514",
        gtag_id="GT-CLAUDE001",
        s3_bucket="origin-os-lam-claude",
        vector_namespace="claude",
        description="Anthropic Claude - Nuanced, brand-safe copy"
    ),
    LLMAgent(
        name="gpt4",
        model_id="openai/gpt-4o",
        gtag_id="GT-GPT4O002",
        s3_bucket="origin-os-lam-gpt4",
        vector_namespace="gpt4",
        description="OpenAI GPT-4o - Versatile, creative"
    ),
    LLMAgent(
        name="gemini",
        model_id="google/gemini-2.0-flash",
        gtag_id="GT-GEMINI03",
        s3_bucket="origin-os-lam-gemini",
        vector_namespace="gemini",
        description="Google Gemini - Fast, multimodal"
    ),
    LLMAgent(
        name="llama",
        model_id="meta/llama-3.1-405b-instruct",
        gtag_id="GT-LLAMA004",
        s3_bucket="origin-os-lam-llama",
        vector_namespace="llama",
        description="Meta LLaMA - Open-source, efficient"
    ),
    LLMAgent(
        name="mistral",
        model_id="mistralai/mistral-large",
        gtag_id="GT-MISTRAL5",
        s3_bucket="origin-os-lam-mistral",
        vector_namespace="mistral",
        description="Mistral AI - European, multilingual"
    )
]


# =============================================================================
# S3 PROVISIONING
# =============================================================================

def provision_s3_buckets() -> Dict[str, str]:
    """Create S3 buckets for each LLM agent."""
    print("\n" + "=" * 60)
    print("  PROVISIONING S3 BUCKETS")
    print("=" * 60)
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    bucket_urls = {}
    
    for agent in LLM_AGENTS:
        bucket_name = agent.s3_bucket
        
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"  ✓ EXISTS: {bucket_name}")
        except s3.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                # Create bucket
                try:
                    if AWS_REGION == "us-east-1":
                        s3.create_bucket(Bucket=bucket_name)
                    else:
                        s3.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
                        )
                    print(f"  ✓ CREATED: {bucket_name}")
                except Exception as create_err:
                    print(f"  ✗ FAILED: {bucket_name} - {create_err}")
                    continue
            else:
                print(f"  ✗ ERROR: {bucket_name} - {e}")
                continue
        
        # Enable versioning
        try:
            s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={"Status": "Enabled"}
            )
        except Exception as e:
            print(f"    ⚠ Versioning failed: {e}")
        
        # Create folder structure
        folders = ["mutations/", "tombstones/", "training/", "logs/", "embeddings/"]
        for folder in folders:
            try:
                s3.put_object(Bucket=bucket_name, Key=folder, Body=b"")
            except:
                pass
        
        bucket_urls[agent.name] = f"s3://{bucket_name}"
    
    return bucket_urls


# =============================================================================
# PINECONE PROVISIONING
# =============================================================================

async def provision_pinecone() -> Dict[str, str]:
    """Create Pinecone index and return namespace URLs."""
    print("\n" + "=" * 60)
    print("  PROVISIONING PINECONE VECTOR DB")
    print("=" * 60)
    
    if not PINECONE_API_KEY:
        print("  ⚠ PINECONE_API_KEY not provided, skipping")
        return {agent.name: "" for agent in LLM_AGENTS}
    
    headers = {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json"
    }
    
    vector_urls = {}
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # List existing indexes
        try:
            response = await client.get(
                "https://api.pinecone.io/indexes",
                headers=headers
            )
            response.raise_for_status()
            indexes = response.json().get("indexes", [])
            index_exists = any(idx.get("name") == PINECONE_INDEX_NAME for idx in indexes)
        except Exception as e:
            print(f"  ✗ Failed to list indexes: {e}")
            return {agent.name: "" for agent in LLM_AGENTS}
        
        # Create index if needed
        if not index_exists:
            print(f"  Creating index: {PINECONE_INDEX_NAME}")
            try:
                response = await client.post(
                    "https://api.pinecone.io/indexes",
                    headers=headers,
                    json={
                        "name": PINECONE_INDEX_NAME,
                        "dimension": 1536,
                        "metric": "cosine",
                        "spec": {
                            "serverless": {
                                "cloud": "aws",
                                "region": "us-east-1"
                            }
                        }
                    }
                )
                response.raise_for_status()
                print(f"  ✓ CREATED: {PINECONE_INDEX_NAME}")
                
                # Wait for index to be ready
                print("  Waiting for index to be ready...")
                await asyncio.sleep(30)
            except Exception as e:
                print(f"  ✗ Failed to create index: {e}")
                return {agent.name: "" for agent in LLM_AGENTS}
        else:
            print(f"  ✓ EXISTS: {PINECONE_INDEX_NAME}")
        
        # Get index host
        try:
            response = await client.get(
                f"https://api.pinecone.io/indexes/{PINECONE_INDEX_NAME}",
                headers=headers
            )
            response.raise_for_status()
            index_info = response.json()
            index_host = index_info.get("host", "")
            
            if index_host:
                base_url = f"https://{index_host}"
                for agent in LLM_AGENTS:
                    vector_urls[agent.name] = base_url
                    print(f"  ✓ Namespace: {agent.vector_namespace} @ {index_host}")
            else:
                print("  ⚠ Could not get index host")
        except Exception as e:
            print(f"  ✗ Failed to get index info: {e}")
    
    return vector_urls


# =============================================================================
# ENV FILE GENERATION
# =============================================================================

def generate_agent_env(bucket_urls: Dict[str, str], vector_urls: Dict[str, str]) -> str:
    """Generate agent-specific environment variables."""
    
    lines = [
        "# =============================================================================",
        "# LLM AGENT ENVIRONMENT VARIABLES",
        "# =============================================================================",
        "# Auto-generated by provision_llm_agents.py",
        "# These are NON-GLOBAL, agent-specific configurations",
        "#",
        "# Global vars (GA_API_KEY, PINECONE_API_KEY, VERCEL_API_KEY, OPENROUTER_API_KEY)",
        "# should be provided separately",
        "# =============================================================================",
        "",
    ]
    
    for agent in LLM_AGENTS:
        upper = agent.name.upper()
        s3_url = bucket_urls.get(agent.name, f"s3://{agent.s3_bucket}")
        vector_url = vector_urls.get(agent.name, "")
        
        lines.extend([
            f"# {agent.description}",
            f"{upper}_MODEL_ID={agent.model_id}",
            f"{upper}_GTAG_ID={agent.gtag_id}",
            f"{upper}_S3_BUCKET={agent.s3_bucket}",
            f"{upper}_S3_URL={s3_url}",
            f"{upper}_VECTOR_NAMESPACE={agent.vector_namespace}",
            f"{upper}_VECTOR_DB_URL={vector_url}",
            "",
        ])
    
    return "\n".join(lines)


def generate_agent_json(bucket_urls: Dict[str, str], vector_urls: Dict[str, str]) -> str:
    """Generate agent config as JSON for programmatic use."""
    
    config = {}
    for agent in LLM_AGENTS:
        config[agent.name] = {
            "model_id": agent.model_id,
            "gtag_id": agent.gtag_id,
            "s3_bucket": agent.s3_bucket,
            "s3_url": bucket_urls.get(agent.name, f"s3://{agent.s3_bucket}"),
            "vector_namespace": agent.vector_namespace,
            "vector_db_url": vector_urls.get(agent.name, ""),
            "description": agent.description
        }
    
    return json.dumps(config, indent=2)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    print("=" * 60)
    print("  LLM AGENT PROVISIONER")
    print("  Creating non-global infrastructure for 5 LLM agents")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Provision S3
    bucket_urls = provision_s3_buckets()
    
    # Provision Pinecone
    vector_urls = await provision_pinecone()
    
    # Generate outputs
    print("\n" + "=" * 60)
    print("  GENERATING CONFIGURATION FILES")
    print("=" * 60)
    
    # ENV file
    env_content = generate_agent_env(bucket_urls, vector_urls)
    env_path = os.path.join(OUTPUT_DIR, ".env.agents")
    with open(env_path, "w") as f:
        f.write(env_content)
    print(f"  ✓ {env_path}")
    
    # JSON file
    json_content = generate_agent_json(bucket_urls, vector_urls)
    json_path = os.path.join(OUTPUT_DIR, "agents.json")
    with open(json_path, "w") as f:
        f.write(json_content)
    print(f"  ✓ {json_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  PROVISIONING COMPLETE")
    print("=" * 60)
    print()
    print("  Agents configured:")
    for agent in LLM_AGENTS:
        print(f"    • {agent.name.upper()}: {agent.gtag_id} | {agent.s3_bucket}")
    print()
    print(f"  Output files in: {OUTPUT_DIR}")
    print("    • .env.agents  (for docker-compose)")
    print("    • agents.json  (for programmatic use)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
