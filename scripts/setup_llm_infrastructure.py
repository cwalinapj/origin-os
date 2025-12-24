#!/usr/bin/env python3
"""
INFRASTRUCTURE SETUP — LLM Agent Resources
============================================

Creates and configures infrastructure for all 5 LLM agents:
1. S3 Buckets (one per LLM)
2. Pinecone Vector DB namespaces (one per LLM)
3. Google Analytics properties (one GTAG per LLM)
"""

import os
import json
import boto3
from typing import Dict, Any
import httpx
import asyncio

# =============================================================================
# CONFIGURATION
# =============================================================================

LLM_AGENTS = {
    "claude": {
        "model_id": "anthropic/claude-sonnet-4-20250514",
        "gtag_id": "GT-CLAUDE001",
        "s3_bucket": "origin-os-lam-claude",
        "vector_namespace": "claude",
        "description": "Anthropic Claude - Nuanced, brand-safe copy"
    },
    "gpt4": {
        "model_id": "openai/gpt-4o",
        "gtag_id": "GT-GPT4O002",
        "s3_bucket": "origin-os-lam-gpt4",
        "vector_namespace": "gpt4",
        "description": "OpenAI GPT-4o - Versatile, creative"
    },
    "gemini": {
        "model_id": "google/gemini-2.0-flash",
        "gtag_id": "GT-GEMINI03",
        "s3_bucket": "origin-os-lam-gemini",
        "vector_namespace": "gemini",
        "description": "Google Gemini - Fast, multimodal"
    },
    "llama": {
        "model_id": "meta/llama-3.1-405b-instruct",
        "gtag_id": "GT-LLAMA004",
        "s3_bucket": "origin-os-lam-llama",
        "vector_namespace": "llama",
        "description": "Meta LLaMA - Open-source, efficient"
    },
    "mistral": {
        "model_id": "mistralai/mistral-large",
        "gtag_id": "GT-MISTRAL5",
        "s3_bucket": "origin-os-lam-mistral",
        "vector_namespace": "mistral",
        "description": "Mistral AI - European, multilingual"
    }
}

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = "origin-os-lam"


def create_s3_buckets() -> Dict[str, str]:
    """Create S3 buckets for each LLM agent."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    bucket_urls = {}
    
    for agent_name, config in LLM_AGENTS.items():
        bucket_name = config["s3_bucket"]
        
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"✓ S3 bucket exists: {bucket_name}")
        except:
            if AWS_REGION == "us-east-1":
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
                )
            print(f"✓ Created S3 bucket: {bucket_name}")
        
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={"Status": "Enabled"}
        )
        
        folders = ["mutations/", "tombstones/", "training/", "logs/"]
        for folder in folders:
            s3.put_object(Bucket=bucket_name, Key=folder)
        
        bucket_urls[agent_name] = f"s3://{bucket_name}"
    
    return bucket_urls


async def create_pinecone_index() -> Dict[str, str]:
    """Create Pinecone index and namespaces."""
    
    if not PINECONE_API_KEY:
        print("⚠ PINECONE_API_KEY not set")
        return {}
    
    headers = {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.pinecone.io/indexes",
            headers=headers
        )
        
        indexes = response.json().get("indexes", [])
        index_exists = any(idx["name"] == PINECONE_INDEX_NAME for idx in indexes)
        
        if not index_exists:
            print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            
            await client.post(
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
            print(f"✓ Created Pinecone index: {PINECONE_INDEX_NAME}")
        else:
            print(f"✓ Pinecone index exists: {PINECONE_INDEX_NAME}")
        
        response = await client.get(
            f"https://api.pinecone.io/indexes/{PINECONE_INDEX_NAME}",
            headers=headers
        )
        index_info = response.json()
        index_host = index_info.get("host", "")
        
        vector_urls = {}
        for agent_name, config in LLM_AGENTS.items():
            vector_urls[agent_name] = f"https://{index_host}"
            print(f"✓ Namespace ready: {config['vector_namespace']}")
        
        return vector_urls


def generate_env_file(bucket_urls: Dict[str, str], vector_urls: Dict[str, str]) -> str:
    """Generate complete .env file."""
    
    env_content = """# =============================================================================
# ORIGIN OS — LLM AGENT INFRASTRUCTURE
# =============================================================================

# GLOBAL
GA_API_KEY=YOUR_GA_API_KEY_HERE
PINECONE_API_KEY=YOUR_PINECONE_API_KEY_HERE
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=origin-os-lam
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_HERE
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_KEY_HERE
VERCEL_API_KEY=YOUR_VERCEL_API_KEY_HERE
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY_HERE

"""
    
    for agent_name, config in LLM_AGENTS.items():
        agent_upper = agent_name.upper()
        bucket_url = bucket_urls.get(agent_name, f"s3://{config['s3_bucket']}")
        vector_url = vector_urls.get(agent_name, "https://origin-os-lam.svc.us-east-1.pinecone.io")
        
        env_content += f"""# {agent_upper}
{agent_upper}_MODEL_ID={config['model_id']}
{agent_upper}_GTAG_ID={config['gtag_id']}
{agent_upper}_S3_BUCKET={config['s3_bucket']}
{agent_upper}_S3_URL={bucket_url}
{agent_upper}_VECTOR_NAMESPACE={config['vector_namespace']}
{agent_upper}_VECTOR_DB_URL={vector_url}

"""
    
    return env_content


async def main():
    print("=" * 60)
    print("  ORIGIN OS — LLM AGENT INFRASTRUCTURE SETUP")
    print("=" * 60)
    print()
    
    print("1. Creating S3 Buckets...")
    try:
        bucket_urls = create_s3_buckets()
    except Exception as e:
        print(f"⚠ S3 setup failed: {e}")
        bucket_urls = {name: f"s3://{config['s3_bucket']}" for name, config in LLM_AGENTS.items()}
    
    print()
    print("2. Setting up Pinecone Vector DB...")
    try:
        vector_urls = await create_pinecone_index()
    except Exception as e:
        print(f"⚠ Pinecone setup failed: {e}")
        vector_urls = {}
    
    print()
    print("3. Generating configuration...")
    env_content = generate_env_file(bucket_urls, vector_urls)
    
    with open(".env.llm-infrastructure", "w") as f:
        f.write(env_content)
    print("✓ Created .env.llm-infrastructure")
    
    print()
    print("=" * 60)
    print("  INFRASTRUCTURE SUMMARY")
    print("=" * 60)
    
    for agent_name, config in LLM_AGENTS.items():
        print(f"\n  {agent_name.upper()}")
        print(f"    Model:     {config['model_id']}")
        print(f"    GTAG:      {config['gtag_id']}")
        print(f"    S3:        {config['s3_bucket']}")
        print(f"    Namespace: {config['vector_namespace']}")
    
    print("\n\nNext: Fill in API keys in .env.llm-infrastructure")


if __name__ == "__main__":
    asyncio.run(main())
