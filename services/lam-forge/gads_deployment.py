#!/usr/bin/env python3
"""
Google Ads Deployment Service
=============================

Closes the loop between LAM decisions and Google's Smart Bidding.

Features:
- Final URL promotion post-convergence
- Conversion adjustment uploads (behavioral signals)
- Sitelink mutation deployment
- Batch processing for API limits
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

import redis.asyncio as redis
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from google.protobuf import field_mask_pb2

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
GOOGLE_ADS_CONFIG = os.getenv("GOOGLE_ADS_CONFIG", "/credentials/google-ads.yaml")
BATCH_INTERVAL = int(os.getenv("BATCH_INTERVAL", "900"))  # 15 minutes
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "2000"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gads-deployment")

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SitelinkMutation:
    """Sitelink mutation from LAM."""
    mutation_id: str
    campaign_id: str
    link_text: str
    description1: str
    description2: str
    final_url: str
    mobile_url: Optional[str] = None


@dataclass
class BehavioralScore:
    """Behavioral score for conversion adjustment."""
    gclid: str
    composite_score: float  # 0.0 to 10.0
    conversion_timestamp: str
    site_id: str


# =============================================================================
# DEPLOYMENT SERVICE
# =============================================================================

class GoogleAdsDeploymentService:
    """
    Handles all Google Ads API interactions for the LAM system.
    """
    
    def __init__(self, redis_client: redis.Redis, config_path: str = GOOGLE_ADS_CONFIG):
        self.redis = redis_client
        self.client = None
        self._config_path = config_path
        self._init_client()
    
    def _init_client(self):
        """Initialize Google Ads client."""
        try:
            self.client = GoogleAdsClient.load_from_storage(self._config_path)
            logger.info("Google Ads client initialized")
        except Exception as e:
            logger.warning(f"Google Ads client not initialized: {e}")
            self.client = None
    
    # ─────────────────────────────────────────────────────────────
    # 1. FINAL URL PROMOTION
    # ─────────────────────────────────────────────────────────────
    
    async def promote_winner_to_google_ads(
        self,
        customer_id: str,
        ad_group_id: str,
        winning_url: str,
        site_id: str
    ) -> Dict[str, Any]:
        """
        Update Final URLs once Divergence Rule confirms winner.
        
        This ensures Google's Quality Score recalculates against
        the champion landing page.
        """
        if not self.client:
            logger.warning("Google Ads client not available")
            return {"status": "skipped", "reason": "client_not_available"}
        
        ad_service = self.client.get_service("AdGroupAdService")
        ga_service = self.client.get_service("GoogleAdsService")
        
        # Query existing ads in the ad group
        query = f"""
            SELECT 
                ad_group_ad.resource_name,
                ad_group_ad.ad.id,
                ad_group_ad.ad.final_urls
            FROM ad_group_ad
            WHERE ad_group.id = {ad_group_id}
            AND ad_group_ad.status != 'REMOVED'
        """
        
        try:
            response = ga_service.search(customer_id=customer_id, query=query)
            
            operations = []
            for row in response:
                operation = self.client.get_type("AdGroupAdOperation")
                ad_group_ad = operation.update
                ad_group_ad.resource_name = row.ad_group_ad.resource_name
                
                # Clear existing and set winner
                ad_group_ad.ad.final_urls.clear()
                ad_group_ad.ad.final_urls.append(winning_url)
                
                # Field mask for partial update
                operation.update_mask.CopyFrom(
                    field_mask_pb2.FieldMask(paths=["ad.final_urls"])
                )
                operations.append(operation)
            
            # Execute batch mutation
            if operations:
                result = ad_service.mutate_ad_group_ads(
                    customer_id=customer_id,
                    operations=operations
                )
                
                # Log promotion event
                await self.redis.xadd(
                    f"promotions:{site_id}",
                    {
                        "ad_group_id": ad_group_id,
                        "winning_url": winning_url,
                        "ads_updated": len(operations),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                logger.info(f"Promoted {winning_url} to {len(operations)} ads in ad group {ad_group_id}")
                
                return {
                    "status": "success",
                    "ads_updated": len(operations),
                    "winning_url": winning_url
                }
            
            return {"status": "no_ads_found"}
            
        except GoogleAdsException as e:
            logger.error(f"Google Ads API error: {e}")
            return {"status": "error", "error": str(e)}
    
    # ─────────────────────────────────────────────────────────────
    # 2. CONVERSION ADJUSTMENT UPLOAD
    # ─────────────────────────────────────────────────────────────
    
    async def upload_behavioral_conversion(
        self,
        customer_id: str,
        score: BehavioralScore,
        conversion_action_id: str
    ) -> Dict[str, Any]:
        """
        Upload composite behavioral score as Conversion Adjustment.
        
        This creates the Double-Loop:
        - Internal: LAM optimizes page structure
        - External: Smart Bidding optimizes traffic quality
        """
        if not self.client:
            return {"status": "skipped", "reason": "client_not_available"}
        
        service = self.client.get_service("ConversionAdjustmentUploadService")
        
        adjustment = self.client.get_type("ConversionAdjustment")
        adjustment.adjustment_type = (
            self.client.enums.ConversionAdjustmentTypeEnum.RESTATEMENT
        )
        
        # Link to specific click via GCLID
        adjustment.gclid_date_time_pair.gclid = score.gclid
        adjustment.gclid_date_time_pair.conversion_date_time = score.conversion_timestamp
        
        # Our behavioral score becomes the restated value
        adjustment.restatement_value.adjusted_value = score.composite_score
        adjustment.conversion_action = (
            f"customers/{customer_id}/conversionActions/{conversion_action_id}"
        )
        adjustment.adjustment_date_time = (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
        )
        
        try:
            request = self.client.get_type("UploadConversionAdjustmentsRequest")
            request.customer_id = customer_id
            request.conversion_adjustments.append(adjustment)
            request.partial_failure = True
            
            response = service.upload_conversion_adjustments(request=request)
            
            return {"status": "success", "gclid": score.gclid}
            
        except GoogleAdsException as e:
            logger.error(f"Conversion adjustment error: {e}")
            return {"status": "error", "error": str(e)}
    
    # ─────────────────────────────────────────────────────────────
    # 3. SITELINK DEPLOYMENT
    # ─────────────────────────────────────────────────────────────
    
    async def deploy_sitelink_mutation(
        self,
        customer_id: str,
        campaign_id: str,
        mutation: SitelinkMutation,
        site_id: str
    ) -> Dict[str, Any]:
        """
        Deploy LAM-optimized sitelink to campaign.
        """
        if not self.client:
            return {"status": "skipped", "reason": "client_not_available"}
        
        asset_service = self.client.get_service("AssetService")
        campaign_asset_service = self.client.get_service("CampaignAssetService")
        
        try:
            # Create Sitelink Asset
            asset_op = self.client.get_type("AssetOperation")
            sitelink = asset_op.create.sitelink_asset
            sitelink.link_text = mutation.link_text[:25]  # Max 25 chars
            sitelink.description1 = mutation.description1[:35]
            sitelink.description2 = mutation.description2[:35]
            sitelink.final_urls.append(mutation.final_url)
            
            if mutation.mobile_url:
                sitelink.final_mobile_urls.append(mutation.mobile_url)
            
            asset_response = asset_service.mutate_assets(
                customer_id=customer_id,
                operations=[asset_op]
            )
            asset_resource = asset_response.results[0].resource_name
            
            # Link to Campaign
            campaign_asset_op = self.client.get_type("CampaignAssetOperation")
            campaign_asset = campaign_asset_op.create
            campaign_asset.campaign = f"customers/{customer_id}/campaigns/{campaign_id}"
            campaign_asset.asset = asset_resource
            campaign_asset.field_type = self.client.enums.AssetFieldTypeEnum.SITELINK
            
            campaign_asset_service.mutate_campaign_assets(
                customer_id=customer_id,
                operations=[campaign_asset_op]
            )
            
            # Track deployment
            await self.redis.hset(
                f"deployed_sitelinks:{site_id}",
                mutation.mutation_id,
                asset_resource
            )
            
            logger.info(f"Deployed sitelink {mutation.link_text} to campaign {campaign_id}")
            
            return {
                "status": "success",
                "asset_resource": asset_resource,
                "mutation_id": mutation.mutation_id
            }
            
        except GoogleAdsException as e:
            logger.error(f"Sitelink deployment error: {e}")
            return {"status": "error", "error": str(e)}
    
    # ─────────────────────────────────────────────────────────────
    # 4. BATCH PROCESSING
    # ─────────────────────────────────────────────────────────────
    
    async def flush_behavioral_batch(self, customer_id: str, site_id: str) -> Dict[str, Any]:
        """
        Batch upload all pending behavioral scores.
        
        Called on a schedule to respect API rate limits.
        """
        if not self.client:
            return {"status": "skipped", "reason": "client_not_available"}
        
        # Get conversion action ID for this site
        conversion_action_id = await self.redis.hget(
            f"site_config:{site_id}",
            "conversion_action_id"
        )
        
        if not conversion_action_id:
            return {"status": "skipped", "reason": "no_conversion_action"}
        
        conversion_action_id = conversion_action_id.decode()
        
        # Pull pending scores from Redis stream
        pending = await self.redis.xrange(
            f"behavioral_scores:{site_id}",
            count=MAX_BATCH_SIZE
        )
        
        if not pending:
            return {"status": "empty", "processed": 0}
        
        service = self.client.get_service("ConversionAdjustmentUploadService")
        adjustments = []
        message_ids = []
        
        for msg_id, data in pending:
            message_ids.append(msg_id)
            
            adjustment = self.client.get_type("ConversionAdjustment")
            adjustment.adjustment_type = (
                self.client.enums.ConversionAdjustmentTypeEnum.RESTATEMENT
            )
            adjustment.gclid_date_time_pair.gclid = data[b"gclid"].decode()
            adjustment.gclid_date_time_pair.conversion_date_time = (
                data[b"conversion_time"].decode()
            )
            adjustment.restatement_value.adjusted_value = float(data[b"score"])
            adjustment.conversion_action = (
                f"customers/{customer_id}/conversionActions/{conversion_action_id}"
            )
            adjustment.adjustment_date_time = (
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
            )
            adjustments.append(adjustment)
        
        try:
            request = self.client.get_type("UploadConversionAdjustmentsRequest")
            request.customer_id = customer_id
            request.conversion_adjustments.extend(adjustments)
            request.partial_failure = True
            
            response = service.upload_conversion_adjustments(request=request)
            
            # Clear processed messages
            if message_ids:
                await self.redis.xdel(f"behavioral_scores:{site_id}", *message_ids)
            
            partial_failures = 0
            if response.partial_failure_error:
                partial_failures = len(response.partial_failure_error.errors)
            
            logger.info(f"Uploaded {len(adjustments)} behavioral scores for {site_id}")
            
            return {
                "status": "success",
                "uploaded": len(adjustments),
                "partial_failures": partial_failures
            }
            
        except GoogleAdsException as e:
            logger.error(f"Batch upload error: {e}")
            return {"status": "error", "error": str(e)}


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(
    title="Google Ads Deployment Service",
    description="Closes the loop between LAM and Google Smart Bidding",
    version="1.0.0"
)

redis_client: Optional[redis.Redis] = None
deployment_service: Optional[GoogleAdsDeploymentService] = None


class PromotionRequest(BaseModel):
    customer_id: str
    ad_group_id: str
    winning_url: str
    site_id: str


class SitelinkRequest(BaseModel):
    customer_id: str
    campaign_id: str
    mutation_id: str
    link_text: str
    description1: str = ""
    description2: str = ""
    final_url: str
    mobile_url: Optional[str] = None
    site_id: str


class BatchFlushRequest(BaseModel):
    customer_id: str
    site_id: str


@app.on_event("startup")
async def startup():
    global redis_client, deployment_service
    redis_client = redis.from_url(REDIS_URL)
    deployment_service = GoogleAdsDeploymentService(redis_client)
    
    # Start batch processing loop
    asyncio.create_task(batch_processing_loop())


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


async def batch_processing_loop():
    """Periodic batch processing of behavioral scores."""
    while True:
        await asyncio.sleep(BATCH_INTERVAL)
        
        # Get all sites with pending scores
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(
                cursor,
                match="behavioral_scores:*",
                count=100
            )
            
            for key in keys:
                site_id = key.decode().split(":")[-1]
                config = await redis_client.hgetall(f"site_config:{site_id}")
                
                if config and b"customer_id" in config:
                    customer_id = config[b"customer_id"].decode()
                    result = await deployment_service.flush_behavioral_batch(
                        customer_id, site_id
                    )
                    logger.info(f"Batch flush for {site_id}: {result}")
            
            if cursor == 0:
                break


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "google_ads_client": deployment_service.client is not None
    }


@app.post("/promote")
async def promote_winner(request: PromotionRequest):
    """Promote winning URL to Google Ads."""
    result = await deployment_service.promote_winner_to_google_ads(
        request.customer_id,
        request.ad_group_id,
        request.winning_url,
        request.site_id
    )
    return result


@app.post("/sitelink")
async def deploy_sitelink(request: SitelinkRequest):
    """Deploy a sitelink mutation."""
    mutation = SitelinkMutation(
        mutation_id=request.mutation_id,
        campaign_id=request.campaign_id,
        link_text=request.link_text,
        description1=request.description1,
        description2=request.description2,
        final_url=request.final_url,
        mobile_url=request.mobile_url
    )
    
    result = await deployment_service.deploy_sitelink_mutation(
        request.customer_id,
        request.campaign_id,
        mutation,
        request.site_id
    )
    return result


@app.post("/batch-flush")
async def batch_flush(request: BatchFlushRequest):
    """Manually trigger batch flush of behavioral scores."""
    result = await deployment_service.flush_behavioral_batch(
        request.customer_id,
        request.site_id
    )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8070)
