#!/usr/bin/env python3
"""
DEVICE ROUTER — Device-Specific Container Routing
==================================================
Instead of responsive CSS (slow), serve device-specific containers (fast).

Device Types:
- mobile-ios       → iPhone optimized
- mobile-android   → Android optimized  
- tablet-portrait  → iPad/tablet portrait
- tablet-landscape → iPad/tablet landscape
- desktop-small    → 1024-1440px
- desktop-large    → 1440-1920px
- desktop-ultra    → 1920px+

Architecture:
  Request → Edge Detection → Device Container → Render
                                    ↓
                            GTM Event with device_type
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(os.getenv("DEVICE_DATA_DIR", "/data/device-router"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

GTM_INTERCEPT_URL = os.getenv("GTM_INTERCEPT_URL", "http://gtm-intercept:8000")
VERCEL_URL = os.getenv("VERCEL_URL", "http://vercel:8000")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("device-router")

# =============================================================================
# DEVICE DETECTION
# =============================================================================

class DeviceType:
    MOBILE_IOS = "mobile-ios"
    MOBILE_ANDROID = "mobile-android"
    TABLET_PORTRAIT = "tablet-portrait"
    TABLET_LANDSCAPE = "tablet-landscape"
    DESKTOP_SMALL = "desktop-small"
    DESKTOP_LARGE = "desktop-large"
    DESKTOP_ULTRA = "desktop-ultra"
    UNKNOWN = "unknown"

# User-Agent patterns
UA_PATTERNS = {
    DeviceType.MOBILE_IOS: [
        r'iPhone',
        r'iPod',
    ],
    DeviceType.MOBILE_ANDROID: [
        r'Android.*Mobile',
        r'Android.*(?!Tablet)',
    ],
    DeviceType.TABLET_PORTRAIT: [
        r'iPad',
        r'Android.*Tablet',
        r'Tablet',
    ],
}

# Client Hints mapping
CLIENT_HINT_MOBILE = {
    "Sec-CH-UA-Mobile": "?1"
}

def detect_device_from_ua(user_agent: str) -> str:
    """Detect device type from User-Agent string"""
    if not user_agent:
        return DeviceType.DESKTOP_LARGE
    
    ua = user_agent.lower()
    
    # iOS devices
    if 'iphone' in ua or 'ipod' in ua:
        return DeviceType.MOBILE_IOS
    
    # iPad
    if 'ipad' in ua:
        return DeviceType.TABLET_PORTRAIT
    
    # Android
    if 'android' in ua:
        if 'mobile' in ua:
            return DeviceType.MOBILE_ANDROID
        else:
            return DeviceType.TABLET_PORTRAIT
    
    # Windows tablets
    if 'tablet' in ua or ('windows' in ua and 'touch' in ua):
        return DeviceType.TABLET_PORTRAIT
    
    # Desktop - default
    return DeviceType.DESKTOP_LARGE

def detect_device_from_hints(headers: Dict[str, str]) -> Optional[str]:
    """Detect device from Client Hints headers (more accurate)"""
    
    # Check mobile hint
    mobile = headers.get("sec-ch-ua-mobile", "").lower()
    platform = headers.get("sec-ch-ua-platform", "").lower().strip('"')
    
    if mobile == "?1":
        # It's mobile
        if "ios" in platform or "iphone" in platform:
            return DeviceType.MOBILE_IOS
        elif "android" in platform:
            return DeviceType.MOBILE_ANDROID
        else:
            return DeviceType.MOBILE_ANDROID  # Default mobile
    
    # Check platform for tablets
    if "ipad" in platform:
        return DeviceType.TABLET_PORTRAIT
    
    return None  # Fall back to UA detection

def detect_viewport_category(width: int, height: int) -> str:
    """Categorize based on viewport dimensions"""
    
    # Portrait vs landscape for tablets
    is_portrait = height > width
    
    if width < 768:
        # Mobile
        return DeviceType.MOBILE_ANDROID  # Generic mobile
    elif width < 1024:
        # Tablet
        return DeviceType.TABLET_PORTRAIT if is_portrait else DeviceType.TABLET_LANDSCAPE
    elif width < 1440:
        return DeviceType.DESKTOP_SMALL
    elif width < 1920:
        return DeviceType.DESKTOP_LARGE
    else:
        return DeviceType.DESKTOP_ULTRA

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Device Router",
    description="Device-Specific Container Routing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MODELS
# =============================================================================

class DeviceInfo(BaseModel):
    device_type: str
    user_agent: Optional[str] = None
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None
    platform: Optional[str] = None
    is_mobile: bool = False
    is_tablet: bool = False
    is_desktop: bool = True

class ContainerConfig(BaseModel):
    site_id: str
    device_type: str
    container_url: str
    vercel_project_id: Optional[str] = None
    active: bool = True

class SiteConfig(BaseModel):
    site_id: str
    name: str
    domain: str
    containers: Dict[str, ContainerConfig] = {}
    default_device: str = DeviceType.DESKTOP_LARGE
    gtm_container_id: Optional[str] = None

class RouteRequest(BaseModel):
    site_id: str
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None

# =============================================================================
# CONTAINER REGISTRY
# =============================================================================

class ContainerRegistry:
    def __init__(self):
        self.sites: Dict[str, SiteConfig] = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load site configs from disk"""
        config_file = DATA_DIR / "sites.json"
        if config_file.exists():
            data = json.loads(config_file.read_text())
            for site_data in data.get("sites", []):
                site = SiteConfig(**site_data)
                self.sites[site.site_id] = site
    
    def _save_configs(self):
        """Save site configs to disk"""
        config_file = DATA_DIR / "sites.json"
        data = {
            "sites": [site.model_dump() for site in self.sites.values()]
        }
        config_file.write_text(json.dumps(data, indent=2))
    
    def register_site(self, config: SiteConfig) -> SiteConfig:
        """Register a new site"""
        self.sites[config.site_id] = config
        self._save_configs()
        return config
    
    def register_container(
        self,
        site_id: str,
        device_type: str,
        container_url: str,
        vercel_project_id: Optional[str] = None
    ) -> ContainerConfig:
        """Register a container for a device type"""
        
        if site_id not in self.sites:
            raise HTTPException(404, f"Site {site_id} not found")
        
        container = ContainerConfig(
            site_id=site_id,
            device_type=device_type,
            container_url=container_url,
            vercel_project_id=vercel_project_id
        )
        
        self.sites[site_id].containers[device_type] = container
        self._save_configs()
        
        return container
    
    def get_container_url(self, site_id: str, device_type: str) -> Optional[str]:
        """Get container URL for site + device"""
        
        site = self.sites.get(site_id)
        if not site:
            return None
        
        # Try exact match
        if device_type in site.containers:
            container = site.containers[device_type]
            if container.active:
                return container.container_url
        
        # Fallback hierarchy
        fallbacks = self._get_fallback_chain(device_type)
        for fallback in fallbacks:
            if fallback in site.containers:
                container = site.containers[fallback]
                if container.active:
                    return container.container_url
        
        # Ultimate fallback to default
        if site.default_device in site.containers:
            return site.containers[site.default_device].container_url
        
        return None
    
    def _get_fallback_chain(self, device_type: str) -> List[str]:
        """Get fallback device types"""
        
        chains = {
            DeviceType.MOBILE_IOS: [DeviceType.MOBILE_ANDROID, DeviceType.TABLET_PORTRAIT],
            DeviceType.MOBILE_ANDROID: [DeviceType.MOBILE_IOS, DeviceType.TABLET_PORTRAIT],
            DeviceType.TABLET_PORTRAIT: [DeviceType.TABLET_LANDSCAPE, DeviceType.DESKTOP_SMALL],
            DeviceType.TABLET_LANDSCAPE: [DeviceType.TABLET_PORTRAIT, DeviceType.DESKTOP_SMALL],
            DeviceType.DESKTOP_SMALL: [DeviceType.DESKTOP_LARGE],
            DeviceType.DESKTOP_LARGE: [DeviceType.DESKTOP_SMALL, DeviceType.DESKTOP_ULTRA],
            DeviceType.DESKTOP_ULTRA: [DeviceType.DESKTOP_LARGE],
        }
        
        return chains.get(device_type, [DeviceType.DESKTOP_LARGE])


# Global registry
registry = ContainerRegistry()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "sites": len(registry.sites),
        "device_types": [
            DeviceType.MOBILE_IOS,
            DeviceType.MOBILE_ANDROID,
            DeviceType.TABLET_PORTRAIT,
            DeviceType.TABLET_LANDSCAPE,
            DeviceType.DESKTOP_SMALL,
            DeviceType.DESKTOP_LARGE,
            DeviceType.DESKTOP_ULTRA,
        ]
    }

@app.get("/detect")
async def detect_device(request: Request):
    """Detect device from request headers"""
    
    headers = dict(request.headers)
    user_agent = headers.get("user-agent", "")
    
    # Try Client Hints first (more accurate)
    device_type = detect_device_from_hints(headers)
    
    # Fall back to User-Agent
    if not device_type:
        device_type = detect_device_from_ua(user_agent)
    
    is_mobile = device_type in [DeviceType.MOBILE_IOS, DeviceType.MOBILE_ANDROID]
    is_tablet = device_type in [DeviceType.TABLET_PORTRAIT, DeviceType.TABLET_LANDSCAPE]
    
    return DeviceInfo(
        device_type=device_type,
        user_agent=user_agent[:200],
        platform=headers.get("sec-ch-ua-platform", ""),
        is_mobile=is_mobile,
        is_tablet=is_tablet,
        is_desktop=not is_mobile and not is_tablet
    )

@app.post("/detect-viewport")
async def detect_from_viewport(width: int, height: int):
    """Detect device from viewport dimensions (client-side)"""
    device_type = detect_viewport_category(width, height)
    
    is_mobile = device_type in [DeviceType.MOBILE_IOS, DeviceType.MOBILE_ANDROID]
    is_tablet = device_type in [DeviceType.TABLET_PORTRAIT, DeviceType.TABLET_LANDSCAPE]
    
    return DeviceInfo(
        device_type=device_type,
        viewport_width=width,
        viewport_height=height,
        is_mobile=is_mobile,
        is_tablet=is_tablet,
        is_desktop=not is_mobile and not is_tablet
    )

# Site management
@app.post("/sites")
async def register_site(config: SiteConfig):
    """Register a new site"""
    site = registry.register_site(config)
    return site.model_dump()

@app.get("/sites")
async def list_sites():
    """List all sites"""
    return {"sites": [s.model_dump() for s in registry.sites.values()]}

@app.get("/sites/{site_id}")
async def get_site(site_id: str):
    """Get site config"""
    site = registry.sites.get(site_id)
    if not site:
        raise HTTPException(404, "Site not found")
    return site.model_dump()

# Container management
@app.post("/sites/{site_id}/containers")
async def register_container(
    site_id: str,
    device_type: str,
    container_url: str,
    vercel_project_id: Optional[str] = None
):
    """Register a container for a device type"""
    container = registry.register_container(
        site_id, device_type, container_url, vercel_project_id
    )
    return container.model_dump()

# Routing
@app.get("/route/{site_id}")
async def route_to_container(site_id: str, request: Request):
    """Route request to appropriate device container"""
    
    # Detect device
    headers = dict(request.headers)
    user_agent = headers.get("user-agent", "")
    
    device_type = detect_device_from_hints(headers)
    if not device_type:
        device_type = detect_device_from_ua(user_agent)
    
    # Get container URL
    container_url = registry.get_container_url(site_id, device_type)
    
    if not container_url:
        raise HTTPException(404, f"No container for site {site_id}, device {device_type}")
    
    return {
        "site_id": site_id,
        "device_type": device_type,
        "container_url": container_url,
        "redirect": True
    }

@app.get("/redirect/{site_id}")
async def redirect_to_container(site_id: str, request: Request):
    """Redirect to appropriate device container"""
    
    headers = dict(request.headers)
    user_agent = headers.get("user-agent", "")
    
    device_type = detect_device_from_hints(headers)
    if not device_type:
        device_type = detect_device_from_ua(user_agent)
    
    container_url = registry.get_container_url(site_id, device_type)
    
    if not container_url:
        raise HTTPException(404, f"No container for site {site_id}")
    
    return RedirectResponse(url=container_url, status_code=302)

# =============================================================================
# EDGE MIDDLEWARE GENERATOR
# =============================================================================

@app.get("/middleware/{site_id}")
async def generate_middleware(site_id: str):
    """Generate Vercel Edge Middleware for device routing"""
    
    site = registry.sites.get(site_id)
    if not site:
        raise HTTPException(404, "Site not found")
    
    # Generate Next.js middleware
    middleware_code = f'''// middleware.ts - Auto-generated by Origin OS Device Router
import {{ NextResponse }} from 'next/server'
import type {{ NextRequest }} from 'next/server'

// Device container URLs
const CONTAINERS = {{
{chr(10).join(f'  "{dt}": "{c.container_url}",' for dt, c in site.containers.items())}
}}

function detectDevice(request: NextRequest): string {{
  const ua = request.headers.get('user-agent') || ''
  const mobile = request.headers.get('sec-ch-ua-mobile')
  const platform = request.headers.get('sec-ch-ua-platform') || ''
  
  // Client hints (most accurate)
  if (mobile === '?1') {{
    if (platform.toLowerCase().includes('ios')) return 'mobile-ios'
    return 'mobile-android'
  }}
  
  // User-Agent fallback
  const uaLower = ua.toLowerCase()
  if (uaLower.includes('iphone') || uaLower.includes('ipod')) return 'mobile-ios'
  if (uaLower.includes('ipad')) return 'tablet-portrait'
  if (uaLower.includes('android')) {{
    if (uaLower.includes('mobile')) return 'mobile-android'
    return 'tablet-portrait'
  }}
  
  return 'desktop-large'
}}

export function middleware(request: NextRequest) {{
  const device = detectDevice(request)
  const containerUrl = CONTAINERS[device] || CONTAINERS['desktop-large']
  
  // Rewrite to device-specific container
  if (containerUrl) {{
    const url = new URL(containerUrl)
    url.pathname = request.nextUrl.pathname
    url.search = request.nextUrl.search
    
    // Add device header for analytics
    const response = NextResponse.rewrite(url)
    response.headers.set('x-device-type', device)
    return response
  }}
  
  return NextResponse.next()
}}

export const config = {{
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
}}
'''
    
    return Response(
        content=middleware_code,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=middleware.ts"}
    )

# =============================================================================
# CLIENT-SIDE DETECTOR SCRIPT
# =============================================================================

@app.get("/detector.js")
async def get_detector_script():
    """Client-side device detector script"""
    
    script = '''
// Origin OS Device Detector
(function() {
  const DEVICE_ROUTER_URL = window.ORIGIN_DEVICE_ROUTER || '';
  
  function detectDevice() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    const ua = navigator.userAgent.toLowerCase();
    
    // Check for iOS
    if (/iphone|ipod/.test(ua)) return 'mobile-ios';
    if (/ipad/.test(ua)) return height > width ? 'tablet-portrait' : 'tablet-landscape';
    
    // Check for Android
    if (/android/.test(ua)) {
      if (/mobile/.test(ua)) return 'mobile-android';
      return height > width ? 'tablet-portrait' : 'tablet-landscape';
    }
    
    // Desktop categories by width
    if (width < 768) return 'mobile-android';
    if (width < 1024) return height > width ? 'tablet-portrait' : 'tablet-landscape';
    if (width < 1440) return 'desktop-small';
    if (width < 1920) return 'desktop-large';
    return 'desktop-ultra';
  }
  
  function getContainerUrl(siteId, callback) {
    const device = detectDevice();
    
    fetch(DEVICE_ROUTER_URL + '/route/' + siteId, {
      headers: {
        'X-Viewport-Width': window.innerWidth,
        'X-Viewport-Height': window.innerHeight
      }
    })
    .then(r => r.json())
    .then(data => callback(null, data))
    .catch(err => callback(err, null));
  }
  
  function redirectToContainer(siteId) {
    getContainerUrl(siteId, function(err, data) {
      if (!err && data.container_url) {
        window.location.href = data.container_url;
      }
    });
  }
  
  // Export
  window.OriginDeviceRouter = {
    detectDevice: detectDevice,
    getContainerUrl: getContainerUrl,
    redirectToContainer: redirectToContainer
  };
})();
'''
    
    return Response(content=script, media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
