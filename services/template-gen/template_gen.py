#!/usr/bin/env python3
"""
NEXTJS TEMPLATE GENERATOR — Device-Specific React Containers
=============================================================
Generates optimized Next.js projects for each device type.
No responsive CSS - each container is tailored for its device.
"""

import os
import json
import logging
from typing import Optional, Dict
from pathlib import Path
import zipfile
import io

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("template-gen")

app = FastAPI(title="Next.js Template Generator", version="1.0.0")

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

class DeviceConfig(BaseModel):
    device_type: str
    viewport_width: int
    viewport_height: int
    pixel_ratio: float = 1.0
    touch_enabled: bool = False

class TemplateConfig(BaseModel):
    project_name: str
    device_type: str
    page_content: str
    gtm_container_id: Optional[str] = None
    experiment_id: Optional[str] = None
    variant_id: Optional[str] = None
    origin_intercept_url: Optional[str] = None
    styles: Optional[str] = None

DEVICE_PRESETS = {
    "mobile-ios": DeviceConfig(device_type="mobile-ios", viewport_width=390, viewport_height=844, pixel_ratio=3.0, touch_enabled=True),
    "mobile-android": DeviceConfig(device_type="mobile-android", viewport_width=412, viewport_height=915, pixel_ratio=2.625, touch_enabled=True),
    "tablet-portrait": DeviceConfig(device_type="tablet-portrait", viewport_width=768, viewport_height=1024, pixel_ratio=2.0, touch_enabled=True),
    "tablet-landscape": DeviceConfig(device_type="tablet-landscape", viewport_width=1024, viewport_height=768, pixel_ratio=2.0, touch_enabled=True),
    "desktop-small": DeviceConfig(device_type="desktop-small", viewport_width=1280, viewport_height=800, pixel_ratio=1.0, touch_enabled=False),
    "desktop-large": DeviceConfig(device_type="desktop-large", viewport_width=1920, viewport_height=1080, pixel_ratio=1.0, touch_enabled=False),
    "desktop-ultra": DeviceConfig(device_type="desktop-ultra", viewport_width=2560, viewport_height=1440, pixel_ratio=1.0, touch_enabled=False),
}

class TemplateGenerator:
    def generate_package_json(self, config: TemplateConfig) -> str:
        return json.dumps({
            "name": config.project_name,
            "version": "0.1.0",
            "private": True,
            "scripts": {"dev": "next dev", "build": "next build", "start": "next start"},
            "dependencies": {"next": "14.0.4", "react": "^18", "react-dom": "^18"},
            "devDependencies": {"typescript": "^5", "tailwindcss": "^3.3.0", "@types/react": "^18"}
        }, indent=2)
    
    def generate_layout(self, config: TemplateConfig) -> str:
        gtm = ""
        if config.gtm_container_id:
            url = config.origin_intercept_url or "http://localhost:8012"
            gtm = f'''<script dangerouslySetInnerHTML={{{{__html: `
(function(w,d,s,l,i,o){{w[l]=w[l]||[];var p=w[l].push.bind(w[l]);w[l].push=function(data){{
if(data&&data.event){{fetch(o+'/event',{{method:'POST',headers:{{'Content-Type':'application/json'}},
body:JSON.stringify({{event_name:data.event,event_params:data,experiment_id:'{config.experiment_id or ""}',
variant_id:'{config.variant_id or ""}',device_type:'{config.device_type}',timestamp:new Date().toISOString()}}),
keepalive:true}}).catch(function(){{}});}}return p(data);}};w[l].push({{'gtm.start':new Date().getTime(),event:'gtm.js'}});
var f=d.getElementsByTagName(s)[0],j=d.createElement(s);j.async=true;
j.src='https://www.googletagmanager.com/gtm.js?id='+i;f.parentNode.insertBefore(j,f);
}})(window,document,'script','dataLayer','{config.gtm_container_id}','{url}');`}}}} />'''
        
        return f'''import './globals.css'
export const metadata = {{ title: '{config.project_name}' }}
export default function RootLayout({{ children }}: {{ children: React.ReactNode }}) {{
  return (<html lang="en"><head>{gtm}</head><body>{{children}}</body></html>)
}}'''
    
    def generate_page(self, config: TemplateConfig) -> str:
        if not config.page_content.strip().startswith("export"):
            return f"'use client'\nexport default function Home() {{ return (<main>{config.page_content}</main>) }}"
        return config.page_content
    
    def generate_css(self, config: TemplateConfig) -> str:
        device = DEVICE_PRESETS.get(config.device_type, DEVICE_PRESETS["desktop-large"])
        touch = "* { -webkit-tap-highlight-color: transparent; } button, a { min-height: 44px; }" if device.touch_enabled else ""
        return f"@tailwind base;@tailwind components;@tailwind utilities;\n:root{{--device-width:{device.viewport_width}px;}}\n{touch}\n{config.styles or ''}"
    
    def generate_zip(self, config: TemplateConfig) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('package.json', self.generate_package_json(config))
            zf.writestr('next.config.js', 'module.exports = { reactStrictMode: true }')
            zf.writestr('tailwind.config.js', 'module.exports = { content: ["./app/**/*.{js,ts,jsx,tsx}"], theme: { extend: {} } }')
            zf.writestr('app/layout.tsx', self.generate_layout(config))
            zf.writestr('app/page.tsx', self.generate_page(config))
            zf.writestr('app/globals.css', self.generate_css(config))
            zf.writestr('vercel.json', '{"framework": "nextjs"}')
        buf.seek(0)
        return buf.getvalue()

generator = TemplateGenerator()

@app.get("/health")
async def health():
    return {"status": "healthy", "devices": list(DEVICE_PRESETS.keys())}

@app.get("/devices")
async def list_devices():
    return {"devices": {k: v.model_dump() for k, v in DEVICE_PRESETS.items()}}

@app.post("/generate")
async def generate_template(config: TemplateConfig):
    return {
        "layout": generator.generate_layout(config),
        "page": generator.generate_page(config),
        "css": generator.generate_css(config)
    }

@app.post("/generate/zip")
async def generate_zip(config: TemplateConfig):
    return StreamingResponse(
        io.BytesIO(generator.generate_zip(config)),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={config.project_name}.zip"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
