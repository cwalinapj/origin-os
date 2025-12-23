#!/usr/bin/env python3
"""
COMMERCE MCP — Headless E-Commerce Integration
===============================================
Unified API for multiple headless commerce backends:
- Shopify Storefront API
- Medusa (open source)
- Saleor (open source)
- BigCommerce
- Swell

Powers the LAM experiment engine for e-commerce A/B testing.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(os.getenv("COMMERCE_DATA_DIR", "/data/commerce"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Commerce backends
SHOPIFY_STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN", "")  # mystore.myshopify.com
SHOPIFY_STOREFRONT_TOKEN = os.getenv("SHOPIFY_STOREFRONT_TOKEN", "")
MEDUSA_URL = os.getenv("MEDUSA_URL", "http://localhost:9000")
SALEOR_URL = os.getenv("SALEOR_URL", "")
BIGCOMMERCE_STORE_HASH = os.getenv("BIGCOMMERCE_STORE_HASH", "")
BIGCOMMERCE_ACCESS_TOKEN = os.getenv("BIGCOMMERCE_ACCESS_TOKEN", "")
SWELL_STORE_ID = os.getenv("SWELL_STORE_ID", "")
SWELL_PUBLIC_KEY = os.getenv("SWELL_PUBLIC_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("commerce-mcp")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Commerce MCP",
    description="Headless E-Commerce Integration for Origin OS",
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

class CommerceBackend(str, Enum):
    SHOPIFY = "shopify"
    MEDUSA = "medusa"
    SALEOR = "saleor"
    BIGCOMMERCE = "bigcommerce"
    SWELL = "swell"

class ProductImage(BaseModel):
    url: str
    alt_text: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

class ProductVariant(BaseModel):
    id: str
    title: str
    sku: Optional[str] = None
    price: float
    compare_at_price: Optional[float] = None
    available: bool = True
    inventory_quantity: Optional[int] = None
    options: Dict[str, str] = {}

class Product(BaseModel):
    id: str
    title: str
    handle: str
    description: Optional[str] = None
    vendor: Optional[str] = None
    product_type: Optional[str] = None
    tags: List[str] = []
    images: List[ProductImage] = []
    variants: List[ProductVariant] = []
    price_range: Dict[str, float] = {}
    available: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class Collection(BaseModel):
    id: str
    title: str
    handle: str
    description: Optional[str] = None
    image: Optional[ProductImage] = None
    products_count: int = 0

class CartItem(BaseModel):
    variant_id: str
    quantity: int
    product_title: Optional[str] = None
    variant_title: Optional[str] = None
    price: Optional[float] = None
    image: Optional[str] = None

class Cart(BaseModel):
    id: str
    items: List[CartItem] = []
    subtotal: float = 0
    total: float = 0
    currency: str = "USD"
    item_count: int = 0

class StoreConfig(BaseModel):
    backend: CommerceBackend
    store_name: Optional[str] = None
    domain: Optional[str] = None
    currency: str = "USD"
    features: Dict[str, bool] = {}

# =============================================================================
# SHOPIFY CLIENT
# =============================================================================

class ShopifyClient:
    def __init__(self):
        self.domain = SHOPIFY_STORE_DOMAIN
        self.token = SHOPIFY_STOREFRONT_TOKEN
        self.api_version = "2024-01"
        self.http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        return self.http_client
    
    @property
    def endpoint(self) -> str:
        return f"https://{self.domain}/api/{self.api_version}/graphql.json"
    
    async def query(self, query: str, variables: Dict = None) -> Dict:
        """Execute Shopify Storefront GraphQL query"""
        client = await self._get_client()
        
        response = await client.post(
            self.endpoint,
            json={"query": query, "variables": variables or {}},
            headers={
                "X-Shopify-Storefront-Access-Token": self.token,
                "Content-Type": "application/json"
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def get_products(self, first: int = 20, after: str = None) -> List[Product]:
        """Get products from Shopify"""
        query = """
        query GetProducts($first: Int!, $after: String) {
          products(first: $first, after: $after) {
            edges {
              node {
                id
                title
                handle
                description
                vendor
                productType
                tags
                availableForSale
                createdAt
                updatedAt
                priceRange {
                  minVariantPrice { amount currencyCode }
                  maxVariantPrice { amount currencyCode }
                }
                images(first: 5) {
                  edges {
                    node { url altText width height }
                  }
                }
                variants(first: 10) {
                  edges {
                    node {
                      id
                      title
                      sku
                      availableForSale
                      quantityAvailable
                      price { amount currencyCode }
                      compareAtPrice { amount currencyCode }
                      selectedOptions { name value }
                    }
                  }
                }
              }
            }
            pageInfo { hasNextPage endCursor }
          }
        }
        """
        
        result = await self.query(query, {"first": first, "after": after})
        products = []
        
        for edge in result.get("data", {}).get("products", {}).get("edges", []):
            node = edge["node"]
            products.append(self._parse_product(node))
        
        return products
    
    async def get_product(self, handle: str) -> Optional[Product]:
        """Get single product by handle"""
        query = """
        query GetProduct($handle: String!) {
          productByHandle(handle: $handle) {
            id
            title
            handle
            description
            vendor
            productType
            tags
            availableForSale
            createdAt
            updatedAt
            priceRange {
              minVariantPrice { amount currencyCode }
              maxVariantPrice { amount currencyCode }
            }
            images(first: 10) {
              edges {
                node { url altText width height }
              }
            }
            variants(first: 50) {
              edges {
                node {
                  id
                  title
                  sku
                  availableForSale
                  quantityAvailable
                  price { amount currencyCode }
                  compareAtPrice { amount currencyCode }
                  selectedOptions { name value }
                }
              }
            }
          }
        }
        """
        
        result = await self.query(query, {"handle": handle})
        node = result.get("data", {}).get("productByHandle")
        
        if node:
            return self._parse_product(node)
        return None
    
    async def get_collections(self, first: int = 20) -> List[Collection]:
        """Get collections"""
        query = """
        query GetCollections($first: Int!) {
          collections(first: $first) {
            edges {
              node {
                id
                title
                handle
                description
                productsCount
                image { url altText }
              }
            }
          }
        }
        """
        
        result = await self.query(query, {"first": first})
        collections = []
        
        for edge in result.get("data", {}).get("collections", {}).get("edges", []):
            node = edge["node"]
            collections.append(Collection(
                id=node["id"],
                title=node["title"],
                handle=node["handle"],
                description=node.get("description"),
                products_count=node.get("productsCount", 0),
                image=ProductImage(url=node["image"]["url"]) if node.get("image") else None
            ))
        
        return collections
    
    async def create_cart(self) -> Cart:
        """Create a new cart"""
        query = """
        mutation CreateCart {
          cartCreate {
            cart {
              id
              checkoutUrl
              cost {
                subtotalAmount { amount currencyCode }
                totalAmount { amount currencyCode }
              }
              totalQuantity
              lines(first: 50) {
                edges {
                  node {
                    id
                    quantity
                    merchandise {
                      ... on ProductVariant {
                        id
                        title
                        price { amount }
                        product { title }
                        image { url }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        result = await self.query(query)
        cart_data = result.get("data", {}).get("cartCreate", {}).get("cart", {})
        
        return self._parse_cart(cart_data)
    
    async def add_to_cart(self, cart_id: str, variant_id: str, quantity: int = 1) -> Cart:
        """Add item to cart"""
        query = """
        mutation AddToCart($cartId: ID!, $lines: [CartLineInput!]!) {
          cartLinesAdd(cartId: $cartId, lines: $lines) {
            cart {
              id
              cost {
                subtotalAmount { amount currencyCode }
                totalAmount { amount currencyCode }
              }
              totalQuantity
              lines(first: 50) {
                edges {
                  node {
                    id
                    quantity
                    merchandise {
                      ... on ProductVariant {
                        id
                        title
                        price { amount }
                        product { title }
                        image { url }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        result = await self.query(query, {
            "cartId": cart_id,
            "lines": [{"merchandiseId": variant_id, "quantity": quantity}]
        })
        
        cart_data = result.get("data", {}).get("cartLinesAdd", {}).get("cart", {})
        return self._parse_cart(cart_data)
    
    def _parse_product(self, node: Dict) -> Product:
        """Parse Shopify product node"""
        images = [
            ProductImage(
                url=img["node"]["url"],
                alt_text=img["node"].get("altText"),
                width=img["node"].get("width"),
                height=img["node"].get("height")
            )
            for img in node.get("images", {}).get("edges", [])
        ]
        
        variants = []
        for v in node.get("variants", {}).get("edges", []):
            vnode = v["node"]
            options = {opt["name"]: opt["value"] for opt in vnode.get("selectedOptions", [])}
            variants.append(ProductVariant(
                id=vnode["id"],
                title=vnode["title"],
                sku=vnode.get("sku"),
                price=float(vnode["price"]["amount"]),
                compare_at_price=float(vnode["compareAtPrice"]["amount"]) if vnode.get("compareAtPrice") else None,
                available=vnode.get("availableForSale", True),
                inventory_quantity=vnode.get("quantityAvailable"),
                options=options
            ))
        
        price_range = {}
        if node.get("priceRange"):
            price_range["min"] = float(node["priceRange"]["minVariantPrice"]["amount"])
            price_range["max"] = float(node["priceRange"]["maxVariantPrice"]["amount"])
        
        return Product(
            id=node["id"],
            title=node["title"],
            handle=node["handle"],
            description=node.get("description"),
            vendor=node.get("vendor"),
            product_type=node.get("productType"),
            tags=node.get("tags", []),
            images=images,
            variants=variants,
            price_range=price_range,
            available=node.get("availableForSale", True),
            created_at=node.get("createdAt"),
            updated_at=node.get("updatedAt")
        )
    
    def _parse_cart(self, cart_data: Dict) -> Cart:
        """Parse Shopify cart"""
        items = []
        for edge in cart_data.get("lines", {}).get("edges", []):
            node = edge["node"]
            merch = node.get("merchandise", {})
            items.append(CartItem(
                variant_id=merch.get("id", ""),
                quantity=node.get("quantity", 0),
                product_title=merch.get("product", {}).get("title"),
                variant_title=merch.get("title"),
                price=float(merch.get("price", {}).get("amount", 0)),
                image=merch.get("image", {}).get("url")
            ))
        
        return Cart(
            id=cart_data.get("id", ""),
            items=items,
            subtotal=float(cart_data.get("cost", {}).get("subtotalAmount", {}).get("amount", 0)),
            total=float(cart_data.get("cost", {}).get("totalAmount", {}).get("amount", 0)),
            item_count=cart_data.get("totalQuantity", 0)
        )


# =============================================================================
# MEDUSA CLIENT (Open Source)
# =============================================================================

class MedusaClient:
    def __init__(self):
        self.base_url = MEDUSA_URL
        self.http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        return self.http_client
    
    async def get_products(self, limit: int = 20, offset: int = 0) -> List[Product]:
        """Get products from Medusa"""
        client = await self._get_client()
        
        response = await client.get(
            f"{self.base_url}/store/products",
            params={"limit": limit, "offset": offset}
        )
        response.raise_for_status()
        data = response.json()
        
        products = []
        for p in data.get("products", []):
            products.append(self._parse_product(p))
        
        return products
    
    async def get_product(self, handle: str) -> Optional[Product]:
        """Get single product"""
        client = await self._get_client()
        
        response = await client.get(
            f"{self.base_url}/store/products",
            params={"handle": handle}
        )
        response.raise_for_status()
        data = response.json()
        
        products = data.get("products", [])
        if products:
            return self._parse_product(products[0])
        return None
    
    async def get_collections(self, limit: int = 20) -> List[Collection]:
        """Get collections"""
        client = await self._get_client()
        
        response = await client.get(
            f"{self.base_url}/store/collections",
            params={"limit": limit}
        )
        response.raise_for_status()
        data = response.json()
        
        collections = []
        for c in data.get("collections", []):
            collections.append(Collection(
                id=c["id"],
                title=c["title"],
                handle=c["handle"],
                description=c.get("metadata", {}).get("description")
            ))
        
        return collections
    
    async def create_cart(self) -> Cart:
        """Create a new cart"""
        client = await self._get_client()
        
        response = await client.post(f"{self.base_url}/store/carts")
        response.raise_for_status()
        data = response.json()
        
        return self._parse_cart(data.get("cart", {}))
    
    async def add_to_cart(self, cart_id: str, variant_id: str, quantity: int = 1) -> Cart:
        """Add item to cart"""
        client = await self._get_client()
        
        response = await client.post(
            f"{self.base_url}/store/carts/{cart_id}/line-items",
            json={"variant_id": variant_id, "quantity": quantity}
        )
        response.raise_for_status()
        data = response.json()
        
        return self._parse_cart(data.get("cart", {}))
    
    def _parse_product(self, p: Dict) -> Product:
        """Parse Medusa product"""
        images = [ProductImage(url=img["url"]) for img in p.get("images", [])]
        
        variants = []
        for v in p.get("variants", []):
            prices = v.get("prices", [])
            price = prices[0]["amount"] / 100 if prices else 0
            variants.append(ProductVariant(
                id=v["id"],
                title=v["title"],
                sku=v.get("sku"),
                price=price,
                available=v.get("inventory_quantity", 0) > 0,
                inventory_quantity=v.get("inventory_quantity")
            ))
        
        return Product(
            id=p["id"],
            title=p["title"],
            handle=p["handle"],
            description=p.get("description"),
            images=images,
            variants=variants,
            available=p.get("status") == "published"
        )
    
    def _parse_cart(self, cart: Dict) -> Cart:
        """Parse Medusa cart"""
        items = []
        for item in cart.get("items", []):
            items.append(CartItem(
                variant_id=item.get("variant_id", ""),
                quantity=item.get("quantity", 0),
                product_title=item.get("title"),
                price=item.get("unit_price", 0) / 100
            ))
        
        return Cart(
            id=cart.get("id", ""),
            items=items,
            subtotal=cart.get("subtotal", 0) / 100,
            total=cart.get("total", 0) / 100,
            item_count=len(items)
        )


# =============================================================================
# UNIFIED COMMERCE CLIENT
# =============================================================================

class CommerceClient:
    def __init__(self):
        self.shopify = ShopifyClient()
        self.medusa = MedusaClient()
        self._default_backend = self._detect_backend()
    
    def _detect_backend(self) -> CommerceBackend:
        """Detect which backend is configured"""
        if SHOPIFY_STORE_DOMAIN and SHOPIFY_STOREFRONT_TOKEN:
            return CommerceBackend.SHOPIFY
        elif MEDUSA_URL:
            return CommerceBackend.MEDUSA
        else:
            return CommerceBackend.MEDUSA  # Default to Medusa (open source)
    
    async def get_products(
        self,
        backend: CommerceBackend = None,
        limit: int = 20
    ) -> List[Product]:
        backend = backend or self._default_backend
        
        if backend == CommerceBackend.SHOPIFY:
            return await self.shopify.get_products(first=limit)
        elif backend == CommerceBackend.MEDUSA:
            return await self.medusa.get_products(limit=limit)
        else:
            raise HTTPException(501, f"Backend {backend} not implemented")
    
    async def get_product(
        self,
        handle: str,
        backend: CommerceBackend = None
    ) -> Optional[Product]:
        backend = backend or self._default_backend
        
        if backend == CommerceBackend.SHOPIFY:
            return await self.shopify.get_product(handle)
        elif backend == CommerceBackend.MEDUSA:
            return await self.medusa.get_product(handle)
        else:
            raise HTTPException(501, f"Backend {backend} not implemented")
    
    async def get_collections(
        self,
        backend: CommerceBackend = None,
        limit: int = 20
    ) -> List[Collection]:
        backend = backend or self._default_backend
        
        if backend == CommerceBackend.SHOPIFY:
            return await self.shopify.get_collections(first=limit)
        elif backend == CommerceBackend.MEDUSA:
            return await self.medusa.get_collections(limit=limit)
        else:
            raise HTTPException(501, f"Backend {backend} not implemented")
    
    async def create_cart(self, backend: CommerceBackend = None) -> Cart:
        backend = backend or self._default_backend
        
        if backend == CommerceBackend.SHOPIFY:
            return await self.shopify.create_cart()
        elif backend == CommerceBackend.MEDUSA:
            return await self.medusa.create_cart()
        else:
            raise HTTPException(501, f"Backend {backend} not implemented")
    
    async def add_to_cart(
        self,
        cart_id: str,
        variant_id: str,
        quantity: int = 1,
        backend: CommerceBackend = None
    ) -> Cart:
        backend = backend or self._default_backend
        
        if backend == CommerceBackend.SHOPIFY:
            return await self.shopify.add_to_cart(cart_id, variant_id, quantity)
        elif backend == CommerceBackend.MEDUSA:
            return await self.medusa.add_to_cart(cart_id, variant_id, quantity)
        else:
            raise HTTPException(501, f"Backend {backend} not implemented")


# Global client
commerce = CommerceClient()

# =============================================================================
# NEXT.JS COMMERCE TEMPLATE GENERATOR
# =============================================================================

class CommerceTemplateGenerator:
    """Generate Next.js Commerce components for LAM experiments"""
    
    def generate_product_card(self, product: Product, device_type: str) -> str:
        """Generate product card component"""
        
        # Adjust styles based on device
        if device_type.startswith("mobile"):
            card_class = "w-full p-2"
            image_size = "w-full aspect-square"
        elif device_type.startswith("tablet"):
            card_class = "w-1/2 p-3"
            image_size = "w-full aspect-square"
        else:
            card_class = "w-1/4 p-4"
            image_size = "w-full aspect-square"
        
        image_url = product.images[0].url if product.images else "/placeholder.png"
        price = product.variants[0].price if product.variants else 0
        
        return f'''
<div className="{card_class} group">
  <a href="/product/{product.handle}" className="block">
    <div className="relative {image_size} overflow-hidden rounded-lg bg-gray-100">
      <img
        src="{image_url}"
        alt="{product.title}"
        className="object-cover w-full h-full group-hover:scale-105 transition-transform"
      />
    </div>
    <div className="mt-2">
      <h3 className="text-sm font-medium text-gray-900 truncate">{product.title}</h3>
      <p className="text-sm text-gray-700">${price:.2f}</p>
    </div>
  </a>
</div>
'''
    
    def generate_product_grid(self, products: List[Product], device_type: str) -> str:
        """Generate product grid component"""
        
        if device_type.startswith("mobile"):
            grid_class = "grid grid-cols-2 gap-2"
        elif device_type.startswith("tablet"):
            grid_class = "grid grid-cols-3 gap-4"
        else:
            grid_class = "grid grid-cols-4 gap-6"
        
        cards = "\n".join([
            self.generate_product_card(p, device_type) for p in products
        ])
        
        return f'''
<div className="{grid_class}">
  {cards}
</div>
'''
    
    def generate_hero_section(self, config: Dict, device_type: str) -> str:
        """Generate hero section"""
        
        if device_type.startswith("mobile"):
            height = "h-[60vh]"
            text_size = "text-3xl"
        else:
            height = "h-[80vh]"
            text_size = "text-6xl"
        
        return f'''
<section className="relative {height} flex items-center justify-center">
  <div className="absolute inset-0 bg-black/40 z-10" />
  <img
    src="{config.get('hero_image', '/hero.jpg')}"
    alt="Hero"
    className="absolute inset-0 w-full h-full object-cover"
  />
  <div className="relative z-20 text-center text-white px-4">
    <h1 className="{text_size} font-bold mb-4">{config.get('headline', 'Shop Now')}</h1>
    <p className="text-xl mb-8">{config.get('subheadline', 'Discover our collection')}</p>
    <a
      href="{config.get('cta_link', '/shop')}"
      className="inline-block bg-white text-black px-8 py-3 rounded-full font-medium hover:bg-gray-100 transition"
      data-gtm-event="hero_cta_click"
    >
      {config.get('cta_text', 'Shop Collection')}
    </a>
  </div>
</section>
'''
    
    def generate_cart_drawer(self, device_type: str) -> str:
        """Generate cart drawer component"""
        
        if device_type.startswith("mobile"):
            width = "w-full"
        else:
            width = "w-96"
        
        return f'''
<div className="fixed inset-y-0 right-0 {width} bg-white shadow-xl z-50 transform translate-x-full transition-transform" id="cart-drawer">
  <div className="flex flex-col h-full">
    <div className="flex items-center justify-between p-4 border-b">
      <h2 className="text-lg font-medium">Shopping Cart</h2>
      <button onClick={{() => closeCart()}} className="p-2">
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={{2}} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
    <div className="flex-1 overflow-y-auto p-4" id="cart-items">
      {{/* Cart items rendered here */}}
    </div>
    <div className="border-t p-4">
      <div className="flex justify-between mb-4">
        <span>Subtotal</span>
        <span id="cart-subtotal">$0.00</span>
      </div>
      <button 
        className="w-full bg-black text-white py-3 rounded-full font-medium hover:bg-gray-800 transition"
        data-gtm-event="checkout_click"
      >
        Checkout
      </button>
    </div>
  </div>
</div>
'''
    
    def generate_full_page(
        self,
        products: List[Product],
        collections: List[Collection],
        device_type: str,
        store_config: Dict
    ) -> str:
        """Generate full commerce page"""
        
        hero = self.generate_hero_section(store_config, device_type)
        grid = self.generate_product_grid(products[:12], device_type)
        cart = self.generate_cart_drawer(device_type)
        
        return f'''
'use client'

import {{ useState, useEffect }} from 'react'

export default function StorePage() {{
  const [cartOpen, setCartOpen] = useState(false)
  const [cart, setCart] = useState({{ items: [], total: 0 }})
  
  const addToCart = async (variantId: string) => {{
    // GTM event
    window.dataLayer?.push({{
      event: 'add_to_cart',
      variant_id: variantId,
      device_type: '{device_type}'
    }})
    
    // Add to cart logic
  }}
  
  return (
    <main className="min-h-screen">
      {hero}
      
      <section className="max-w-7xl mx-auto px-4 py-16">
        <h2 className="text-2xl font-bold mb-8">Featured Products</h2>
        {grid}
      </section>
      
      {cart}
    </main>
  )
}}
'''


template_gen = CommerceTemplateGenerator()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "backends": {
            "shopify": bool(SHOPIFY_STORE_DOMAIN and SHOPIFY_STOREFRONT_TOKEN),
            "medusa": bool(MEDUSA_URL),
            "saleor": bool(SALEOR_URL),
            "bigcommerce": bool(BIGCOMMERCE_STORE_HASH),
            "swell": bool(SWELL_STORE_ID)
        },
        "default_backend": commerce._default_backend.value
    }

# Products
@app.get("/products")
async def get_products(
    backend: Optional[CommerceBackend] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """Get products from commerce backend"""
    products = await commerce.get_products(backend=backend, limit=limit)
    return {"products": [p.model_dump() for p in products]}

@app.get("/products/{handle}")
async def get_product(handle: str, backend: Optional[CommerceBackend] = None):
    """Get single product by handle"""
    product = await commerce.get_product(handle, backend=backend)
    if not product:
        raise HTTPException(404, "Product not found")
    return product.model_dump()

# Collections
@app.get("/collections")
async def get_collections(
    backend: Optional[CommerceBackend] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """Get collections"""
    collections = await commerce.get_collections(backend=backend, limit=limit)
    return {"collections": [c.model_dump() for c in collections]}

# Cart
@app.post("/cart")
async def create_cart(backend: Optional[CommerceBackend] = None):
    """Create a new cart"""
    cart = await commerce.create_cart(backend=backend)
    return cart.model_dump()

@app.post("/cart/{cart_id}/add")
async def add_to_cart(
    cart_id: str,
    variant_id: str,
    quantity: int = 1,
    backend: Optional[CommerceBackend] = None
):
    """Add item to cart"""
    cart = await commerce.add_to_cart(cart_id, variant_id, quantity, backend=backend)
    return cart.model_dump()

# Template Generation
@app.post("/template/product-grid")
async def generate_product_grid(device_type: str, limit: int = 12):
    """Generate product grid component"""
    products = await commerce.get_products(limit=limit)
    html = template_gen.generate_product_grid(products, device_type)
    return {"html": html, "product_count": len(products)}

@app.post("/template/full-page")
async def generate_full_page(device_type: str, store_config: Dict = None):
    """Generate full commerce page for device"""
    products = await commerce.get_products(limit=12)
    collections = await commerce.get_collections(limit=5)
    
    config = store_config or {
        "headline": "Shop the Collection",
        "subheadline": "Premium products for every occasion",
        "cta_text": "Shop Now",
        "cta_link": "/shop"
    }
    
    page = template_gen.generate_full_page(products, collections, device_type, config)
    return {"page": page, "device_type": device_type}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
