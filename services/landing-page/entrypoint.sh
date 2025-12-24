#!/bin/sh

# Substitute environment variables in HTML
sed -i "s|__PAGE_ID__|${PAGE_ID:-unknown}|g" /usr/share/nginx/html/index.html
sed -i "s|__ROUTER_URL__|${ROUTER_URL:-http://localhost:8024}|g" /usr/share/nginx/html/index.html
sed -i "s|__VARIANT_TYPE__|${VARIANT_TYPE:-unknown}|g" /usr/share/nginx/html/index.html

# Execute the main command
exec "$@"
