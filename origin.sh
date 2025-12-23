#!/bin/bash
# Origin OS - Secure Docker Launcher
# Decrypts .env, starts containers, then re-encrypts

SCRIPT_DIR="$HOME"
ENV_FILE="$SCRIPT_DIR/.env"
ENCRYPTED_FILE="$SCRIPT_DIR/.env.encrypted"

start() {
    echo "=============================================="
    echo "🚀 ORIGIN OS - SECURE START"
    echo "=============================================="
    
    # Check if encrypted
    if [ -f "$ENCRYPTED_FILE" ] && [ ! -f "$ENV_FILE" ]; then
        echo "🔓 Decrypting credentials..."
        openssl enc -aes-256-cbc -d -pbkdf2 -in "$ENCRYPTED_FILE" -out "$ENV_FILE"
        if [ $? -ne 0 ]; then
            echo "❌ Wrong password"
            exit 1
        fi
        chmod 600 "$ENV_FILE"
    fi
    
    # Start containers
    echo "🐳 Starting containers..."
    cd "$SCRIPT_DIR"
    docker compose up -d
    
    echo ""
    echo "✅ Origin OS started!"
    echo ""
    echo "Services:"
    echo "  • UI:      http://localhost:8000"
    echo "  • Codex:   http://localhost:8001"
    echo "  • MCP:     http://localhost:8002"
    echo "  • Supreme: http://localhost:8080"
}

stop() {
    echo "=============================================="
    echo "🛑 ORIGIN OS - SECURE STOP"
    echo "=============================================="
    
    # Stop containers
    echo "🐳 Stopping containers..."
    cd "$SCRIPT_DIR"
    docker compose down
    
    # Re-encrypt .env
    if [ -f "$ENV_FILE" ]; then
        echo ""
        echo "🔒 Re-encrypting credentials..."
        read -p "Encrypt .env? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            openssl enc -aes-256-cbc -salt -pbkdf2 -in "$ENV_FILE" -out "$ENCRYPTED_FILE"
            if [ $? -eq 0 ]; then
                shred -u "$ENV_FILE" 2>/dev/null || rm -P "$ENV_FILE" 2>/dev/null || rm "$ENV_FILE"
                echo "✅ Credentials encrypted"
            fi
        fi
    fi
    
    echo ""
    echo "✅ Origin OS stopped"
}

case "${1:-start}" in
    start|up)
        start
        ;;
    stop|down)
        stop
        ;;
    restart)
        stop
        start
        ;;
    *)
        echo "Usage: ./origin.sh [start|stop|restart]"
        ;;
esac
