#!/bin/bash
# Origin OS - Secure .env Manager
# Encrypts .env when not in use, only decrypts with your password

ENV_FILE="$HOME/.env"
ENCRYPTED_FILE="$HOME/.env.encrypted"
LOCK_FILE="$HOME/.env.lock"

encrypt_env() {
    if [ ! -f "$ENV_FILE" ]; then
        echo "❌ No .env file found"
        exit 1
    fi
    
    echo "🔒 Encrypting .env..."
    openssl enc -aes-256-cbc -salt -pbkdf2 -in "$ENV_FILE" -out "$ENCRYPTED_FILE"
    
    if [ $? -eq 0 ]; then
        # Securely delete original
        shred -u "$ENV_FILE" 2>/dev/null || rm -P "$ENV_FILE" 2>/dev/null || rm "$ENV_FILE"
        echo "✅ .env encrypted and original deleted"
        echo "📁 Encrypted file: $ENCRYPTED_FILE"
    else
        echo "❌ Encryption failed"
        exit 1
    fi
}

decrypt_env() {
    if [ ! -f "$ENCRYPTED_FILE" ]; then
        echo "❌ No encrypted .env found"
        exit 1
    fi
    
    echo "🔓 Decrypting .env..."
    openssl enc -aes-256-cbc -d -pbkdf2 -in "$ENCRYPTED_FILE" -out "$ENV_FILE"
    
    if [ $? -eq 0 ]; then
        # Set restrictive permissions
        chmod 600 "$ENV_FILE"
        echo "✅ .env decrypted"
        echo "⚠️  Remember to run './secure_env.sh lock' when done!"
    else
        echo "❌ Decryption failed (wrong password?)"
        rm -f "$ENV_FILE"
        exit 1
    fi
}

view_env() {
    if [ -f "$ENV_FILE" ]; then
        echo "📄 Current .env (unencrypted):"
        echo "================================"
        cat "$ENV_FILE"
    elif [ -f "$ENCRYPTED_FILE" ]; then
        echo "🔒 .env is encrypted. Decrypting temporarily..."
        openssl enc -aes-256-cbc -d -pbkdf2 -in "$ENCRYPTED_FILE" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "❌ Wrong password"
        fi
    else
        echo "❌ No .env file found"
    fi
}

status() {
    echo "=============================================="
    echo "🔐 SECURE ENV STATUS"
    echo "=============================================="
    
    if [ -f "$ENV_FILE" ]; then
        echo "⚠️  .env is UNLOCKED (unencrypted)"
        echo "   Run: ./secure_env.sh lock"
    elif [ -f "$ENCRYPTED_FILE" ]; then
        echo "✅ .env is LOCKED (encrypted)"
        echo "   Run: ./secure_env.sh unlock"
    else
        echo "❌ No .env file found"
    fi
}

case "${1:-status}" in
    lock|encrypt)
        encrypt_env
        ;;
    unlock|decrypt)
        decrypt_env
        ;;
    view)
        view_env
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: ./secure_env.sh [lock|unlock|view|status]"
        echo ""
        echo "Commands:"
        echo "  lock    - Encrypt .env (deletes plaintext)"
        echo "  unlock  - Decrypt .env for editing"
        echo "  view    - View contents (prompts for password if locked)"
        echo "  status  - Check if .env is locked/unlocked"
        ;;
esac
