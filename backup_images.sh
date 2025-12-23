#!/bin/bash
# Origin OS - Backup & Encrypt Container Images
# Usage: ./backup_images.sh [backup|restore]

set -e

BACKUP_DIR="$HOME/origin-os-backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="origin-os-images-$DATE.tar.gz"
ENCRYPTED_FILE="$BACKUP_FILE.enc"

# Images to backup
IMAGES=(
    "root1-ui"
    "root1-codex"
    "root1-mcp"
    "root1-vault"
    "supremex-landing-supremex-landing"
)

backup() {
    echo "=============================================="
    echo "🔐 ORIGIN OS - BACKUP & ENCRYPT"
    echo "=============================================="
    
    mkdir -p "$BACKUP_DIR"
    cd "$BACKUP_DIR"
    
    # Export images
    echo ""
    echo "📦 Exporting Docker images..."
    for img in "${IMAGES[@]}"; do
        if docker image inspect "$img" > /dev/null 2>&1; then
            echo "   Saving: $img"
            docker save "$img" >> "images-$DATE.tar"
        else
            echo "   ⚠️ Not found: $img"
        fi
    done
    
    # Compress
    echo ""
    echo "🗜️  Compressing..."
    gzip "images-$DATE.tar"
    mv "images-$DATE.tar.gz" "$BACKUP_FILE"
    
    # Encrypt with password
    echo ""
    echo "🔒 Encrypting backup..."
    echo "   Enter encryption password:"
    openssl enc -aes-256-cbc -salt -pbkdf2 -in "$BACKUP_FILE" -out "$ENCRYPTED_FILE"
    
    # Remove unencrypted backup
    rm "$BACKUP_FILE"
    
    # Show result
    SIZE=$(du -h "$ENCRYPTED_FILE" | cut -f1)
    echo ""
    echo "=============================================="
    echo "✅ BACKUP COMPLETE"
    echo "=============================================="
    echo "📁 Location: $BACKUP_DIR/$ENCRYPTED_FILE"
    echo "📊 Size: $SIZE"
    echo ""
    echo "To restore: ./backup_images.sh restore $ENCRYPTED_FILE"
}

restore() {
    RESTORE_FILE="$1"
    
    if [ -z "$RESTORE_FILE" ]; then
        echo "Usage: ./backup_images.sh restore <encrypted_file>"
        exit 1
    fi
    
    echo "=============================================="
    echo "🔓 ORIGIN OS - RESTORE FROM BACKUP"
    echo "=============================================="
    
    cd "$BACKUP_DIR"
    
    # Decrypt
    echo ""
    echo "🔓 Decrypting backup..."
    DECRYPTED="${RESTORE_FILE%.enc}"
    openssl enc -aes-256-cbc -d -pbkdf2 -in "$RESTORE_FILE" -out "$DECRYPTED"
    
    # Extract and load images
    echo ""
    echo "📦 Loading Docker images..."
    gunzip -c "$DECRYPTED" | docker load
    
    # Cleanup
    rm "$DECRYPTED"
    
    echo ""
    echo "=============================================="
    echo "✅ RESTORE COMPLETE"
    echo "=============================================="
    docker images | grep -E "root1-|supremex"
}

# Main
case "${1:-backup}" in
    backup)
        backup
        ;;
    restore)
        restore "$2"
        ;;
    *)
        echo "Usage: ./backup_images.sh [backup|restore <file>]"
        exit 1
        ;;
esac
