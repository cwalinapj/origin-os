#!/usr/bin/env python3
"""
Origin OS - Conversation Logger & Training Data Generator
Captures conversations for local LLM fine-tuning
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Encrypted storage location
TRAINING_DATA_DIR = Path.home() / ".origin-os" / "training-data"

class ConversationLogger:
    """
    Logs conversations in a format suitable for LLM fine-tuning
    
    Output format (JSONL - one per line):
    {"messages": [...], "metadata": {...}}
    """
    
    def __init__(self, data_dir: Path = TRAINING_DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_file = self.data_dir / "conversations.jsonl"
        
        # Set restrictive permissions (owner read/write only)
        os.chmod(self.data_dir, 0o700)
    
    def log_conversation(
        self,
        messages: List[Dict],
        tools_used: List[str] = None,
        services: List[str] = None,
        success: bool = True
    ):
        """Log a conversation for training"""
        
        content_hash = hashlib.sha256(
            json.dumps(messages).encode()
        ).hexdigest()[:16]
        
        entry = {
            "id": content_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "messages": messages,
            "metadata": {
                "tools_used": tools_used or [],
                "services": services or [],
                "success": success
            }
        }
        
        with open(self.conversations_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        return content_hash
    
    def export_for_training(self, output_file: str = None) -> str:
        """Export conversations in OpenAI fine-tuning format"""
        output_file = output_file or str(self.data_dir / "training_export.jsonl")
        
        training_data = []
        
        if not self.conversations_file.exists():
            return output_file
            
        with open(self.conversations_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                
                system_msg = {
                    "role": "system",
                    "content": "You are Origin OS, an AI that manages GTM, deploys code, and automates tasks."
                }
                
                training_data.append({
                    "messages": [system_msg] + entry["messages"]
                })
        
        with open(output_file, "w") as f:
            for entry in training_data:
                f.write(json.dumps(entry) + "\n")
        
        return output_file
    
    def get_stats(self) -> Dict:
        """Get training data statistics"""
        if not self.conversations_file.exists():
            return {"conversations": 0, "messages": 0}
        
        conversations = 0
        messages = 0
        
        with open(self.conversations_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                conversations += 1
                messages += len(entry["messages"])
        
        return {
            "conversations": conversations,
            "messages": messages,
            "file_size": self.conversations_file.stat().st_size
        }
