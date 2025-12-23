#!/usr/bin/env python3
"""
Origin OS UI - Multi-LLM Chat Interface
Uses OpenRouter for multiple LLM access
"""

import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import httpx

app = FastAPI(title="Origin OS")

# OpenRouter API (supports Claude, GPT-4, Gemini, etc.)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

conversations: Dict[str, List] = {}

SYSTEM_PROMPT = """You are Origin OS, an AI assistant that helps manage Google Tag Manager, web scraping, and automation tasks. Be concise and helpful."""


async def call_llm(messages: List[Dict], model: str = "anthropic/claude-3.5-sonnet") -> str:
    """Call LLM via OpenRouter (supports multiple models)"""
    
    if not OPENROUTER_API_KEY:
        return "⚠️ OPENROUTER_API_KEY not configured"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Origin OS"
                },
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    "max_tokens": 4096
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            return f"API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error: {e}"


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Origin OS</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a; color: #e0e0e0;
            height: 100vh; display: flex; flex-direction: column;
        }
        header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 1rem 2rem; border-bottom: 1px solid #333;
            display: flex; align-items: center; gap: 1rem;
        }
        .logo { font-size: 1.5rem; font-weight: bold; color: #00d4ff; }
        .model-select {
            margin-left: auto;
            padding: 0.5rem; background: #1a1a1a; color: #e0e0e0;
            border: 1px solid #333; border-radius: 4px;
        }
        main { flex: 1; display: flex; flex-direction: column; padding: 1rem 2rem; overflow: hidden; }
        .messages { flex: 1; overflow-y: auto; margin-bottom: 1rem; }
        .message {
            max-width: 80%; margin-bottom: 1rem;
            padding: 1rem; border-radius: 12px; line-height: 1.6;
            white-space: pre-wrap;
        }
        .message.user { background: #1a3a5c; margin-left: auto; }
        .message.assistant { background: #1a1a1a; border: 1px solid #333; }
        .message code { background: #000; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.9rem; }
        .message pre { background: #000; padding: 1rem; border-radius: 8px; overflow-x: auto; margin: 0.5rem 0; }
        .input-area { display: flex; gap: 1rem; }
        input {
            flex: 1; padding: 1rem; border: 1px solid #333;
            border-radius: 8px; background: #1a1a1a; color: #e0e0e0; font-size: 1rem;
        }
        input:focus { outline: none; border-color: #00d4ff; }
        button {
            padding: 1rem 2rem; background: #00d4ff; color: #000;
            border: none; border-radius: 8px; font-weight: bold; cursor: pointer;
        }
        button:hover { background: #00b8e6; }
        button:disabled { background: #333; color: #666; }
        .status { font-size: 0.8rem; color: #666; margin-top: 0.5rem; }
    </style>
</head>
<body>
    <header>
        <div class="logo">⚡ Origin OS</div>
        <select class="model-select" id="model">
            <option value="anthropic/claude-3.5-sonnet">Claude 3.5 Sonnet</option>
            <option value="openai/gpt-4-turbo">GPT-4 Turbo</option>
            <option value="google/gemini-pro-1.5">Gemini Pro 1.5</option>
            <option value="meta-llama/llama-3.1-70b-instruct">Llama 3.1 70B</option>
            <option value="mistralai/mixtral-8x7b-instruct">Mixtral 8x7B</option>
        </select>
    </header>
    <main>
        <div class="messages" id="messages">
            <div class="message assistant">👋 Welcome to Origin OS! I'm connected via OpenRouter - you can switch models above.

Try asking me to:
• Create GTM tags
• Write Python code
• Help with automation
• Anything else!</div>
        </div>
        <div class="input-area">
            <input type="text" id="input" placeholder="Type a message..." onkeypress="if(event.key==='Enter')sendMessage()">
            <button onclick="sendMessage()" id="btn">Send</button>
        </div>
        <div class="status" id="status">Ready</div>
    </main>
    <script>
        function formatMessage(text) {
            // Code blocks
            text = text.replace(/```(\\w+)?\\n([\\s\\S]*?)```/g, '<pre><code>$2</code></pre>');
            // Inline code
            text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
            // Bold
            text = text.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
            return text;
        }
        
        async function sendMessage() {
            const input = document.getElementById('input');
            const model = document.getElementById('model').value;
            const msg = input.value.trim();
            if (!msg) return;
            
            input.value = '';
            document.getElementById('messages').innerHTML += `<div class="message user">${msg}</div>`;
            document.getElementById('btn').disabled = true;
            document.getElementById('status').textContent = `Thinking with ${model.split('/')[1]}...`;
            
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg, model: model})
                });
                const data = await res.json();
                document.getElementById('messages').innerHTML += `<div class="message assistant">${formatMessage(data.response)}</div>`;
                document.getElementById('status').textContent = `Response from ${model.split('/')[1]}`;
            } catch(e) {
                document.getElementById('messages').innerHTML += `<div class="message assistant">Error: ${e}</div>`;
                document.getElementById('status').textContent = 'Error';
            }
            
            document.getElementById('btn').disabled = false;
            document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
        }
    </script>
</body>
</html>
"""


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    model = data.get("model", "anthropic/claude-3.5-sonnet")
    
    if not message:
        raise HTTPException(status_code=400, detail="No message")
    
    session_id = "default"
    if session_id not in conversations:
        conversations[session_id] = []
    
    conversations[session_id].append({"role": "user", "content": message})
    if len(conversations[session_id]) > 20:
        conversations[session_id] = conversations[session_id][-20:]
    
    response = await call_llm(conversations[session_id], model)
    conversations[session_id].append({"role": "assistant", "content": response})
    
    return JSONResponse({"response": response})


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
