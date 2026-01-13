"""
Tests for WebSocket chat endpoint.

These are integration tests that require a running Qwen3 service.
Run with: pytest tests/test_websocket.py -v
"""

import asyncio
import json
from uuid import uuid4

import pytest
import websockets


@pytest.mark.asyncio
async def test_websocket_connection(ws_url):
    """Test basic WebSocket connection."""
    async with websockets.connect(ws_url) as ws:
        # Connection should be established
        assert ws.open


@pytest.mark.asyncio
async def test_start_session(ws_url):
    """Test starting a chat session."""
    session_id = f"test-{uuid4().hex[:8]}"
    
    async with websockets.connect(ws_url) as ws:
        # Send start_session
        await ws.send(json.dumps({
            "type": "start_session",
            "session_id": session_id,
            "config": {
                "system_prompt": "You are a test assistant.",
            },
        }))
        
        # Should receive session_started
        response = await ws.recv()
        data = json.loads(response)
        
        assert data["type"] == "session_started"
        assert data["session_id"] == session_id
        
        # End session
        await ws.send(json.dumps({"type": "end_session"}))


@pytest.mark.asyncio
async def test_chat_streaming(ws_url):
    """Test streaming chat response."""
    session_id = f"test-{uuid4().hex[:8]}"
    
    async with websockets.connect(ws_url) as ws:
        # Start session
        await ws.send(json.dumps({
            "type": "start_session",
            "session_id": session_id,
            "config": {
                "generation": {"max_tokens": 50},
            },
        }))
        
        response = await ws.recv()
        assert json.loads(response)["type"] == "session_started"
        
        # Send message
        await ws.send(json.dumps({
            "type": "send_message",
            "content": "Say 'hello' and nothing else.",
        }))
        
        # Collect tokens
        tokens = []
        while True:
            response = await ws.recv()
            data = json.loads(response)
            
            if data["type"] == "token":
                tokens.append(data["token"])
            elif data["type"] == "message_done":
                break
            elif data["type"] == "error":
                pytest.fail(f"Received error: {data['message']}")
        
        # Should have received some tokens
        assert len(tokens) > 0
        
        # End session
        await ws.send(json.dumps({"type": "end_session"}))


@pytest.mark.asyncio
async def test_cancel_generation(ws_url):
    """Test cancelling an in-progress generation."""
    session_id = f"test-{uuid4().hex[:8]}"
    
    async with websockets.connect(ws_url) as ws:
        # Start session
        await ws.send(json.dumps({
            "type": "start_session",
            "session_id": session_id,
            "config": {
                "generation": {"max_tokens": 500},  # Long response
            },
        }))
        
        await ws.recv()  # session_started
        
        # Send message requesting long response
        await ws.send(json.dumps({
            "type": "send_message",
            "content": "Write a very long story about a cat.",
        }))
        
        # Wait for a few tokens
        token_count = 0
        for _ in range(5):
            response = await ws.recv()
            data = json.loads(response)
            if data["type"] == "token":
                token_count += 1
        
        # Cancel
        await ws.send(json.dumps({"type": "cancel"}))
        
        # Should be able to end session
        await ws.send(json.dumps({"type": "end_session"}))


@pytest.mark.asyncio
async def test_invalid_first_message(ws_url):
    """Test error handling for invalid first message."""
    async with websockets.connect(ws_url) as ws:
        # Send something other than start_session
        await ws.send(json.dumps({
            "type": "send_message",
            "content": "Hello!",
        }))
        
        # Should receive error
        response = await ws.recv()
        data = json.loads(response)
        
        assert data["type"] == "error"
        assert data["code"] == "INVALID_MESSAGE"
