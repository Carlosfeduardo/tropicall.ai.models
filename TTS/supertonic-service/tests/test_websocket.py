"""
Tests for WebSocket API.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# We need to mock the inference queue before importing the app
with patch("src.inference.inference_queue.inference_queue", None):
    from src.main import app


class TestWebSocketProtocol:
    """Tests for WebSocket protocol."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_invalid_first_message(self, client):
        """Should reject non-start_session as first message."""
        with client.websocket_connect("/ws/tts") as ws:
            ws.send_json({"type": "send_text", "text": "test"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert data["code"] == "INVALID_MESSAGE"

    def test_invalid_json(self, client):
        """Should handle invalid JSON gracefully."""
        with client.websocket_connect("/ws/tts") as ws:
            ws.send_text("not json")
            # Should close connection or send error


class TestProtocolMessages:
    """Tests for protocol message parsing."""

    def test_start_session_message(self):
        """Should parse start_session correctly."""
        from src.protocol.messages import StartSessionMessage
        
        msg = StartSessionMessage(
            session_id="test-123",
            config={"voice": "pf_dora", "lang_code": "p"}
        )
        assert msg.session_id == "test-123"
        assert msg.config.voice == "pf_dora"

    def test_send_text_message(self):
        """Should parse send_text correctly."""
        from src.protocol.messages import SendTextMessage
        
        msg = SendTextMessage(text="Hello world")
        assert msg.text == "Hello world"

    def test_segment_done_message(self):
        """Should create segment_done with metrics."""
        from src.protocol.messages import SegmentDoneMessage
        
        msg = SegmentDoneMessage(
            segment_id="seg-001",
            request_id="req-001",
            ttfa_ms=45.2,
            rtf=0.18,
            audio_duration_ms=1200,
            total_samples=28800
        )
        
        assert msg.segment_id == "seg-001"
        assert msg.ttfa_ms == 45.2
        assert msg.rtf == 0.18

    def test_error_message(self):
        """Should create error messages correctly."""
        from src.protocol.messages import ErrorCode, ErrorMessage
        
        msg = ErrorMessage(
            code=ErrorCode.CLIENT_SLOW,
            message="Too many pending segments"
        )
        
        assert msg.code == ErrorCode.CLIENT_SLOW
