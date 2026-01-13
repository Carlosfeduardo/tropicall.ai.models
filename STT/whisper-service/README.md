# Whisper STT Microservice

Real-time Speech-to-Text microservice using Faster-Whisper (CTranslate2) for Brazilian Portuguese, optimized for LiveKit agents.

## Features

- **GPU-accelerated inference** with Faster-Whisper (CTranslate2)
- **Streaming WebSocket API** with partial and final transcripts
- **VAD-based endpointing** using Silero VAD
- **Admission control** with queue depth and latency limits
- **Low latency**: Partial updates ~200-400ms, final within ~1s
- **High concurrency**: 50+ connected sessions, multiple active talkers
- **Prometheus metrics** for observability

## Model Options

| Model | Speed | Quality | VRAM | Notes |
|-------|-------|---------|------|-------|
| `distil-large-v3` | Fast | Good | ~2GB | **Recommended** for streaming |
| `large-v3` | Slow | Best | ~4GB | Best accuracy |
| `medium` | Medium | Good | ~2GB | Balanced |
| `small` | Fast | Fair | ~1GB | Low resource |
| `base` | Fastest | Basic | ~0.5GB | Testing only |

Set via environment: `STT_WHISPER_MODEL=distil-large-v3`

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the service
python -m src.main
```

### Docker

```bash
# Build with default model (distil-large-v3)
docker build -t whisper-stt .

# Build with specific model
docker build --build-arg WHISPER_MODEL=large-v3 -t whisper-stt .

# Run
docker run --gpus all -p 8000:8000 whisper-stt
```

### RunPod

1. Use the Docker image or build from this Dockerfile
2. Expose port 8000
3. Set environment variables as needed
4. Health endpoint: `/health/ready`

## WebSocket Protocol

### Endpoint

```
ws://<host>:8000/ws/stt
```

### Session Lifecycle

1. Client connects and sends `start_session`
2. Server responds with `session_started`
3. Client streams binary audio chunks (PCM16)
4. Server sends `vad_event`, `partial_transcript`, `final_transcript`
5. Client sends `flush` or `end_session`
6. Server sends `session_ended`

### Client -> Server Messages (JSON)

#### start_session
```json
{
  "type": "start_session",
  "session_id": "unique-session-id",
  "config": {
    "lang_code": "pt",
    "sample_rate": 16000,
    "vad_enabled": true,
    "vad_threshold": 0.5,
    "partial_results": true,
    "word_timestamps": true
  }
}
```

#### flush
Force finalization of current utterance:
```json
{"type": "flush"}
```

#### end_session
Graceful close:
```json
{"type": "end_session"}
```

#### cancel
Drop buffers and stop processing:
```json
{"type": "cancel"}
```

### Client -> Server Audio (Binary)

- **Format**: PCM16 little-endian, mono, 16kHz
- **Recommended chunk size**: 20ms = 640 samples = 1280 bytes
- Send continuously during speech

### Server -> Client Messages (JSON)

#### session_started
```json
{
  "type": "session_started",
  "session_id": "unique-session-id"
}
```

#### vad_event
```json
{
  "type": "vad_event",
  "session_id": "unique-session-id",
  "state": "speech_start",  // or "speech_end"
  "t_ms": 1234.56
}
```

#### partial_transcript
```json
{
  "type": "partial_transcript",
  "session_id": "unique-session-id",
  "segment_id": "seg-abc123",
  "text": "Olá como",
  "confidence": 0.95,
  "is_final": false,
  "timing": {
    "preprocess_ms": 1.2,
    "queue_wait_ms": 5.3,
    "inference_ms": 145.6,
    "postprocess_ms": 0.8,
    "server_total_ms": 152.9
  }
}
```

#### final_transcript
```json
{
  "type": "final_transcript",
  "session_id": "unique-session-id",
  "segment_id": "seg-abc123",
  "text": "Olá, como você está?",
  "confidence": 0.98,
  "is_final": true,
  "words": [
    {"word": "Olá", "start": 0.0, "end": 0.4, "probability": 0.99},
    {"word": "como", "start": 0.5, "end": 0.8, "probability": 0.97},
    {"word": "você", "start": 0.9, "end": 1.2, "probability": 0.96},
    {"word": "está", "start": 1.3, "end": 1.6, "probability": 0.98}
  ],
  "audio_duration_ms": 1800.0,
  "timing": {
    "preprocess_ms": 1.5,
    "queue_wait_ms": 8.2,
    "inference_ms": 280.4,
    "postprocess_ms": 1.1,
    "server_total_ms": 291.2
  }
}
```

#### error
```json
{
  "type": "error",
  "code": "QUEUE_FULL",
  "message": "Queue full: 20/20",
  "session_id": "unique-session-id"
}
```

Error codes: `INVALID_MESSAGE`, `SESSION_EXISTS`, `MAX_SESSIONS_REACHED`, `QUEUE_FULL`, `QUEUE_CONGESTION`, `INFERENCE_ERROR`, `INTERNAL_ERROR`

#### session_ended
```json
{
  "type": "session_ended",
  "session_id": "unique-session-id",
  "reason": "completed",  // or "cancelled", "error", "timeout"
  "total_audio_ms": 45000.0,
  "total_segments": 12
}
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `STT_HOST` | `0.0.0.0` | Host to bind to |
| `STT_PORT` | `8000` | Port to bind to |
| `STT_MAX_SESSIONS` | `100` | Maximum concurrent sessions |
| `STT_WHISPER_MODEL` | `distil-large-v3` | Whisper model to use |
| `STT_COMPUTE_TYPE` | `float16` | Compute type (`float16`, `int8_float16`) |
| `STT_DEVICE` | `cuda` | Device (`cuda`, `cpu`) |
| `STT_LANGUAGE` | `pt` | Default language code |
| `STT_VAD_ENABLED` | `true` | Enable VAD |
| `STT_VAD_THRESHOLD` | `0.5` | VAD threshold (0.0-1.0) |
| `STT_VAD_MIN_SPEECH_MS` | `250` | Min speech duration (ms) |
| `STT_VAD_MIN_SILENCE_MS` | `500` | Min silence for end-of-speech (ms) |
| `STT_PARTIAL_INTERVAL_MS` | `300` | Partial transcript interval (ms) |
| `STT_MAX_PARTIAL_WINDOW_S` | `8` | Max audio window for partials (s) |
| `STT_MAX_UTTERANCE_S` | `30` | Max utterance duration (s) |
| `STT_MAX_BUFFER_MS` | `5000` | Max audio buffer per session (ms) |
| `STT_MAX_QUEUE_DEPTH` | `20` | Max queue depth |
| `STT_MAX_ESTIMATED_SERVER_TOTAL_MS` | `500` | Max estimated latency (ms) |
| `STT_LOG_LEVEL` | `INFO` | Logging level |

## Health Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Service info |
| `GET /health/ready` | Readiness probe (model loaded) |
| `GET /health/live` | Liveness probe |
| `GET /health/startup` | Startup probe |
| `GET /health/status` | Detailed status with queue metrics |
| `GET /metrics` | Prometheus metrics |

### /health/status Response

```json
{
  "status": "ready",
  "accepting_requests": true,
  "model": {
    "model_size": "distil-large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "ready": true
  },
  "queue": {
    "depth": 2,
    "avg_partial_ms": 145.5,
    "avg_final_ms": 280.3,
    "estimated_server_total_ms": 425.8,
    "accepting_requests": true,
    "total_requests": 1234,
    "rejected_requests": 5
  },
  "sessions": {
    "active": 15,
    "max": 100
  },
  "capacity": {
    "estimated_max_concurrent": 3,
    "current_load_percent": 66.7
  }
}
```

## Example Client (Python)

```python
import asyncio
import json
import websockets

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/ws/stt") as ws:
        # Start session
        await ws.send(json.dumps({
            "type": "start_session",
            "session_id": "my-session",
            "config": {"lang_code": "pt"}
        }))
        
        # Wait for confirmation
        response = json.loads(await ws.recv())
        assert response["type"] == "session_started"
        
        # Stream audio (PCM16, 16kHz, mono)
        with open("audio.raw", "rb") as f:
            while chunk := f.read(1280):  # 20ms chunks
                await ws.send(chunk)
                await asyncio.sleep(0.02)  # Real-time pacing
        
        # Flush and end
        await ws.send(json.dumps({"type": "flush"}))
        await ws.send(json.dumps({"type": "end_session"}))
        
        # Receive transcripts
        async for msg in ws:
            data = json.loads(msg)
            if data["type"] == "final_transcript":
                print(f"Transcript: {data['text']}")
            elif data["type"] == "session_ended":
                break

asyncio.run(stream_audio())
```

## LiveKit Integration

```python
from clients.livekit_plugin import WhisperSTT

stt = WhisperSTT(
    service_url="ws://whisper-stt:8000/ws/stt",
    language="pt",
)

# Use in LiveKit agent
agent = Agent(
    stt=stt,
    # ...
)
```

## Tuning Guide

### Latency Optimization

1. **Use distil-large-v3** for best speed/quality tradeoff
2. **Reduce partial_interval_ms** (e.g., 200ms) for faster updates
3. **Reduce max_partial_window_s** to limit re-decode scope
4. **Use int8_float16** compute type if GPU memory is tight

### Throughput Optimization

1. **Increase max_queue_depth** for bursty workloads
2. **Increase max_estimated_server_total_ms** to allow more queuing
3. **Use a larger GPU** (A100/H100) for lower inference time

### Quality Optimization

1. **Use large-v3** model for best accuracy
2. **Increase beam_size** (5-10) for better decoding
3. **Lower vad_threshold** (0.3-0.4) to catch quieter speech
4. **Increase vad_min_silence_ms** (700-1000) for slower speakers

### Memory Optimization

1. **Use smaller model** (medium, small)
2. **Use int8_float16** compute type
3. **Reduce max_sessions** limit
4. **Reduce max_buffer_ms** per session

## Troubleshooting

### "Model not ready" error
- Wait for startup to complete (check `/health/startup`)
- Ensure GPU has enough memory for the model
- Check logs for model loading errors

### High latency
- Check `/health/status` for queue depth
- Reduce `max_queue_depth` to reject early
- Use faster model (distil-large-v3)

### "QUEUE_FULL" errors
- Too many concurrent active talkers
- Increase `max_queue_depth` or reduce load
- Scale horizontally with multiple instances

### Poor transcription quality
- Ensure audio is 16kHz mono PCM16
- Check VAD threshold (might be too aggressive)
- Try a larger model (large-v3)
- Ensure language is set correctly

### GPU memory errors
- Use smaller model or int8_float16 compute type
- Reduce max_partial_window_s
- Restart the service to clear memory

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                           │
├─────────────────────────────────────────────────────────────────┤
│  WebSocket Endpoint (/ws/stt)                                   │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────┐                                               │
│  │Session Manager│─────► Session 1 ────┐                        │
│  └──────────────┘       Session 2 ────┼──► Audio Buffer        │
│                         Session N ────┘         │               │
│                                                 ▼               │
│                                           ┌──────────┐          │
│                                           │Silero VAD│          │
│                                           └──────────┘          │
│                                                 │               │
│                                                 ▼               │
│                                        ┌─────────────────┐      │
│                                        │Utterance Aggregator│   │
│                                        └─────────────────┘      │
│                                                 │               │
│                                    ┌────────────┴────────────┐  │
│                                    ▼                         ▼  │
│                              Partial Job              Final Job │
│                                    │                         │  │
│                                    └──────────┬──────────────┘  │
│                                               ▼                 │
│                                    ┌─────────────────┐          │
│                                    │ Inference Queue │          │
│                                    │ (Single Consumer)│         │
│                                    └─────────────────┘          │
│                                               │                 │
│                                               ▼                 │
│                                    ┌─────────────────┐          │
│                                    │ Whisper Worker  │          │
│                                    │ (GPU, ThreadPool=1)│       │
│                                    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## License

MIT
