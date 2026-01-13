# Supertonic 2 TTS Microservice

Real-time TTS microservice using **Supertonic 2** for multilingual speech synthesis, optimized for LiveKit agents.

## Features

- **Lightning Fast**: Up to 167x faster than real-time (M4 Pro with WebGPU)
- **Lightweight**: Only 66M parameters
- **GPU-accelerated inference** via ONNX Runtime with CUDA EP
- **WebSocket API** for streaming text-to-speech
- **Multilingual Support**: English, Korean, Spanish, Portuguese, French
- **Debounce-based text bundling** for optimal segment quality
- **Fairness control** with max inflight segments per session
- **Barge-in support** with request cancellation
- **Prometheus metrics** for observability

## Supported Languages

| Language   | Code |
|------------|------|
| English    | en   |
| Korean     | ko   |
| Spanish    | es   |
| Portuguese | pt   |
| French     | fr   |

## Supported Voices

| Voice ID | Gender | Notes |
|----------|--------|-------|
| `F1` | Female | Default female |
| `F2` | Female | Alternative |
| `F3` | Female | Alternative |
| `F4` | Female | Alternative |
| `F5` | Female | Alternative |
| `M1` | Male | Default male |
| `M2` | Male | Alternative |
| `M3` | Male | Alternative |
| `M4` | Male | Alternative |
| `M5` | Male | Alternative |

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
# Build
docker build -t supertonic-tts .

# Run with GPU
docker run --gpus all -p 8000:8000 supertonic-tts
```

## WebSocket Protocol

### Client -> Server (JSON)

```json
// Start session
{"type": "start_session", "session_id": "uuid", "config": {"voice": "F1", "lang_code": "pt"}}

// Send text (can send multiple times)
{"type": "send_text", "text": "Olá, como você está?"}

// Flush pending text
{"type": "flush"}

// End session gracefully
{"type": "end_session"}

// Cancel immediately (barge-in)
{"type": "cancel"}
```

### Server -> Client

- **JSON**: Control messages (`session_started`, `segment_done`, `error`, `session_ended`)
- **Binary**: PCM 16-bit LE audio frames (480 samples = 20ms @ 24kHz)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TTS_MAX_SESSIONS` | 100 | Max concurrent sessions |
| `TTS_MAX_INFLIGHT_SEGMENTS` | 2 | Max segments in flight per session |
| `TTS_DEFAULT_VOICE` | F1 | Default voice |
| `TTS_LANG_CODE` | pt | Language code (en, ko, es, pt, fr) |
| `TTS_NUM_INFERENCE_STEPS` | 2 | Inference steps (2=fastest, 5=quality) |
| `TTS_DEBOUNCE_MS` | 150 | Debounce timer for text bundling |
| `TTS_CHUNK_SIZE_MS` | 20 | Audio chunk size in ms |
| `LOG_LEVEL` | INFO | Logging level |

## Health Endpoints

- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
- `GET /health/startup` - Startup probe

## Metrics

Prometheus metrics available at `GET /metrics`:

- `tts_time_to_first_audio_seconds` - TTFA histogram
- `tts_realtime_factor` - RTF histogram
- `tts_queue_depth` - Inference queue depth
- `tts_active_sessions` - Currently active sessions
- `tts_gpu_utilization_percent` - GPU utilization

## Architecture

```
LiveKit Agents -> WebSocket -> Session Manager -> DebouncedTextAccumulator
                                    |
                                    v
                            InferenceQueue (global, single consumer)
                                    |
                                    v
                            SupertonicWorker (ONNX Runtime, ThreadPoolExecutor max=1)
                                    |
                                    v
                            Audio Streamer -> Binary frames + segment_done
```

## Performance

Supertonic 2 benchmarks (2-step inference):

| Platform | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|----------|------------------|-----------------|------------------|
| M4 Pro - CPU | 912 chars/s | 1048 chars/s | 1263 chars/s |
| M4 Pro - WebGPU | 996 chars/s | 1801 chars/s | 2509 chars/s |
| RTX 4090 | 2615 chars/s | 6548 chars/s | 12164 chars/s |

Real-time Factor (lower is better):

| Platform | Short | Mid | Long |
|----------|-------|-----|------|
| M4 Pro - CPU | 0.015 | 0.013 | 0.012 |
| M4 Pro - WebGPU | 0.014 | 0.007 | 0.006 |
| RTX 4090 | 0.005 | 0.002 | 0.001 |

## LiveKit Plugin Usage

```python
from clients.livekit_plugin import SupertonicTTS

tts = SupertonicTTS(
    service_url="ws://supertonic-tts:8000/ws/tts",
    voice="F1",
    lang_code="pt",
)

# In your LiveKit agent:
agent = Agent(
    tts=tts,
    ...
)
```

## Build

```bash
cd /Users/carlosfernandes/Desktop/etc/tropicall.ai.models/TTS/supertonic-service && \
docker buildx build \
    --platform linux/amd64 \
    -t c02gkkvdmd6m/tropicall-ai-tts-supertonic:latest \
    --push \
    --progress=plain \
    . 2>&1 | tee /tmp/supertonic_docker_build.log
```

## References

- [Supertonic 2 on HuggingFace](https://huggingface.co/Supertone/supertonic-2)
- [Supertonic GitHub](https://github.com/supertone-inc/supertonic)

## License

MIT
