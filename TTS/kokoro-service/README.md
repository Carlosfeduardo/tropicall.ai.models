# Kokoro TTS Microservice

Real-time TTS microservice using Kokoro-82M for Brazilian Portuguese (pt-BR), optimized for LiveKit agents.

## Features

- **GPU-accelerated inference** with single consumer queue (serialized access)
- **WebSocket API** for streaming text-to-speech
- **Debounce-based text bundling** for optimal segment quality
- **Fairness control** with max inflight segments per session
- **Barge-in support** with request cancellation
- **Prometheus metrics** for observability

## Supported Voices (pt-BR)

| Voice ID | Gender | Notes |
|----------|--------|-------|
| `pf_dora` | Female | Recommended default |
| `pm_alex` | Male | Good general quality |
| `pm_santa` | Male | Alternative |

**Required:** `lang_code='p'` for Portuguese (pt-BR)

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
# Build (replace SHA with actual model revision)
docker build --build-arg HF_MODEL_REVISION=<sha> -t kokoro-tts .

# Run
docker run --gpus all -p 8000:8000 kokoro-tts
```

## WebSocket Protocol

### Client -> Server (JSON)

```json
// Start session
{"type": "start_session", "session_id": "uuid", "config": {"voice": "pf_dora", "lang_code": "p"}}

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
| `TTS_DEFAULT_VOICE` | pf_dora | Default voice |
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
                            KokoroWorker (GPU, ThreadPoolExecutor max=1)
                                    |
                                    v
                            Audio Streamer -> Binary frames + segment_done
```

# BUILD
```
cd /Users/carlosfernandes/Desktop/etc/tropicall.ai.models/TTS/kokoro-service && \
docker buildx build \
    --platform linux/amd64 \
    -t c02gkkvdmd6m/tropicall-ai-tts:latest \
    --push \
    --progress=plain \
    . 2>&1 | tee /tmp/kokoro_docker_build3.log
```


```
 cd /Users/carlosfernandes/Desktop/etc/tropicall.ai.models/TTS/kokoro-service && python3 livekit_load_test.py --sessions 1 --duration 30
 ```

## License

MIT
