# Qwen3 LLM Microservice

Real-time LLM inference microservice with **Qwen3-32B-Instruct** via **vLLM**.

Production-ready service with:
- WebSocket streaming for real-time chat
- OpenAI-compatible HTTP API
- LiveKit Agents plugin
- Prometheus metrics
- Kubernetes/RunPod deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Qwen3 LLM Service                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  WebSocket   │  │   OpenAI     │  │   Health/Metrics     │  │
│  │  /ws/chat    │  │   /v1/...    │  │   /health, /metrics  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘  │
│         │                 │                                     │
│  ┌──────┴─────────────────┴──────┐                             │
│  │       Session Manager         │                             │
│  │   (conversation history)      │                             │
│  └──────────────┬────────────────┘                             │
│                 │                                               │
│  ┌──────────────┴────────────────┐                             │
│  │       Request Handler         │                             │
│  │    (admission control)        │                             │
│  └──────────────┬────────────────┘                             │
│                 │                                               │
│  ┌──────────────┴────────────────┐                             │
│  │      vLLM AsyncLLMEngine      │                             │
│  │    (Qwen3-32B-Instruct)       │                             │
│  └──────────────┬────────────────┘                             │
│                 │                                               │
│  ┌──────────────┴────────────────┐                             │
│  │         NVIDIA H100           │                             │
│  │           80GB                │                             │
│  └───────────────────────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Using Docker Compose

```bash
# Build and run
docker compose up -d

# Check health
curl http://localhost:8001/health/ready

# Test chat
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export LLM_MODEL_ID="Qwen/Qwen3-32B-Instruct"
export LLM_PORT=8001

# Run the service
python -m src.main
```

## API Reference

### WebSocket Endpoint: `/ws/chat`

Real-time streaming chat via WebSocket.

#### Protocol

**Client → Server:**

```json
// 1. Start session
{
  "type": "start_session",
  "session_id": "unique-session-id",
  "config": {
    "system_prompt": "You are a helpful assistant.",
    "generation": {
      "max_tokens": 2048,
      "temperature": 0.7
    }
  }
}

// 2. Send message
{
  "type": "send_message",
  "content": "Hello, how are you?"
}

// 3. Cancel (optional, for barge-in)
{
  "type": "cancel"
}

// 4. End session
{
  "type": "end_session"
}
```

**Server → Client:**

```json
// Session confirmed
{"type": "session_started", "session_id": "..."}

// Streamed tokens
{"type": "token", "token": "Hello", "request_id": "..."}
{"type": "token", "token": "!", "request_id": "..."}

// Generation complete
{
  "type": "message_done",
  "request_id": "...",
  "content": "Hello! How can I help you today?",
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
  "metrics": {"ttft_ms": 120.5, "total_time_ms": 850.2, "tokens_per_second": 45.3}
}

// Session ended
{
  "type": "session_ended",
  "reason": "completed",
  "metrics": {...}
}
```

### OpenAI-Compatible API: `/v1/chat/completions`

Drop-in replacement for OpenAI API.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"  # Optional
)

# Streaming
stream = client.chat.completions.create(
    model="qwen3-32b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Non-streaming
response = client.chat.completions.create(
    model="qwen3-32b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=False
)
print(response.choices[0].message.content)
```

### Health Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health/ready` | Readiness probe (model loaded) |
| `/health/live` | Liveness probe |
| `/health/startup` | Startup probe |
| `/health/status` | Detailed status with capacity info |

### Metrics

Prometheus metrics at `/metrics`:

- `llm_time_to_first_token_seconds` - TTFT histogram
- `llm_tokens_per_second` - Token generation rate
- `llm_request_latency_seconds` - Total request latency
- `llm_active_sessions` - Active WebSocket sessions
- `llm_pending_requests` - Pending requests in queue
- `llm_gpu_utilization_percent` - GPU utilization
- `llm_gpu_memory_used_bytes` - GPU memory usage

## LiveKit Agents Integration

Use with LiveKit Agents for voice AI:

```python
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.voice_assistant import VoicePipelineAgent
from livekit.plugins import silero

# Import the Qwen3 plugin
from clients.livekit_plugin import Qwen3LLM

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Create Qwen3 LLM
    llm = Qwen3LLM(
        base_url="http://qwen3-service:8001",
        temperature=0.7,
        max_tokens=512,
    )

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=...,  # Your STT
        llm=llm,
        tts=...,  # Your TTS
    )

    agent.start(ctx.room)
    await agent.say("Hello! How can I help you?")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## Configuration

Environment variables with `LLM_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_HOST` | `0.0.0.0` | Bind host |
| `LLM_PORT` | `8001` | Bind port |
| `LLM_MODEL_ID` | `Qwen/Qwen3-32B-Instruct` | Model ID |
| `LLM_MAX_MODEL_LEN` | `8192` | Max context length |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory fraction |
| `LLM_MAX_SESSIONS` | `50` | Max WebSocket sessions |
| `LLM_MAX_TOKENS` | `2048` | Default max tokens |
| `LLM_TEMPERATURE` | `0.7` | Default temperature |
| `LLM_ENABLE_THINKING` | `false` | Enable Qwen3 thinking mode |
| `LLM_MAX_CONCURRENT_REQUESTS` | `100` | Max concurrent requests |
| `LLM_LOG_LEVEL` | `INFO` | Logging level |

## Deployment

### RunPod

1. Build and push Docker image (cross-platform from MacBook to Linux):

```bash
# Build for linux/amd64 and push to registry
docker buildx build \
    --platform linux/amd64 \
    -t c02gkkvdmd6m/tropicall-ai-llm:latest \
    --push \
    --progress=plain \
    . 2>&1 | tee /tmp/qwen3_docker_build.log
```

> **Note:** We use `buildx` with `--platform linux/amd64` because development is on MacBook (ARM) 
> but the container runs on Linux x86_64 (RunPod/Kubernetes).

2. Create RunPod pod using `deploy/runpod/template.json`

3. Select **NVIDIA H100 80GB** GPU

### Kubernetes

Apply manifests from `deploy/k8s/`:

```bash
kubectl apply -f deploy/k8s/
```

## Performance

Typical performance on H100 80GB:

| Metric | Value |
|--------|-------|
| TTFT (p50) | ~100ms |
| TTFT (p95) | ~250ms |
| Tokens/sec | 40-60 |
| Max concurrent sessions | 50+ |
| GPU memory | ~65GB |

## Project Structure

```
qwen3-service/
├── src/
│   ├── main.py              # FastAPI app + lifespan
│   ├── config.py            # Settings
│   ├── api/
│   │   ├── websocket.py     # WebSocket endpoint
│   │   ├── openai_compat.py # OpenAI API
│   │   ├── health.py        # Health checks
│   │   └── metrics.py       # Prometheus
│   ├── core/
│   │   ├── session.py       # Session management
│   │   └── token_streamer.py
│   ├── inference/
│   │   ├── vllm_engine.py   # vLLM wrapper
│   │   └── request_handler.py
│   ├── observability/
│   │   ├── logging.py
│   │   └── metrics.py
│   └── protocol/
│       ├── errors.py
│       └── messages.py
├── clients/
│   ├── python_client.py     # Python client
│   └── livekit_plugin/      # LiveKit integration
├── deploy/
│   └── runpod/
│       └── template.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## License

MIT License
