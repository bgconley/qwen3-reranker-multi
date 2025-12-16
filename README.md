# Qwen3-Reranker Multi-Backend Service

High-performance HTTP reranker service running **Qwen3-Reranker-4B** with multiple backend support.

## Overview

This service provides a standalone reranker API that is **drop-in compatible** with the `wekadocs-matrix` reranker provider interface. It supports multiple inference backends:

| Backend | Platform | Performance | Use Case |
|---------|----------|-------------|----------|
| **PyTorch** (PRIMARY) | CUDA/MPS/CPU | ~30-50ms/batch (CUDA) | Lambda Cloud, production NVIDIA |
| **vLLM** (SECONDARY) | CUDA only | ~20-40ms/batch | High-throughput production |
| **MLX** (TERTIARY) | Apple Silicon | ~50-100ms/batch | macOS local development |

### Key Features

- **Multi-backend**: PyTorch (CUDA/MPS/CPU), vLLM (high-throughput), MLX (Apple Silicon)
- **Auto-detection**: Automatically selects best available backend
- **Yes/No probability scoring**: Official Qwen3 reranker scoring method
- **Batched inference**: Configurable batch sizes per backend
- **Concurrency guard**: Prevents memory spikes from concurrent requests
- **Docker support**: CUDA and vLLM Dockerfiles included
- **Score parity**: All backends produce equivalent scores (within tolerance)

## Requirements

### For Lambda Cloud / CUDA Deployment (Recommended)
- **Linux** with NVIDIA GPU (SM 8.0+ for Flash Attention)
- **Python 3.11+**
- **~10-12 GB** GPU memory

### For Apple Silicon Development
- **macOS** on Apple Silicon (M1/M2/M3)
- **Python 3.11+**
- **~10-12 GB** unified memory

## Installation

```bash
# Clone the repository
cd /path/to/qwen3-reranker-multi

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install with desired backend(s)
pip install -e ".[cuda]"      # PyTorch + Flash Attention (Lambda Cloud)
pip install -e ".[vllm]"      # vLLM (high-throughput CUDA)
pip install -e ".[mlx]"       # MLX (Apple Silicon)
pip install -e ".[all]"       # All backends

# For development
pip install -e ".[dev]"
```

## Quick Start

### Option 1: Lambda Cloud / CUDA

```bash
# Set profile for CUDA
export QWEN_RERANK_PROFILE=qwen3_4b_cuda
./scripts/run_cuda.sh
```

### Option 2: vLLM (High-Throughput)

```bash
# For high-throughput production
export QWEN_RERANK_PROFILE=qwen3_4b_vllm
./scripts/run_vllm.sh
```

### Option 3: Apple Silicon (MLX)

```bash
# For macOS development
export QWEN_RERANK_PROFILE=qwen3_4b_mlx_fp16
./scripts/run_mlx.sh
```

### Option 4: Auto-Detect

```bash
# Auto-detect best backend
./scripts/run_dev.sh
```

### Verify Service

```bash
# Check health
curl http://localhost:9003/health

# Check backend info
curl http://localhost:9003/healthz
```

### Send a Rerank Request

```bash
curl -X POST http://localhost:9003/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany.",
      "The Eiffel Tower is in Paris."
    ]
  }'
```

Response:
```json
{
  "results": [
    {"index": 0, "score": 0.9876},
    {"index": 2, "score": 0.8234},
    {"index": 1, "score": 0.0123}
  ],
  "model": "Qwen/Qwen3-Reranker-4B",
  "meta": {
    "max_length": 8192,
    "batch_size": 16,
    "scoring": "p_yes_softmax(no,yes)",
    "truncated_docs": 0,
    "elapsed_ms": 42.5
  }
}
```

## Docker Deployment

### PyTorch CUDA

```bash
# Build
docker build -t qwen3-reranker:cuda .

# Run
docker run --gpus all -p 9003:9003 qwen3-reranker:cuda
```

### vLLM

```bash
# Build
docker build -t qwen3-reranker:vllm -f Dockerfile.vllm .

# Run
docker run --gpus all -p 9003:9003 qwen3-reranker:vllm
```

### Docker Compose

```bash
# Run PyTorch CUDA
docker-compose up qwen3-reranker-cuda

# Run vLLM
docker-compose --profile vllm up qwen3-reranker-vllm
```

## API Endpoints

### POST /v1/rerank

Rerank documents for a query. **Compatible with wekadocs-matrix**.

**Request:**
```json
{
  "query": "string",
  "documents": ["doc1", "doc2", ...],
  "model": "string (optional, logged)",
  "instruction": "string (optional)",
  "top_n": 10,
  "return_documents": false,
  "max_length": 8192
}
```

**Response:**
```json
{
  "results": [
    {"index": 0, "score": 0.95},
    {"index": 2, "score": 0.82}
  ],
  "model": "Qwen/Qwen3-Reranker-4B",
  "meta": {...}
}
```

### GET /health
Basic health check (wekadocs-compatible).

### GET /ready
Readiness probe (returns 200 only after warmup).

### GET /healthz
Detailed health with backend info.

### GET /v1/config
Current service configuration.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN_RERANK_BACKEND` | `auto` | Backend: auto, pytorch, vllm, mlx |
| `QWEN_RERANK_PROFILE` | `qwen3_4b_cuda` | Profile from `reranker_profiles.yaml` |
| `QWEN_RERANK_PORT` | `9003` | Service port |
| `QWEN_RERANK_HOST` | `0.0.0.0` | Bind host |
| `QWEN_RERANK_LOG_LEVEL` | `INFO` | Log level |
| `QWEN_RERANK_DEVICE` | (from profile) | PyTorch device override |
| `QWEN_RERANK_MAX_LENGTH` | (from profile) | Max sequence length |
| `QWEN_RERANK_BATCH_SIZE` | (from profile) | Batch size |

### Profiles

Profiles are defined in `config/reranker_profiles.yaml`:

**PyTorch (PRIMARY)**
- `qwen3_4b_cuda`: CUDA with Flash Attention (recommended)
- `qwen3_4b_mps`: Apple Silicon via PyTorch
- `qwen3_4b_cpu`: CPU fallback

**vLLM (SECONDARY)**
- `qwen3_4b_vllm`: Single GPU high-throughput
- `qwen3_4b_vllm_multi_gpu`: Multi-GPU tensor parallelism

**MLX (TERTIARY)**
- `qwen3_4b_mlx_fp16`: FP16 (best quality)
- `qwen3_4b_mlx_8bit`: 8-bit quantized
- `qwen3_4b_mlx_4bit`: 4-bit quantized (lowest memory)

## Integration with wekadocs-matrix

This service is designed as a drop-in replacement for existing rerankers:

```bash
# In wekadocs-matrix
export RERANK_PROVIDER=bge-reranker-service
export RERANKER_BASE_URL=http://your-server:9003
```

## Performance Comparison

| Backend | Platform | Latency/batch | Memory | Notes |
|---------|----------|---------------|--------|-------|
| PyTorch CUDA | Linux | ~30-50ms | 9-10GB | Flash Attention 2 |
| vLLM | Linux | ~20-40ms | 8-9GB | Continuous batching |
| PyTorch MPS | macOS | ~150-300ms | 10-12GB | 2-3x slower than MLX |
| MLX | macOS | ~50-100ms | 9-10GB | Best for Apple Silicon |
| PyTorch CPU | Any | ~500ms+ | 12-16GB | Fallback only |

## Architecture

```
src/qwen3_reranker/
├── __init__.py
├── version.py
├── api/
│   ├── __init__.py
│   ├── app.py           # FastAPI application
│   └── models.py        # Pydantic schemas
├── backends/
│   ├── __init__.py
│   ├── base.py          # Backend Protocol
│   ├── registry.py      # Auto-detection
│   ├── pytorch_backend.py  # PRIMARY
│   ├── vllm_backend.py     # SECONDARY
│   └── mlx_backend.py      # TERTIARY
└── core/
    ├── __init__.py
    ├── config.py        # Configuration
    ├── errors.py        # Exceptions
    ├── prompt.py        # Prompt formatting
    ├── scoring.py       # Yes/no scoring
    ├── tokenization.py  # Tokenization
    └── batching.py      # Request batching
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=qwen3_reranker
```

### Code Quality

```bash
ruff format src tests
ruff check src tests
mypy src
```

## License

MIT
