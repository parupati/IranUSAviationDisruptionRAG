---
title: Aviation Disruption RAG API
emoji: ✈️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# Hugging Face Spaces — Deployment Guide

This folder contains everything needed to deploy the Aviation Disruption RAG API as a Docker-based Hugging Face Space.

**Live Space:** [https://huggingface.co/spaces/parupati/iran-us-aviation-rag](https://huggingface.co/spaces/parupati/iran-us-aviation-rag)

**API Docs:** [https://parupati-iran-us-aviation-rag.hf.space/docs](https://parupati-iran-us-aviation-rag.hf.space/docs)

---

## Prerequisites

- A [Hugging Face](https://huggingface.co) account
- `huggingface_hub` Python package (`pip install huggingface_hub`)
- Logged in via `huggingface-cli login`
- An OpenAI API key

## Files

```
hf-space/
├── api.py              # FastAPI app with /query, /portfolio-chat, /health
├── Dockerfile          # Docker build config
├── requirements.txt    # Python dependencies
├── portfolio_info.md   # Portfolio data for the chat endpoint
├── src/
│   ├── ingest.py       # CSV ingestion + vector store builder
│   └── rag.py          # RAG retrieval + LLM chain
└── data/               # 6 CSV dataset files
```

## Initial Setup (One-Time)

### 1. Create the Space

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    "iran-us-aviation-rag",
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)
```

### 2. Upload Files

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="hf-space",
    repo_id="parupati/iran-us-aviation-rag",
    repo_type="space",
    ignore_patterns=[".git*"],
)
```

### 3. Set the OpenAI API Key as a Secret

```python
from huggingface_hub import HfApi

api = HfApi()
api.add_space_secret(
    repo_id="parupati/iran-us-aviation-rag",
    key="OPENAI_API_KEY",
    value="sk-your-openai-key-here",
)
```

Or set it manually at: Settings → Repository Secrets → New Secret

## Deploying Updates

### Upload changed files

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="hf-space",
    repo_id="parupati/iran-us-aviation-rag",
    repo_type="space",
    ignore_patterns=[".git*"],
)
```

### Restart the Space (reuses cached Docker image)

Use when you only changed a Space secret or want a quick restart. Does NOT pick up code/file changes.

```python
from huggingface_hub import HfApi

api = HfApi()
api.restart_space("parupati/iran-us-aviation-rag")
```

### Factory Rebuild (rebuilds Docker image from scratch)

Use when you changed `Dockerfile`, `requirements.txt`, `api.py`, source files, or data files.

```python
from huggingface_hub import HfApi

api = HfApi()
api.restart_space("parupati/iran-us-aviation-rag", factory_reboot=True)
```

## Building the Docker Image Locally

```bash
cd hf-space

# Build
docker build -t aviation-rag .

# Run (replace with your actual key)
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-your-key aviation-rag

# Test
curl http://localhost:7860/health
```

## Monitoring

- **Build logs:** https://huggingface.co/spaces/parupati/iran-us-aviation-rag/logs/build
- **Runtime logs:** https://huggingface.co/spaces/parupati/iran-us-aviation-rag/logs/container
- **Space settings:** https://huggingface.co/spaces/parupati/iran-us-aviation-rag/settings

## Common Operations Reference

| Task | Command |
|---|---|
| Upload files | `api.upload_folder(folder_path="hf-space", repo_id="...", repo_type="space")` |
| Set a secret | `api.add_space_secret(repo_id="...", key="...", value="...")` |
| Simple restart | `api.restart_space("...")` |
| Full rebuild | `api.restart_space("...", factory_reboot=True)` |
| Check status | `api.get_space_runtime("...")` |
| Pause Space | `api.pause_space("...")` |
| Resume Space | `api.resume_space("...")` |

## Notes

- Free HF Spaces sleep after ~48 hours of inactivity. First request after sleep takes 30-60 seconds (cold start).
- The vector store is built during `docker build` (in the Dockerfile's `RUN python src/ingest.py`), so it's baked into the image.
- `portfolio_info.md` is loaded at startup and kept in memory for the `/portfolio-chat` endpoint.
- CORS is set to `allow_origins=["*"]` to allow calls from any frontend domain.
