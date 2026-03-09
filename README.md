# Aviation Disruption RAG System

A Retrieval-Augmented Generation (RAG) system for querying the impact of the 2026 Iran-US conflict on global civil aviation. Built with LangChain, ChromaDB, HuggingFace embeddings, and OpenAI GPT-4o.

**Live Demo:** [https://parupati-iran-us-aviation-rag.hf.space/docs](https://parupati-iran-us-aviation-rag.hf.space/docs)

**Portfolio Integration:** [https://madhukarparupati.web.app/aviationRag](https://madhukarparupati.web.app/aviationRag)

---

## Architecture

```
CSV Data (6 files)
  → Ingestion (ingest.py) — converts rows to natural language
  → Embedding (HuggingFace all-MiniLM-L6-v2, runs on CPU)
  → Vector Store (ChromaDB, local on disk)
  → Query (rag.py) — retrieves top-k chunks via similarity search
  → Generation (OpenAI GPT-4o) — produces grounded answers
  → FastAPI (api.py) — serves /query, /portfolio-chat, /health
  → Docker → Hugging Face Spaces
```

## Dataset

Source: [Global Civil Aviation Disruption 2026 — Kaggle](https://www.kaggle.com/datasets/zkskhurram/global-civil-aviation-disruption2026-iranus-war)

| File | Records | Description |
|---|---|---|
| `airline_losses_estimate.csv` | 35 | Daily financial losses per airline |
| `airport_disruptions.csv` | 35 | Disruptions at major airport hubs |
| `airspace_closures.csv` | 25 | Country-level airspace closures with NOTAM references |
| `conflict_events.csv` | 27 | Timeline of military events and aviation impact |
| `flight_cancellations.csv` | 47 | Cancelled flights with routes, aircraft, reasons |
| `flight_reroutes.csv` | 45 | Rerouted flights with extra distance, fuel cost, delay |

## Project Structure

```
IranUSAviationDisruptionRAG/
├── src/
│   ├── ingest.py              # CSV → natural language → ChromaDB
│   └── rag.py                 # RAG query engine (retrieval + LLM)
├── data/                      # 6 CSV dataset files
├── api.py                     # FastAPI backend (local development)
├── app.py                     # Streamlit UI (local development)
├── test_retrieval.py          # Vector store verification script
├── hf-space/                  # Hugging Face Spaces deployment
│   ├── api.py                 # FastAPI with /query + /portfolio-chat
│   ├── Dockerfile             # Docker build for HF Spaces
│   ├── portfolio_info.md      # Portfolio data for chat endpoint
│   ├── requirements.txt       # Python dependencies
│   ├── src/                   # ingest.py + rag.py (copy)
│   └── data/                  # CSV files (copy)
├── requirements.txt           # Local dev dependencies
├── Dockerfile                 # Local Docker build
└── INSTRUCTIONS.md            # Detailed local setup guide
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check, returns document count |
| `POST` | `/query` | RAG query — returns answer + source documents |
| `POST` | `/portfolio-chat` | Portfolio chatbot — conversational Q&A |
| `GET` | `/docs` | Swagger UI (auto-generated) |

### POST /query

```json
{
  "question": "Which airline had the highest daily financial loss?",
  "k": 5
}
```

Response:
```json
{
  "answer": "Emirates had the highest estimated daily loss at $4,200,000...",
  "sources": [
    { "category": "airline_losses", "content": "Emirates (UAE) faces..." }
  ]
}
```

### POST /portfolio-chat

```json
{
  "messages": [
    { "role": "user", "content": "What are Madhukar's skills?" }
  ]
}
```

Response:
```json
{
  "reply": "Madhukar is an expert in Angular, React, and JavaScript..."
}
```

## Local Development

### Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/parupati/IranUSAviationDisruptionRAG.git
cd IranUSAviationDisruptionRAG
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
echo OPENAI_API_KEY=sk-your-key > .env

# 4. Build vector store
python src/ingest.py

# 5. Run the API
python api.py
# → http://localhost:7860/docs
```

### Run Streamlit UI

```bash
streamlit run app.py
```

### Test retrieval (no API key needed)

```bash
python test_retrieval.py
```

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for detailed setup steps.

## Tech Stack

- **Python 3.12** — Runtime
- **LangChain** — RAG orchestration
- **ChromaDB** — Local vector database
- **HuggingFace sentence-transformers** — Embeddings (all-MiniLM-L6-v2, CPU)
- **OpenAI GPT-4o** — Answer generation
- **FastAPI + Uvicorn** — API server
- **Streamlit** — Local UI
- **Docker** — Containerization
- **Hugging Face Spaces** — Deployment

## License

Dataset is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
