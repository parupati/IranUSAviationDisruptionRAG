---
title: Building a RAG System from Scratch: Turning Aviation Disruption Data into an AI-Powered Q&A App
published: true
tags: rag, ai, python, langchain
series:
canonical_url:
cover_image:
---

I recently built a Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions about the 2026 Iran-US conflict's impact on global civil aviation — and get accurate, source-backed answers in seconds.

**Try the live demo:** [https://parupati.com/aviationRag](https://parupati.com/aviationRag)

**Source code:** [GitHub](https://github.com/parupati/IranUSAviationDisruptionRAG)

In this article, I'll walk through the architecture, the decisions I made, and what I learned along the way.

---

## The Problem

The [Global Civil Aviation Disruption 2026](https://www.kaggle.com/datasets/zkskhurram/global-civil-aviation-disruption2026-iranus-war) dataset on Kaggle contains 6 CSV files with 218 records covering airline financial losses, airport disruptions, airspace closures, flight cancellations, reroutes, and a timeline of conflict events.

Raw CSV data isn't exactly user-friendly. If you wanted to know "Which airline suffered the most?" or "What airports in Iran were closed?", you'd have to manually dig through spreadsheets. I wanted to make this data conversational — ask a question, get a clear answer with sources.

That's exactly what RAG does.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a pattern that combines two things:

1. **Retrieval** — Find the most relevant pieces of information from your data
2. **Generation** — Feed those pieces to an LLM to produce a human-readable answer

The key insight: instead of fine-tuning a model on your data (expensive, slow), you just give the LLM the right context at query time. The model doesn't need to "know" your data — it just needs to read it.

```
User Question
  → Embed the question into a vector
  → Search vector store for similar chunks
  → Retrieve top-k relevant chunks
  → Send chunks + question to LLM
  → LLM generates a grounded answer
```

---

## Architecture

Here's what I built:

```
CSV Files (6 tables, 218 records)
  → Python ingestion script converts each row to natural language
  → HuggingFace sentence-transformers embeds each chunk (all-MiniLM-L6-v2)
  → ChromaDB stores the vectors locally
  → FastAPI serves the /query endpoint
  → Angular frontend provides the chat UI
  → Deployed on Hugging Face Spaces (Docker)
```

### Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | LangChain | Mature RAG framework, pluggable components |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Fast, runs on CPU, no GPU needed |
| Vector Store | ChromaDB | Zero-config, file-based, perfect for small-medium datasets |
| LLM | OpenAI GPT-4o | Best answer quality for generation |
| API | FastAPI | Async, auto-generates Swagger docs, production-ready |
| Frontend | Angular | Integrated into my existing portfolio site |
| Deployment | Hugging Face Spaces (Docker) | Free tier, auto-scaling, git-based deploys |

---

## The Interesting Part: Structured Data + RAG

Most RAG tutorials use PDFs or text documents. My dataset was **structured CSV data** — rows and columns, not paragraphs. This required an extra step: converting each row into a natural language sentence before embedding.

For example, a row from `airline_losses_estimate.csv`:

```
Emirates, UAE, 4200000, 18, 62, 2835200, 9180
```

Becomes:

> "Emirates (UAE) faces an estimated daily financial loss of $4,200,000 USD due to the Iran-US conflict. 18 flights were cancelled and 62 were rerouted, incurring $2,835,200 in additional fuel costs. Approximately 9,180 passengers were impacted."

This is important because embedding models understand natural language, not CSV columns. Each of the 6 CSV files has its own conversion function that produces a descriptive sentence with all the context needed for retrieval.

---

## Building It: Step by Step

### 1. Ingestion

The ingestion script reads all 6 CSVs, converts each row to a natural language chunk, and stores it in ChromaDB with metadata (source file, category, original field values).

```python
# Each CSV file has a dedicated row-to-text converter
def row_to_text_airline_losses(row):
    return (
        f"{row['airline']} ({row['country']}) faces an estimated daily "
        f"financial loss of ${row['estimated_daily_loss_usd']:,.0f} USD..."
    )
```

218 documents across 6 categories — small enough to fit in a single ChromaDB collection, large enough to need proper retrieval.

### 2. Embedding

I used `all-MiniLM-L6-v2` from HuggingFace's sentence-transformers. It produces 384-dimensional vectors and runs comfortably on CPU. No GPU, no cloud embedding API, no cost.

```python
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
```

### 3. Retrieval + Generation

At query time, the user's question is embedded with the same model, and ChromaDB returns the top-k most similar chunks. These chunks are injected into a prompt template and sent to GPT-4o:

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

The prompt instructs the model to act as an aviation intelligence analyst and answer using ONLY the provided context — no hallucination.

### 4. API

FastAPI wraps the RAG pipeline into a clean REST endpoint:

```
POST /query
{
  "question": "Which airline had the highest financial loss?",
  "k": 5
}
```

Response includes the answer and the source documents used to generate it — full transparency.

### 5. Deployment

The entire system is containerized with Docker and deployed on Hugging Face Spaces (free tier). The vector store is built during the Docker build phase, so it's baked into the image — no cold-start database initialization.

---

## What I Learned

**1. Structured data needs extra love in RAG.** You can't just throw CSVs at an embedding model. Converting rows to natural language sentences dramatically improves retrieval quality.

**2. You don't need a GPU for embeddings.** `all-MiniLM-L6-v2` runs in milliseconds on CPU for small datasets. Don't over-engineer the infrastructure.

**3. ChromaDB is perfect for prototyping.** Zero config, runs embedded in your Python process, persists to disk. For 218 documents, it's instant.

**4. Hugging Face Spaces is underrated for API hosting.** Free Docker-based deployment with auto-generated URLs. The cold-start after inactivity (30-60 seconds) is the main trade-off.

**5. Context-stuffing beats RAG for small data.** I also built a portfolio chatbot endpoint on the same API — it just stuffs the entire markdown file into the system prompt. No embeddings, no vector store. When your data fits in the context window, keep it simple.

---

## Try It Yourself

**Live Demo:** [https://parupati.com/aviationRag](https://parupati.com/aviationRag)

**Example questions to try:**
- "Which airline suffered the highest daily financial loss?"
- "What airports in Iran were closed?"
- "How many flights were cancelled from Dubai on March 1st?"
- "What was the aviation impact of the Natanz airstrike?"
- "Which countries closed their airspace and for how long?"

**Source Code:** [https://github.com/parupati/IranUSAviationDisruptionRAG](https://github.com/parupati/IranUSAviationDisruptionRAG)

**API Docs:** [https://parupati-iran-us-aviation-rag.hf.space/docs](https://parupati-iran-us-aviation-rag.hf.space/docs)

---

## What's Next

- Adding **hybrid search** (vector + keyword) via Azure AI Search for better retrieval
- Exploring **streaming responses** for a more interactive chat experience
- Evaluating retrieval quality with metrics like precision@k and MRR

If you're building your first RAG system, start small — a few CSVs, a local vector store, and a cloud LLM. Get the pipeline working end-to-end, then optimize. The fundamentals transfer directly to production-scale systems.

---

*Built with Python, LangChain, ChromaDB, HuggingFace, OpenAI GPT-4o, FastAPI, Angular, and Hugging Face Spaces.*

*Connect with me on [LinkedIn](https://www.linkedin.com/in/parupati/) or check out more projects on [GitHub](https://github.com/parupati).*
