import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

vectorstore = None
embeddings = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, embeddings
    chroma_dir = os.path.join(os.path.dirname(__file__), "chroma_db")

    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        print("Vector store not found. Running ingestion...")
        from ingest import load_documents, build_vector_store
        docs = load_documents()
        build_vector_store(docs)
        print(f"Ingestion complete: {len(docs)} documents indexed.")

    from rag import get_vectorstore
    vectorstore = get_vectorstore()
    print(f"Vector store loaded: {vectorstore._collection.count()} documents")
    yield


app = FastAPI(
    title="Aviation Disruption RAG API",
    description="RAG API for querying the 2026 Iran-US conflict aviation disruption dataset",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class SourceDocument(BaseModel):
    category: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


@app.get("/health")
def health():
    doc_count = vectorstore._collection.count() if vectorstore else 0
    return {"status": "healthy", "documents_indexed": doc_count}


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "your-openai-api-key-here":
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    from rag import get_rag_chain
    try:
        chain, retriever = get_rag_chain(k=req.k)
        answer = chain.invoke(req.question)
        source_docs = retriever.invoke(req.question)

        sources = [
            SourceDocument(
                category=doc.metadata.get("category", "unknown"),
                content=doc.page_content,
            )
            for doc in source_docs
        ]
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=True)
