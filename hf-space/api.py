import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

vectorstore = None
portfolio_context = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, portfolio_context
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

    md_path = os.path.join(os.path.dirname(__file__), "portfolio_info.md")
    with open(md_path, "r", encoding="utf-8") as f:
        portfolio_context = f.read()
    print(f"Portfolio context loaded: {len(portfolio_context)} chars")
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
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured as a Space secret")

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


class ChatMessage(BaseModel):
    role: str
    content: str


class PortfolioChatRequest(BaseModel):
    messages: list[ChatMessage]


class PortfolioChatResponse(BaseModel):
    reply: str


PORTFOLIO_SYSTEM_PROMPT = """You are Madhukar's AI assistant on his portfolio website. Answer questions about Madhukar using ONLY the information provided below. Be friendly, concise, and professional. If asked something not covered in the data, say you don't have that information and suggest contacting Madhukar via LinkedIn.

--- Madhukar's Portfolio Information ---
{context}
--- End of Information ---"""


@app.post("/portfolio-chat", response_model=PortfolioChatResponse)
def portfolio_chat(req: PortfolioChatRequest):
    if not req.messages or not req.messages[-1].content.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    from openai import OpenAI
    try:
        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": PORTFOLIO_SYSTEM_PROMPT.format(context=portfolio_context)},
        ]
        for msg in req.messages:
            messages.append({"role": msg.role, "content": msg.content})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        reply = response.choices[0].message.content
        return PortfolioChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=True)
