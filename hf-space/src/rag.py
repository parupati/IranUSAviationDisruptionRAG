import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

PROMPT_TEMPLATE = """You are an aviation intelligence analyst specializing in the 2026 Iran-US conflict's impact on global civil aviation. Use ONLY the provided context to answer the question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Provide a clear, detailed answer based on the data:"""


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(k=5):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "lambda_mult": 0.7})
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def query(question, k=5):
    chain, retriever = get_rag_chain(k=k)
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    return answer, source_docs


if __name__ == "__main__":
    import sys

    if not os.environ.get("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"] == "your-openai-api-key-here":
        print("ERROR: Set your OPENAI_API_KEY in the .env file first.")
        sys.exit(1)

    print("Aviation Disruption RAG System")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\nSearching and generating answer...\n")
        answer, sources = query(question)
        print(f"Answer:\n{answer}\n")
        print(f"--- Sources ({len(sources)} chunks retrieved) ---")
        for i, doc in enumerate(sources, 1):
            cat = doc.metadata.get("category", "?")
            print(f"  [{i}] ({cat}) {doc.page_content[:120]}...")
        print()
