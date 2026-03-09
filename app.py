import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from rag import get_vectorstore, get_rag_chain

st.set_page_config(
    page_title="Aviation Disruption RAG",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Iran-US Aviation Disruption RAG")
st.caption("Ask questions about the 2026 Iran-US conflict's impact on global civil aviation")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    k_results = st.slider("Number of sources to retrieve", min_value=1, max_value=15, value=5)
    show_sources = st.checkbox("Show source documents", value=True)

    st.divider()
    st.subheader("Example Questions")
    examples = [
        "Which airline had the highest daily financial loss?",
        "What airports in Iran were closed?",
        "How many flights were cancelled from Dubai?",
        "What was the aviation impact of the Natanz airstrike?",
        "Which flights were rerouted and what was the extra cost?",
        "What countries closed their airspace?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["example_query"] = ex

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📄 Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}]** `{src['category']}` — {src['text']}")

prompt = st.chat_input("Ask a question about aviation disruptions...")

if "example_query" in st.session_state:
    prompt = st.session_state.pop("example_query")

if prompt:
    if not os.environ.get("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"] == "your-openai-api-key-here":
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                chain, retriever = get_rag_chain(k=k_results)
                answer = chain.invoke(prompt)
                source_docs = retriever.invoke(prompt)

                st.markdown(answer)

                sources_data = []
                if show_sources and source_docs:
                    with st.expander(f"📄 Sources ({len(source_docs)} chunks)"):
                        for i, doc in enumerate(source_docs, 1):
                            cat = doc.metadata.get("category", "unknown")
                            st.markdown(f"**[{i}]** `{cat}` — {doc.page_content}")
                            sources_data.append({"category": cat, "text": doc.page_content})

                msg_data = {"role": "assistant", "content": answer}
                if sources_data:
                    msg_data["sources"] = sources_data
                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.error(f"Error: {e}")
