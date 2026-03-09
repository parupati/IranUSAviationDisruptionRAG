I built a RAG (Retrieval-Augmented Generation) system that turns structured aviation disruption data into a conversational AI — ask a question, get an accurate answer with sources.

The dataset covers the 2026 Iran-US conflict's impact on global civil aviation: 35 airlines, 35 airports, 25 airspace closures, 47 cancelled flights, and 45 rerouted flights.

What made this interesting:

- Most RAG systems work with PDFs or text. This one works with structured CSV data — each row is converted to a natural language sentence before embedding.
- Embeddings run locally on CPU using HuggingFace (no GPU needed).
- ChromaDB stores vectors on disk — zero config, zero cost.
- GPT-4o generates grounded answers from the retrieved chunks.
- The whole thing runs as a Docker container on Hugging Face Spaces (free tier).

Tech stack: Python, LangChain, ChromaDB, HuggingFace, OpenAI GPT-4o, FastAPI, Angular, Docker

Try it live: https://parupati.com/aviationRag

Example questions you can ask:
- "Which airline had the highest daily financial loss?"
- "What airports in Iran were closed?"
- "What was the aviation impact of the Natanz airstrike?"

Source code: https://github.com/parupati/IranUSAviationDisruptionRAG

Full article with architecture walkthrough and lessons learned in the comments.

#RAG #AI #LLM #MachineLearning #LangChain #OpenAI #Python #FastAPI #HuggingFace #FullStack #SoftwareEngineering #GenAI #RetrievalAugmentedGeneration
