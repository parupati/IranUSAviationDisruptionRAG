import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.chdir(os.path.dirname(__file__))

from rag import get_vectorstore

vs = get_vectorstore()
results = vs.similarity_search("Which airlines had the highest financial losses?", k=3)
for i, doc in enumerate(results, 1):
    cat = doc.metadata.get("category", "?")
    print(f"[{i}] ({cat})")
    print(f"    {doc.page_content[:200]}")
    print()

results2 = vs.similarity_search("What airports were closed in Iran?", k=3)
print("---")
for i, doc in enumerate(results2, 1):
    cat = doc.metadata.get("category", "?")
    print(f"[{i}] ({cat})")
    print(f"    {doc.page_content[:200]}")
    print()
