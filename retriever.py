"""
retriever.py — Step 2 helper: Search the database for relevant text

What this file does in plain English:
  When you ask a question, this file:
  1. Converts your question into numbers (same embedding model as ingest.py)
  2. Searches the database for chunks whose numbers are closest to your question's numbers
  3. Returns the most relevant text chunks

Think of it like a smart Ctrl+F — instead of exact word matching,
it finds text that MEANS the same thing as your query.
"""

import chromadb
from sentence_transformers import SentenceTransformer

# Must match what was used in ingest.py
EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"

# How many chunks to retrieve per question.
# More = more context for Claude, but also more noise.
TOP_K = 5


class Retriever:
    def __init__(self):
        print("Loading retriever...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = client.get_collection(name="web_pages")

    def search(self, query: str) -> list[dict]:
        """
        Find the TOP_K most relevant chunks for a given question.

        Returns a list of dicts like:
          [{"text": "...", "source": "https://..."}, ...]
        """
        # Convert the question into numbers
        query_embedding = self.embedder.encode(query).tolist()

        # Ask Chroma: "find me the 5 chunks most similar to this embedding"
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )

        # Unpack the results into a clean list
        chunks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
            })

        return chunks
