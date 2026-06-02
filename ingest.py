"""
ingest.py — Step 1: Read websites and save their content to the database

What this file does in plain English:
  1. Visit each URL you give it
  2. Extract just the readable text (no HTML junk)
  3. Split the text into small chunks (like cutting a book into paragraphs)
  4. Convert each chunk into a list of numbers (called an "embedding") that
     captures the meaning of the text
  5. Save everything into a local database (Chroma) so we can search it later

Run this file first before using the chatbot.
"""

import httpx
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import re

# ── Settings ──────────────────────────────────────────────────────────────────

# How many characters per chunk. 500 is a good balance:
# too small = loses context, too large = too noisy for search
CHUNK_SIZE = 500

# How many characters chunks overlap. Overlap prevents losing meaning
# at the boundary between two chunks (like a sentence cut in half)
CHUNK_OVERLAP = 50

# Where the database will be saved on your computer
DB_PATH = "./chroma_db"

# The embedding model — this runs locally, no API key needed.
# It converts text → numbers so we can do similarity search.
EMBED_MODEL = "all-MiniLM-L6-v2"  # Small, fast, good quality

# ── Functions ─────────────────────────────────────────────────────────────────

def scrape_url(url: str) -> str:
    """
    Visit a URL and return just the plain text (no HTML tags).
    Think of this as copy-pasting the readable content of a webpage.
    """
    print(f"  Scraping: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (RAG Bot)"}
    
    response = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
    response.raise_for_status()  # Crash loudly if the URL is broken
    
    # BeautifulSoup parses the raw HTML
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove navigation menus, scripts, and styling — we only want readable text
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    
    # Get all text and collapse whitespace into single spaces
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into overlapping chunks.

    Example with chunk_size=10, overlap=3:
      "The quick brown fox jumps"
      → ["The quick b", "ck brown fo", "own fox jum", ...]

    Overlap ensures we don't lose a sentence that falls on a boundary.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # Move forward, but back up by overlap amount
    return chunks


def make_id(url: str, chunk_index: int) -> str:
    """
    Create a unique ID for each chunk so we can store and find it later.
    Example: "a3f9b2_chunk_0", "a3f9b2_chunk_1", ...
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
    return f"{url_hash}_chunk_{chunk_index}"


def ingest_urls(urls: list[str]):
    """
    Main function: scrape all URLs, chunk the text, embed it, and save to DB.
    """
    print("Loading embedding model (first run may download ~90MB)...")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Connect to (or create) the local Chroma database
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # A "collection" is like a table in a regular database
    collection = client.get_or_create_collection(name="web_pages")

    total_chunks = 0

    for url in urls:
        try:
            # Step 1: Get the text
            text = scrape_url(url)
            
            if len(text) < 100:
                print(f"  ⚠️  Very little text found at {url}, skipping.")
                continue

            # Step 2: Split into chunks
            chunks = chunk_text(text)
            print(f"  → {len(chunks)} chunks created from {len(text)} characters")

            # Step 3: Convert chunks to embeddings (numbers)
            # This is the "magic" — similar meanings get similar numbers
            embeddings = embedder.encode(chunks).tolist()

            # Step 4: Save to Chroma
            collection.upsert(  # upsert = insert or update if already exists
                ids=[make_id(url, i) for i in range(len(chunks))],
                documents=chunks,
                embeddings=embeddings,
                metadatas=[{"source": url} for _ in chunks],  # remember which URL each chunk came from
            )

            total_chunks += len(chunks)
            print(f"  ✅ Saved {len(chunks)} chunks from {url}")

        except Exception as e:
            print(f"  ❌ Error processing {url}: {e}")

    print(f"\n✅ Done! {total_chunks} total chunks saved to '{DB_PATH}'")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ✏️  EDIT THIS LIST with the URLs you want to index
    urls_to_index = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://en.wikipedia.org/wiki/Large_language_model",
    ]

    print("🔍 Starting ingestion...\n")
    ingest_urls(urls_to_index)
