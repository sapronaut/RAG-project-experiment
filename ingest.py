import httpx
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import re

# ── Settings ──────────────────────────────────────────────────────────────

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DB_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Functions ─────────────────────────────────────────────────────────────

def scrape_url(url: str) -> str:
    print(f"  Scraping: {url}")

    if "wikipedia.org/wiki/" in url:
        title = url.split("/wiki/")[-1]
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
        }
        headers = {"User-Agent": "RAGBot/1.0 (saptarshi@example.com)"}
        response = httpx.get(api_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()
        pages = data["query"]["pages"]
        text = next(iter(pages.values())).get("extract", "")
        print(f"  Got {len(text)} chars via MediaWiki API")
        return text

    # For non-Wikipedia URLs
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    response = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    print(f"  Got {len(text)} chars via scraping")
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def make_id(url: str, chunk_index: int) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
    return f"{url_hash}_chunk_{chunk_index}"


def ingest_urls(urls: list[str]):
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="web_pages")

    total_chunks = 0

    for url in urls:
        try:
            text = scrape_url(url)

            if len(text) < 100:
                print(f"  Very little text found at {url}. Skipping.")
                continue

            chunks = chunk_text(text)
            print(f"  Created {len(chunks)} chunks from {len(text)} characters")

            embeddings = embedder.encode(chunks).tolist()

            collection.upsert(
                ids=[make_id(url, i) for i in range(len(chunks))],
                documents=chunks,
                embeddings=embeddings,
                metadatas=[{"source": url} for _ in chunks],
            )

            total_chunks += len(chunks)
            print(f"  ✅ Saved {len(chunks)} chunks from {url}")

        except Exception as e:
            print(f"  ❌ Error processing {url}: {e}")

    print(f"\nDone! {total_chunks} chunks saved to '{DB_PATH}'")


# ── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    urls_to_index = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://en.wikipedia.org/wiki/Vector_database",
    ]

    print("Starting ingestion...\n")
    ingest_urls(urls_to_index)