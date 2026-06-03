# rag-chatbot

Ask questions about any webpage. The bot reads the pages, remembers them, and answers using only what it found there.

## tech stack

| layer | tool |
|---|---|
| scraping | httpx + BeautifulSoup |
| embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| vector db | ChromaDB (local, persisted to `./chroma_db`) |
| llm | Llama 3.1 8B via Groq API |
| ui | Streamlit |

## setup

```bash
pip install httpx beautifulsoup4 chromadb sentence-transformers streamlit groq
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) → API Keys → Create.

## usage

**1. Index your URLs**

Edit the list in `ingest.py`:
```python
urls_to_index = [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Vector_database",
]
```

Then run:
```bash
python ingest.py
```

**2. Start the app**
```bash
python -m streamlit run app.py
```

Paste your Groq API key in the sidebar and start asking questions.

## structure

```
├── ingest.py       # scrape → chunk → embed → store in ChromaDB
├── retriever.py    # embed query → search ChromaDB → return top-5 chunks
├── app.py          # streamlit chat UI → retriever + Groq LLM
├── requirements.txt
└── chroma_db/      # auto-created on first ingest, do not commit
```

## how it works

```
URLs → scrape → split into chunks → embed → ChromaDB
                                                ↓
question → embed → similarity search → top 5 chunks → Llama → answer
```

The LLM only sees the retrieved chunks, not the whole database — this keeps answers grounded and prevents hallucination.

## notes

- Re-run `ingest.py` anytime to add more URLs. Existing data is preserved.
- Wikipedia URLs use the MediaWiki API instead of scraping to avoid 403 blocks.
- Add `chroma_db/` and `*.pkl` to your `.gitignore` before pushing.
