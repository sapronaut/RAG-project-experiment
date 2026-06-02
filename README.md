# rag-chatbot

A minimal RAG pipeline that lets you ask questions about any webpage. Built with Python, Chroma, and Claude.

## stack

- **scraping** — httpx + BeautifulSoup
- **embeddings** — sentence-transformers (`all-MiniLM-L6-v2`, runs locally)
- **vector db** — Chroma (persisted to `./chroma_db`)
- **llm** — Claude via Anthropic API
- **ui** — Streamlit

## setup

```bash
pip install -r requirements.txt
```

Get an API key at [console.anthropic.com](https://console.anthropic.com).

## usage

**1. Index your URLs**

Edit the list in `ingest.py`:
```python
urls_to_index = [
    "https://example.com/page",
]
```

Then run:
```bash
python ingest.py
```

**2. Start the app**
```bash
streamlit run app.py
```

Paste your API key in the sidebar and ask away.

## structure

```
├── ingest.py       # scrape → chunk → embed → store
├── retriever.py    # query → search → return top-k chunks
├── app.py          # streamlit chat UI
└── chroma_db/      # auto-created on first ingest
```

## notes

- Re-run `ingest.py` anytime to add more URLs. Existing data is preserved.
- The bot answers strictly from indexed content — it won't hallucinate beyond it.
- Tune `CHUNK_SIZE` in `ingest.py` if answers feel off (default: 500 chars).
