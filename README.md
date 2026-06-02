<<<<<<< HEAD
# 🔍 RAG Chatbot — Web Page Q&A

Ask questions about any website. The bot reads the pages, remembers them, and answers using only what it found there.

---

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Normal ChatGPT knows things from its training data.  
RAG lets the AI answer from *your* documents instead.

The flow:
```
Your URLs → Read & Save → [Database]
                               ↓
Your Question → Search → Relevant Chunks → Claude → Answer
```

---

## Project Structure

```
rag_project/
├── ingest.py       # Step 1: Read URLs and save to database
├── retriever.py    # Step 2: Search the database (used by app.py)
├── app.py          # Step 3: Chat UI
└── requirements.txt
```

---

## Setup (one time)

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Get an Anthropic API key**  
Sign up at https://console.anthropic.com and copy your key.

---

## Usage

**Step 1 — Tell it what to read**

Open `ingest.py` and edit this part:
```python
urls_to_index = [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Large_language_model",
    # Add your own URLs here!
]
```

Then run:
```bash
python ingest.py
```

This creates a `chroma_db/` folder — your local database.  
You only need to do this again when you want to add new URLs.

**Step 2 — Start the chatbot**
```bash
streamlit run app.py
```

Your browser will open automatically. Paste your API key in the sidebar and start asking questions!

---

## Tips

- You can add more URLs anytime — just add them to `ingest.py` and run it again. Old data is preserved.
- The bot only knows what's in the indexed pages. If it says "I don't know", the answer isn't in your documents.
- Adjust `CHUNK_SIZE` in `ingest.py` if answers feel too fragmented (increase) or too broad (decrease).
=======
# RAG-project-experiment
>>>>>>> 732be177d75826532830ce4d702b7c71c99c40e6
