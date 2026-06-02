"""
app.py — Step 3: The chat interface

What this file does in plain English:
  - Shows a browser-based chat UI (powered by Streamlit)
  - When you type a question:
      1. Searches your indexed documents for relevant chunks (via retriever.py)
      2. Sends those chunks + your question to Claude
      3. Claude reads ONLY those chunks and answers based on them
      4. The answer appears in the chat, with sources listed

Run with:
  streamlit run app.py
"""

import streamlit as st
import anthropic
import os
from retriever import Retriever

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Chatbot", page_icon="🔍", layout="centered")
st.title("🔍 RAG Chatbot")
st.caption("Ask questions about the websites you've indexed.")

# ── Sidebar: API key + info ────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Get yours at https://console.anthropic.com",
        placeholder="sk-ant-...",
    )
    
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. Run `ingest.py` to load URLs
    2. Ask questions here
    3. The bot finds relevant text and passes it to Claude
    4. Claude answers using **only** what's in your documents
    """)
    
    st.divider()
    st.markdown("**Want to add more URLs?**")
    st.markdown("Edit the `urls_to_index` list in `ingest.py` and run it again.")

# ── Load retriever (cached so it only loads once) ─────────────────────────────

@st.cache_resource
def load_retriever():
    """
    @st.cache_resource means Streamlit loads this only once,
    not on every message — important since loading the model takes a few seconds.
    """
    try:
        return Retriever()
    except Exception as e:
        return None

retriever = load_retriever()

if retriever is None:
    st.error("❌ No indexed documents found. Please run `ingest.py` first.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────

# st.session_state persists data between reruns (Streamlit reruns on every interaction)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Main chat logic ───────────────────────────────────────────────────────────

if user_question := st.chat_input("Ask a question about your documents..."):
    
    # Guard: need API key to call Claude
    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # ── Retrieve relevant chunks ──────────────────────────────────────────────
    
    with st.spinner("Searching documents..."):
        chunks = retriever.search(user_question)

    # Format the retrieved chunks into a single block of context text
    # This is what gets sent to Claude alongside the question
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    )

    # ── Build the prompt for Claude ───────────────────────────────────────────
    
    # We use a "system prompt" to give Claude its instructions.
    # Crucially: we tell it to ONLY use the provided context.
    # This prevents hallucination — Claude won't make things up.
    system_prompt = """You are a helpful research assistant.
You will be given CONTEXT extracted from web pages, and a QUESTION from the user.

Rules:
- Answer using ONLY the information in the context below.
- If the context doesn't contain enough information to answer, say so honestly.
- Be concise and clear.
- At the end of your answer, list the sources you used.

CONTEXT:
{context}
""".format(context=context_block)

    # ── Call Claude ───────────────────────────────────────────────────────────
    
    with st.chat_message("assistant"):
        with st.spinner("Claude is thinking..."):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[
                        # We only send the current question, not full history.
                        # For a production app you'd include chat history too.
                        {"role": "user", "content": user_question}
                    ],
                )
                
                answer = response.content[0].text
                st.markdown(answer)
                
                # Show which chunks were used (expandable, so it's not cluttered)
                with st.expander("📄 Retrieved context chunks"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}** — `{chunk['source']}`")
                        st.text(chunk["text"][:300] + "...")
                        st.divider()

            except anthropic.AuthenticationError:
                answer = "❌ Invalid API key. Please check your key in the sidebar."
                st.error(answer)
            except Exception as e:
                answer = f"❌ Error: {e}"
                st.error(answer)

    # Save assistant's reply to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
