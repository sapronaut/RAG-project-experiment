import streamlit as st
from groq import Groq
from retriever import Retriever

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Chatbot", page_icon="🔍", layout="centered")
st.title("🔍 RAG Chatbot")
st.caption("Ask questions about the websites you've indexed.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get yours at https://console.groq.com",
        placeholder="gsk-...",
    )

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. Run `ingest.py` to load URLs
    2. Ask questions here
    3. The bot finds relevant text and passes it to Llama
    4. Llama answers using **only** what's in your documents
    """)

    st.divider()
    st.markdown("**Want to add more URLs?**")
    st.markdown("Edit the `urls_to_index` list in `ingest.py` and run it again.")

# ── Load retriever ────────────────────────────────────────────────────────────

@st.cache_resource
def load_retriever():
    try:
        return Retriever()
    except Exception:
        return None

retriever = load_retriever()

if retriever is None:
    st.error("❌ No indexed documents found. Please run `ingest.py` first.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Main chat logic ───────────────────────────────────────────────────────────

if user_question := st.chat_input("Ask a question about your documents..."):

    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Searching documents..."):
        chunks = retriever.search(user_question)

    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    )

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

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                client = Groq(api_key=api_key)

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_question},
                    ]
                )

                answer = response.choices[0].message.content
                st.markdown(answer)

                with st.expander("📄 Retrieved context chunks"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}** — `{chunk['source']}`")
                        st.text(chunk["text"][:300] + "...")
                        st.divider()

            except Exception as e:
                answer = f"❌ Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})