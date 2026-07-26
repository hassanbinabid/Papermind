"""
streamlit_app.py — PaperMind RAG frontend.
Run with: streamlit run streamlit_app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PaperMind RAG",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f7f9fc; }

    /* Chat message bubbles */
    .user-bubble {
        background: #2563eb;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 0;
        color: #ffffff;
        font-size: 15px;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-bubble {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        margin: 8px 0;
        color: #1a202c;
        font-size: 15px;
        max-width: 85%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Source citation chips */
    .source-chip {
        display: inline-block;
        background: #eaf2ff;
        border: 1px solid #93c5fd;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px 4px;
        font-size: 12px;
        color: #1d4ed8;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #1a202c !important;
        border: 1px solid #cbd5e0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* Status badges */
    .badge-ok {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-warn {
        background: #fef3c7;
        color: #b45309;
        border: 1px solid #fcd34d;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Divider */
    hr { border-color: #e2e8f0; }

    /* Make body text dark by default (covers sidebar markdown too) */
    body, .stMarkdown, [data-testid="stSidebar"] * {
        color: #1a202c;
    }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline (cached so it only loads once) ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the embedding model and verify ChromaDB is ready."""
    try:
        from app.embeddings import embed_query
        from app.vectorstore import collection_count
        from app.pipeline import run_rag_pipeline
        count = collection_count()
        return run_rag_pipeline, embed_query, count, None
    except Exception as e:
        return None, None, 0, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 PaperMind")
    st.markdown("---")

    # Load status
    with st.spinner("Loading model..."):
        run_rag_pipeline, embed_query, chunk_count, load_error = load_pipeline()

    if load_error:
        st.markdown(f'<span class="badge-warn">⚠ Error</span>', unsafe_allow_html=True)
        st.error(load_error)
    else:
        st.markdown('<span class="badge-ok">✓ Ready</span>', unsafe_allow_html=True)
        st.markdown(f"**📄 Chunks indexed:** {chunk_count:,}")

    st.markdown("---")

    # Clear chat
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">🧠 PaperMind</p>
    <p class="main-subtitle">Ask anything about your document — answers are grounded in the source.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🙋 {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="assistant-bubble">🧠 {msg["content"]}</div>',
            unsafe_allow_html=True
        )
        # Show sources if available
        if msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">📄 {s["source"]} · p.{s["page"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(
                f'<div style="margin: 4px 0 16px 0;">{chips}</div>',
                unsafe_allow_html=True
            )


# ── Input ─────────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your document...")

# Handle suggestion button clicks
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

# ── Process question ──────────────────────────────────────────────────────────
if question and question.strip():
    if load_error or run_rag_pipeline is None:
        st.error("Pipeline not loaded. Check the sidebar for errors.")
    elif chunk_count == 0:
        st.warning(
            "No documents indexed yet. "
            "Run `python main.py` first to ingest your PDF."
        )
    else:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": question.strip()
        })

        # Show user bubble immediately
        st.markdown(
            f'<div class="user-bubble">🙋 {question.strip()}</div>',
            unsafe_allow_html=True
        )

        # Generate answer
        with st.spinner("🔍 Searching document and generating answer..."):
            try:
                result = run_rag_pipeline(question.strip())
                answer  = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer  = f"Sorry, something went wrong: {str(e)}"
                sources = []

        # Show answer bubble
        st.markdown(
            f'<div class="assistant-bubble">🧠 {answer}</div>',
            unsafe_allow_html=True
        )

        # Show source chips
        if sources:
            chips = "".join(
                f'<span class="source-chip">📄 {s["source"]} · p.{s["page"]}</span>'
                for s in sources
            )
            st.markdown(
                f'<div style="margin: 4px 0 16px 0;">{chips}</div>',
                unsafe_allow_html=True
            )

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }