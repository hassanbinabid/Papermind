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
    .stApp { background-color: #0f1117; }

    /* Chat message bubbles */
    .user-bubble {
        background: #1e3a5f;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 0;
        color: #e8f4fd;
        font-size: 15px;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-bubble {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        margin: 8px 0;
        color: #e2e8f0;
        font-size: 15px;
        max-width: 85%;
    }

    /* Source citation chips */
    .source-chip {
        display: inline-block;
        background: #0d3349;
        border: 1px solid #1e6a9e;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px 4px;
        font-size: 12px;
        color: #7ec8e3;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #1a1f2e !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2533;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-subtitle {
        color: #718096;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* Status badges */
    .badge-ok {
        background: #1a3a2a;
        color: #68d391;
        border: 1px solid #2f855a;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-warn {
        background: #3a2a1a;
        color: #f6ad55;
        border: 1px solid #c05621;
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
    hr { border-color: #2d3748; }
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

    # Model info
    st.markdown("**🤖 Model**")
    st.markdown("`meta-llama/llama-3.3-70b-instruct:free`")
    st.markdown("via OpenRouter")

    st.markdown("**🔍 Embeddings**")
    st.markdown("`all-MiniLM-L6-v2`")
    st.markdown("Local · No API needed")

    st.markdown("**💾 Vector Store**")
    st.markdown("ChromaDB · Persistent")

    st.markdown("---")

    # Clear chat
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='color:#4a5568; font-size:11px; text-align:center;'>"
        "PaperMind RAG · Phase 1<br>"
        "AI Index Report 2026"
        "</div>",
        unsafe_allow_html=True
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">🧠 PaperMind</p>
    <p class="main-subtitle">Ask anything about your document — answers are grounded in the source.</p>
</div>
""", unsafe_allow_html=True)

# Suggested questions
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Try asking:")
    cols = st.columns(2)
    suggestions = [
        "What is the main contribution of this report?",
        "What are the key AI trends in 2026?",
        "How has AI investment changed over the years?",
        "What does the report say about AI safety?",
    ]
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, use_container_width=True, key=f"sug_{i}"):
                st.session_state.pending_question = suggestion
                st.rerun()

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
        })
