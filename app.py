"""
NIT Warangal RAG Chatbot — Professional Streamlit frontend
White + blue minimalist theme · Claude-style shimmer loading · Streaming answers
"""

import sys
import os
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Pipeline"))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from rag_chain import (
    load_vectorstore, create_rag_chain,
    make_embeddings, make_llm,
    next_embed_key, next_llm_key,
    TOP_K, llm_keys, embed_keys,
    LLM_PROVIDER, LLM_MODEL_GOOGLE, LLM_MODEL_GROQ,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NITW Academic Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — white/blue theme + shimmer animation + polished chrome
# ─────────────────────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
/* ── Global reset ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

/* hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── App shell ────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #FFFFFF;
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #F8FAFC !important;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Main content padding ─────────────────────────────────── */
[data-testid="stMain"] .block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;          /* room for pinned input */
    max-width: 860px;
    margin: 0 auto;
}

/* ── Chat messages ────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.25rem 0 !important;
}

/* assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"]) {
    background: transparent !important;
}

/* ── Shimmer keyframe ─────────────────────────────────────── */
@keyframes shimmer {
    0%   { background-position: -800px 0; }
    100% { background-position:  800px 0; }
}

.shimmer-line {
    height: 14px;
    border-radius: 7px;
    background: linear-gradient(
        90deg,
        #EFF6FF 0%,
        #BFDBFE 30%,
        #93C5FD 50%,
        #BFDBFE 70%,
        #EFF6FF 100%
    );
    background-size: 800px 100%;
    animation: shimmer 1.6s ease-in-out infinite;
    margin-bottom: 10px;
}
.shimmer-line.w80 { width: 80%; }
.shimmer-line.w60 { width: 60%; }
.shimmer-line.w72 { width: 72%; }

.shimmer-label {
    font-size: 12px;
    color: #94A3B8;
    margin-bottom: 14px;
    letter-spacing: 0.03em;
}

/* ── Suggested prompt cards ───────────────────────────────── */
.prompt-card button {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    color: #334155 !important;
    font-size: 14px !important;
    text-align: left !important;
    padding: 12px 16px !important;
    transition: border-color 0.15s, background 0.15s !important;
}
.prompt-card button:hover {
    border-color: #2563EB !important;
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
}

/* ── Source expander ──────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    background: #F8FAFC !important;
    margin-top: 6px;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important;
    color: #64748B !important;
    font-weight: 500;
}
[data-testid="stExpander"] summary:hover {
    color: #2563EB !important;
}

/* ── Chat input ───────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1.5px solid #E2E8F0 !important;
    background: #FFFFFF !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* ── Sidebar history item ─────────────────────────────────── */
.hist-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 7px 10px;
    border-radius: 8px;
    margin-bottom: 2px;
    font-size: 13px;
    color: #475569;
    cursor: default;
    line-height: 1.4;
    border: 1px solid transparent;
    transition: background 0.12s;
}
.hist-item:hover {
    background: #EFF6FF;
    border-color: #DBEAFE;
    color: #1D4ED8;
}
.hist-dot {
    flex-shrink: 0;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #CBD5E1;
    margin-top: 5px;
}

/* ── Source card inside expander ──────────────────────────── */
.src-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.src-title {
    font-size: 12px;
    font-weight: 600;
    color: #2563EB;
    margin-bottom: 4px;
    letter-spacing: 0.01em;
}
.src-excerpt {
    font-size: 12px;
    color: #64748B;
    line-height: 1.5;
    border-left: 3px solid #BFDBFE;
    padding-left: 8px;
    margin: 0;
}

/* ── Divider override ─────────────────────────────────────── */
hr { border-color: #E2E8F0 !important; margin: 0.75rem 0 !important; }

/* ── New chat button ──────────────────────────────────────── */
[data-testid="stSidebar"] button[kind="secondary"] {
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F0 !important;
    color: #334155 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    border-color: #2563EB !important;
    color: #2563EB !important;
    background: #EFF6FF !important;
}

/* ── Status/spinner override ──────────────────────────────── */
[data-testid="stStatusWidget"] {
    display: none !important;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shimmer HTML (shown while retrieving)
# ─────────────────────────────────────────────────────────────────────────────
SHIMMER_HTML = """
<div style="padding: 6px 0 4px 0;">
  <div class="shimmer-label">Searching knowledge base…</div>
  <div class="shimmer-line w80"></div>
  <div class="shimmer-line w60"></div>
  <div class="shimmer-line w72"></div>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
# RAG helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def get_rag_components():
    vs = load_vectorstore()
    prompt, vs = create_rag_chain(vs)
    return prompt, vs


def retrieve_docs(question: str, vectorstore):
    for _ in range(len(embed_keys) * 3):
        key = next_embed_key()
        try:
            vec = make_embeddings(key).embed_query(question)
            return vectorstore.similarity_search_by_vector(vec, k=TOP_K)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            raise
    return []


def stream_answer(prompt, context: str, question: str):
    """Generator yielding string tokens — rotates LLM keys on 429."""
    for attempt in range(len(llm_keys) * 3):
        key = next_llm_key()
        chain = prompt | make_llm(key) | StrOutputParser()
        try:
            yield from chain.stream({"context": context, "question": question})
            return
        except Exception as e:
            if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < len(llm_keys) * 3 - 1:
                continue
            raise


def format_sources(docs) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "excerpt": doc.page_content[:280].replace("\n", " ").strip(),
        }
        for doc in docs
    ]


def source_html(sources: list[dict]) -> str:
    cards = ""
    for i, s in enumerate(sources, 1):
        name = s["source"].replace("\\", "/").split("/")[-1]
        cards += f"""
        <div class="src-card">
          <div class="src-title">{i}. {name}</div>
          <p class="src-excerpt">{s['excerpt']}…</p>
        </div>"""
    return cards


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ─────────────────────────────────────────────────────────────────────────────
# Load RAG once
# ─────────────────────────────────────────────────────────────────────────────
try:
    rag_prompt, vectorstore = get_rag_components()
except Exception as exc:
    st.error(f"Failed to load knowledge base: {exc}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + title
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="width:36px; height:36px; border-radius:10px;
                        background:linear-gradient(135deg,#2563EB,#60A5FA);
                        display:flex; align-items:center; justify-content:center;
                        font-size:18px; flex-shrink:0;">🎓</div>
            <div>
                <div style="font-weight:700; font-size:15px; color:#0F172A; line-height:1.2;">NITW Assistant</div>
                <div style="font-size:11px; color:#94A3B8;">Academic Knowledge Base</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.latest_sources = []
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Conversation history ───────────────────────────────────
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    if user_msgs:
        st.markdown(
            "<div style='font-size:11px;font-weight:600;color:#94A3B8;"
            "letter-spacing:0.07em;text-transform:uppercase;margin-bottom:6px;'>"
            "This session</div>",
            unsafe_allow_html=True,
        )
        for msg in user_msgs[-10:]:            # show latest 10
            snippet = msg["content"][:55] + ("…" if len(msg["content"]) > 55 else "")
            st.markdown(
                f'<div class="hist-item"><div class="hist-dot"></div><span>{snippet}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='font-size:13px;color:#CBD5E1;padding:8px 4px;'>"
            "No questions yet.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Latest sources panel ───────────────────────────────────
    if st.session_state.latest_sources:
        st.markdown(
            "<div style='font-size:11px;font-weight:600;color:#94A3B8;"
            "letter-spacing:0.07em;text-transform:uppercase;margin-bottom:8px;'>"
            "Last cited sources</div>",
            unsafe_allow_html=True,
        )
        for i, s in enumerate(st.session_state.latest_sources, 1):
            name = s["source"].replace("\\", "/").split("/")[-1]
            st.markdown(
                f"""<div style='font-size:12px;color:#475569;padding:5px 8px;
                    border-left:3px solid #BFDBFE;margin-bottom:5px;
                    background:#F8FAFC;border-radius:0 6px 6px 0;'>
                    <b style='color:#2563EB;'>{i}.</b> {name}</div>""",
                unsafe_allow_html=True,
            )

    # ── Model info ─────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    provider = LLM_PROVIDER.upper()
    model_name = LLM_MODEL_GROQ if LLM_PROVIDER == "groq" else LLM_MODEL_GOOGLE
    st.markdown(
        f"""
        <div style="font-size:11px; color:#94A3B8; line-height:1.8;">
            <div><b style="color:#CBD5E1;">LLM</b> &nbsp;{model_name}</div>
            <div><b style="color:#CBD5E1;">Provider</b> &nbsp;{provider}</div>
            <div><b style="color:#CBD5E1;">Embeddings</b> &nbsp;gemini-embedding-001</div>
            <div><b style="color:#CBD5E1;">Top-K chunks</b> &nbsp;{TOP_K}</div>
            <div><b style="color:#CBD5E1;">API keys</b> &nbsp;{len(embed_keys)} embed · {len(llm_keys)} LLM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main: welcome screen or chat history
# ─────────────────────────────────────────────────────────────────────────────
SUGGESTED = [
    "What is the minimum attendance required for end semester exams?",
    "How is CGPA calculated and what is the minimum to pass?",
    "Who is eligible for the makeup examination?",
    "What is the minimum CGPA required for the Minor program?",
]

if not st.session_state.messages:
    # ── Welcome screen ─────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 0 2rem 0;">
            <div style="display:inline-flex; align-items:center; justify-content:center;
                        width:60px; height:60px; border-radius:16px;
                        background:linear-gradient(135deg,#2563EB,#60A5FA);
                        font-size:28px; margin-bottom:20px;">🎓</div>
            <h1 style="font-size:28px; font-weight:700; color:#0F172A;
                       letter-spacing:-0.02em; margin:0 0 8px 0;">
                NIT Warangal Academic Assistant
            </h1>
            <p style="font-size:15px; color:#64748B; max-width:480px; margin:0 auto;">
                Ask questions about regulations, syllabus, policies, and circulars
                grounded in official NITW documents.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:12px; font-weight:600; color:#94A3B8; "
        "letter-spacing:0.07em; text-transform:uppercase; "
        "text-align:center; margin-bottom:12px;'>Try asking</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    for i, s in enumerate(SUGGESTED):
        with (col1 if i % 2 == 0 else col2):
            st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_prompt = s
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

else:
    # ── Render conversation history ────────────────────────────
    for msg in st.session_state.messages:
        avatar = "🎓" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 Sources — {len(msg['sources'])} passages"):
                    st.markdown(source_html(msg["sources"]), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chat input + answer pipeline
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about NITW regulations, policies, syllabus…")

# Pick up either the text-box input or a clicked suggested prompt
question = user_input or st.session_state.pop("pending_prompt", None)

if question:
    # ── Append & show user message ─────────────────────────────
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # ── Assistant turn ─────────────────────────────────────────
    with st.chat_message("assistant", avatar="🎓"):

        # 1. Shimmer while retrieving
        shimmer_ph = st.empty()
        shimmer_ph.markdown(SHIMMER_HTML, unsafe_allow_html=True)

        try:
            docs = retrieve_docs(question, vectorstore)
        except Exception as exc:
            shimmer_ph.empty()
            st.error(f"Retrieval error: {exc}")
            st.stop()

        shimmer_ph.empty()

        # 2. Stream answer
        context = "\n\n".join(
            f"--- Document {i} (Source: {doc.metadata.get('source', 'Unknown')}) ---\n{doc.page_content}"
            for i, doc in enumerate(docs, 1)
        )

        try:
            full_answer = st.write_stream(stream_answer(rag_prompt, context, question))
        except Exception as exc:
            st.error(f"Generation error: {exc}")
            st.stop()

        # 3. Sources expander below answer
        sources_data = format_sources(docs)
        if sources_data:
            with st.expander(f"📎 Sources — {len(sources_data)} passages"):
                st.markdown(source_html(sources_data), unsafe_allow_html=True)

    # ── Persist ────────────────────────────────────────────────
    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer, "sources": sources_data}
    )
    st.session_state.latest_sources = sources_data
