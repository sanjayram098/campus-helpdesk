
"""CampusAI — Professional College Helpdesk RAG chatbot."""
from __future__ import annotations

import importlib
from typing import List

import streamlit as st

import modules.llm_service
importlib.reload(modules.llm_service)

from config import MAX_PDF_SIZE_MB
from modules.llm_service import check_connection, get_llm_service
from modules.pdf_processor import ProcessedDocument, process_pdf
from modules.rag_pipeline import RAGPipeline
from modules.vector_store import VectorStore


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CampusAI | College Helpdesk",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Premium dark UI
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #0b1120;
        --panel: #111c30;
        --blue: #60a5fa;
        --muted: #94a3b8;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, #172849 0%, transparent 42%),
            #0b1120;
        color: #e5edf8;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #0d1729;
        border-right: 1px solid #263650;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    h1, h2, h3 {
        color: #f1f5f9 !important;
        letter-spacing: -0.5px;
    }

    p, label, .stCaption {
        color: #cbd5e1;
    }

    .hero {
        background: linear-gradient(135deg, #172b4d, #111c30 70%);
        border: 1px solid #304767;
        border-radius: 22px;
        padding: 30px 32px;
        margin: 8px 0 25px 0;
        box-shadow: 0 12px 35px rgba(0,0,0,.18);
    }

    .hero-tag {
        color: #93c5fd;
        text-transform: uppercase;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: clamp(28px, 4vw, 42px);
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.15;
        margin-bottom: 12px;
    }

    .hero-desc {
        font-size: 15px;
        line-height: 1.7;
        color: #bdcbe0;
        max-width: 750px;
    }

    .section-label {
        color: #93c5fd;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #263650;
        border-radius: 16px;
        padding: 14px 18px;
        background: rgba(17, 28, 48, .72);
        margin-bottom: 12px;
    }

    [data-testid="stChatInput"] {
        border-color: #385274 !important;
        border-radius: 16px !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #f1f5f9 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #334968;
        font-weight: 600;
        transition: all .2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #60a5fa;
        color: #bfdbfe;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        border: none;
        color: white;
    }

    [data-testid="stFileUploader"] {
        background: #111c30;
        border: 1px dashed #385274;
        border-radius: 13px;
        padding: 8px;
    }

    [data-testid="stExpander"] {
        background: rgba(17, 28, 48, .6);
        border: 1px solid #263650;
        border-radius: 12px;
    }

    hr {
        border-color: #263650;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 28px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "messages": [],
        "pending_files": {},
        "indexed_docs": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ---------------------------------------------------------
# Cached services
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    return VectorStore()


@st.cache_resource(show_spinner=False)
def get_rag() -> RAGPipeline:
    return RAGPipeline(vector_store=get_vector_store())


def refresh_indexed_docs() -> None:
    st.session_state.indexed_docs = get_vector_store().list_documents()


def render_sources(sources: list) -> None:
    """Display retrieved document references."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})"):
        for source in sources:
            filename = source.get("filename", "Unknown document")
            page = source.get("page_number", "?")

            try:
                score = float(source.get("score", 0))
                score_text = f"{score:.2f}"
            except (TypeError, ValueError):
                score_text = "N/A"

            st.markdown(
                f"**{filename}** · Page {page} · "
                f"Relevance score: {score_text}"
            )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-tag">AI-Powered Student Support</div>
    <div class="hero-title">🎓 CampusAI Helpdesk</div>
    <div class="hero-desc">
        Your college information assistant. Upload handbooks and other
        college PDFs, then ask questions in natural language.
        Answers are generated using information retrieved from your documents.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 CampusAI")
    st.caption("COLLEGE KNOWLEDGE ASSISTANT")
    st.divider()

    llm = get_llm_service()
    status = llm.status()

    st.markdown(
        '<div class="section-label">System Status</div>',
        unsafe_allow_html=True,
    )

    if status.get("has_key"):
        st.success("Groq API key configured")
        st.caption(f"Model: {status.get('model', 'Configured model')}")
    else:
        st.warning("Groq API key not found. Check your .env file.")

    with st.expander("API Diagnostics"):
        st.write(
            "**API key configured:** "
            + ("Yes (hidden)" if status.get("has_key") else "No")
        )
        st.write(f"**Active model:** `{status.get('model', 'Unknown')}`")

        if st.button("Test API Connection", use_container_width=True):
            with st.spinner("Testing Groq connection..."):
                try:
                    ok, message = check_connection(llm)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
                except Exception as exc:
                    st.error(f"Connection test failed: {exc}")

    st.divider()

    st.markdown(
        '<div class="section-label">Knowledge Base</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload college PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum size: {MAX_PDF_SIZE_MB} MB per file.",
    )

    if uploaded:
        for file in uploaded:
            name = file.name

            if name in st.session_state.pending_files:
                continue

            data = file.getvalue()

            if len(data) > MAX_PDF_SIZE_MB * 1024 * 1024:
                st.error(f"{name} exceeds the file size limit.")
                continue

            st.session_state.pending_files[name] = data

    if st.session_state.pending_files:
        st.caption(
            f"{len(st.session_state.pending_files)} file(s) ready to process"
        )

        for name in st.session_state.pending_files:
            st.caption(f"📄 {name}")

        if st.button(
            "⚙️ Process & Index Documents",
            type="primary",
            use_container_width=True,
        ):
            vector_store = get_vector_store()
            failures = 0

            with st.spinner(
                "Extracting text and building the knowledge base..."
            ):
                for name, data in list(
                    st.session_state.pending_files.items()
                ):
                    try:
                        result: ProcessedDocument = process_pdf(
                            data, filename=name
                        )

                        if result.success and result.chunks:
                            vector_store.add_chunks(
                                result.chunks,
                                reindex=False,
                            )
                        elif result.success and result.is_scanned:
                            st.warning(
                                f"{name}: scanned PDF detected. "
                                "OCR may be required to extract its text."
                            )
                            failures += 1
                        else:
                            st.error(
                                f"{name}: "
                                f"{getattr(result, 'error', 'Processing failed')}"
                            )
                            failures += 1

                    except Exception as exc:
                        st.error(f"{name}: processing failed — {exc}")
                        failures += 1

            st.session_state.pending_files = {}
            refresh_indexed_docs()

            if failures:
                st.warning(
                    "Processing finished with some issues. "
                    "Check the messages above."
                )
            else:
                st.success("Documents processed successfully!")

            st.rerun()

    st.divider()

    refresh_indexed_docs()
    docs = st.session_state.indexed_docs

    st.markdown(
        '<div class="section-label">Indexed Documents</div>',
        unsafe_allow_html=True,
    )

    st.metric("Documents in knowledge base", len(docs))

    if not docs:
        st.info("No indexed documents yet. Upload a PDF to begin.")
    else:
        for doc in docs:
            with st.container():
                st.markdown(f"**📄 {doc['filename']}**")
                st.caption(
                    f"{doc['chunk_count']} searchable text chunks"
                )

                if st.button(
                    "Remove document",
                    key=f"del_{doc['doc_id']}",
                    use_container_width=True,
                ):
                    get_vector_store().delete_document(doc["doc_id"])
                    refresh_indexed_docs()
                    st.rerun()

    st.divider()

    # Download conversation
    if st.session_state.messages:
        chat_text = "\n\n".join(
            f"{'You' if message['role'] == 'user' else 'CampusAI'}:\n"
            f"{message['content']}"
            for message in st.session_state.messages
        )

        st.download_button(
            "📥 Download Chat History",
            data=chat_text,
            file_name="campusai_chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if st.button(
        "🧹 Clear Chat History",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "PDFs are processed by this application. Retrieved text may be "
        "sent to Groq to generate answers."
    )


# ---------------------------------------------------------
# Main chat area
# ---------------------------------------------------------
refresh_indexed_docs()

if not st.session_state.messages:
    st.markdown("### Welcome to CampusAI 👋")
    st.write(
        "Ask a question about your college documents. "
        "For best results, upload and index your PDFs first."
    )

    suggestions = [
        "Summarize the uploaded handbook",
        "What student guidelines are mentioned?",
        "What information is available in the documents?",
    ]

    columns = st.columns(len(suggestions))

    for index, question in enumerate(suggestions):
        with columns[index]:
            if st.button(
                question,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state["suggested_prompt"] = question
                st.rerun()


st.markdown(
    '<div class="section-label">CHAT ASSISTANT</div>',
    unsafe_allow_html=True,
)

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))


# Accept typed or suggested questions
prompt = st.chat_input("Ask something about your college documents...")

if not prompt and st.session_state.get("suggested_prompt"):
    prompt = st.session_state.pop("suggested_prompt")


# ---------------------------------------------------------
# Handle user question
# ---------------------------------------------------------
if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    if not get_vector_store().list_documents():
        answer = (
            "No documents have been indexed yet. Upload and process "
            "at least one PDF in the sidebar before asking questions."
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": [],
        })

        with st.chat_message("assistant"):
            st.warning(answer)

    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    result = get_rag().answer(prompt)
                    answer = result.answer
                    sources = result.sources or []

                    st.markdown(answer)
                    render_sources(sources)

                    if not sources and not result.evidence_found:
                        st.caption(
                            "No relevant sources were found in the "
                            "uploaded documents. Verify the answer "
                            "against your official college information."
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as exc:
                    error_message = (
                        "Sorry, I couldn't generate an answer this time. "
                        "Please try again or check the application terminal."
                    )

                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    })

                    st.caption(f"Technical details: {exc}")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
    CampusAI College Helpdesk · Retrieval-Augmented Generation · Built with Streamlit
</div>
""", unsafe_allow_html=True)
