"""
Talk to Documents — RAG Assistant
Upload PDFs or text files and ask questions with source citations and conversation memory.
Supports OpenAI or Google Gemini (set LLM_PROVIDER=gemini and GEMINI_API_KEY in .env).
"""

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader

# Load .env from the same directory as this script (so it's found regardless of cwd)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").lower()
if GEMINI_API_KEY and not OPENAI_API_KEY:
    LLM_PROVIDER = "gemini"

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "doc_chunks"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

# Model names (override in .env when providers add or change models)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")

if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        st.error("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env to use Gemini.")
        st.stop()
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
else:
    if not OPENAI_API_KEY:
        st.error("Set OPENAI_API_KEY (or OPEN_API_KEY) in .env to use OpenAI.")
        st.stop()
    import openai
    openai.api_key = OPENAI_API_KEY


class GeminiEmbeddingFunction:
    """Chroma-compatible embedding function using Gemini API."""

    def __init__(self, api_key: str, model_name: str = GEMINI_EMBEDDING_MODEL):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = model_name

    def _embed(self, texts: list, task_type: str) -> list:
        if not texts:
            return []
        all_embeddings = []
        for text in texts:
            result = self._genai.embed_content(
                model=self._model,
                content=text,
                task_type=task_type,
            )
            emb = result.get("embedding") or getattr(result, "embedding", None)
            if emb is None:
                emb = (result.get("embeddings") or getattr(result, "embeddings", []))[0]
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                all_embeddings.append(emb)
            else:
                all_embeddings.append(list(emb) if hasattr(emb, "__iter__") else [])
        return all_embeddings

    def __call__(self, input: list) -> list:
        """Embed documents (for indexing)."""
        return self._embed(input, task_type="retrieval_document")

    def embed_query(self, input):  # Chroma calls this when querying
        """Embed query text(s). Input may be a string or a list of strings."""
        if isinstance(input, str):
            input = [input]
        return self._embed(input, task_type="retrieval_query")


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if not text or not text.strip():
        return []
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def parse_pdf(file) -> list[dict]:
    """Extract text from PDF; returns list of {content, page_number}."""
    reader = PdfReader(file)
    result = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            result.append({"content": text.strip(), "page_number": str(i + 1)})
    return result


def parse_txt(file) -> list[dict]:
    """Parse plain text file; single 'page' 1."""
    text = file.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if not text.strip():
        return []
    return [{"content": text.strip(), "page_number": "1"}]


def load_document(uploaded_file) -> list[dict]:
    """Dispatch to PDF or TXT parser."""
    name = (uploaded_file.name or "").lower()
    if name.endswith(".pdf"):
        return parse_pdf(uploaded_file)
    if name.endswith(".txt"):
        uploaded_file.seek(0)
        return parse_txt(uploaded_file)
    return []


def doc_to_chunks(doc_blocks: list[dict]) -> list[tuple[str, dict]]:
    """Turn document blocks into chunks with metadata for vector store."""
    chunks_with_meta = []
    for block in doc_blocks:
        content = block.get("content", "")
        page = block.get("page_number", "")
        for c in chunk_text(content):
            chunks_with_meta.append((c, {"page_number": page}))
    return chunks_with_meta


# ---------------------------------------------------------------------------
# Vector store (Chroma)
# ---------------------------------------------------------------------------

def get_chroma_client():
    """Persistent Chroma client."""
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))


def get_embedding_function():
    if LLM_PROVIDER == "gemini":
        return GeminiEmbeddingFunction(api_key=GEMINI_API_KEY, model_name=GEMINI_EMBEDDING_MODEL)
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name=OPENAI_EMBEDDING_MODEL
    )


def build_index(uploaded_file):
    """Parse file, chunk, embed, and store in Chroma. Returns (collection, doc_name)."""
    doc_blocks = load_document(uploaded_file)
    if not doc_blocks:
        return None, None
    chunks_with_meta = doc_to_chunks(doc_blocks)
    if not chunks_with_meta:
        return None, None

    client = get_chroma_client()
    ef = get_embedding_function()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef, metadata={"description": "rag"})

    ids = []
    documents = []
    metadatas = []
    for i, (text, meta) in enumerate(chunks_with_meta):
        ids.append(str(i))
        documents.append(text)
        metadatas.append(meta)
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection, uploaded_file.name


def query_index(collection, query: str, n: int = TOP_K):
    """Return top-k chunks with metadata."""
    res = collection.query(query_texts=[query], n_results=n, include=["documents", "metadatas"])
    docs = (res["documents"] or [[]])[0]
    metas = (res["metadatas"] or [[]])[0]
    return list(zip(docs, metas))


# ---------------------------------------------------------------------------
# LLM with conversation memory and citations
# ---------------------------------------------------------------------------

def build_rag_context_and_messages(query: str, retrieved: list[tuple], history: list[dict]):
    """Build context string and message list for either OpenAI or Gemini."""
    context_parts = []
    for i, (text, meta) in enumerate(retrieved, 1):
        page = meta.get("page_number", "?")
        context_parts.append(f"[Source {i} (Page {page})]\n{text}")
    context = "\n\n---\n\n".join(context_parts)
    system = (
        "You are a helpful assistant that answers questions based only on the provided document excerpts. "
        "Use a clear, neutral tone. If the answer is not in the excerpts, say so. "
        "When you use information from a source, cite it inline like: [Source 1] or [Source 2, Page 3]. "
        "Do not invent page numbers; use only the Source/Page labels from the excerpts."
    )
    full_system = f"{system}\n\nDocument excerpts:\n\n{context}"
    max_turns = 6
    history_msgs = history[-(max_turns * 2) :]
    return full_system, history_msgs


def chat_with_rag_openai(collection, query: str, history: list[dict]) -> str:
    retrieved = query_index(collection, query)
    if not retrieved:
        return "No relevant passages found in the document. Try rephrasing or upload a document first."
    full_system, history_msgs = build_rag_context_and_messages(query, retrieved, history)
    messages = [{"role": "system", "content": full_system}]
    for msg in history_msgs:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})
    import openai
    resp = openai.chat.completions.create(
        model=OPENAI_CHAT_MODEL, messages=messages, temperature=0
    )
    return (resp.choices[0].message.content or "").strip()


def chat_with_rag_gemini(collection, query: str, history: list[dict]) -> str:
    retrieved = query_index(collection, query)
    if not retrieved:
        return "No relevant passages found in the document. Try rephrasing or upload a document first."
    full_system, history_msgs = build_rag_context_and_messages(query, retrieved, history)
    prompt_parts = [full_system]
    for msg in history_msgs:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt_parts.append(f"{role}: {msg['content']}")
    prompt_parts.append(f"User: {query}")
    prompt_parts.append("Assistant:")
    prompt_text = "\n\n".join(prompt_parts)
    model = genai.GenerativeModel(GEMINI_CHAT_MODEL)
    response = model.generate_content(prompt_text, generation_config={"temperature": 0})
    if not response or not response.text:
        return "Sorry, I couldn't generate a response."
    return response.text.strip()


def chat_with_rag(collection, query: str, history: list[dict]) -> str:
    if LLM_PROVIDER == "gemini":
        return chat_with_rag_gemini(collection, query, history)
    return chat_with_rag_openai(collection, query, history)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Talk to Documents", page_icon="📄", layout="centered")
st.title("📄 Talk to Documents")
st.caption("Upload a PDF or text file, then ask questions. Answers include source citations.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# File upload
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"], key="uploader")

if uploaded_file is not None:
    # Rebuild index only when file changes (compare by name + size)
    file_key = (uploaded_file.name, uploaded_file.size)
    if getattr(st.session_state, "last_file_key", None) != file_key:
        with st.spinner("Indexing document..."):
            col, doc_name = build_index(uploaded_file)
            if col is None:
                st.warning("No text could be extracted from this file.")
            else:
                st.session_state.collection = col
                st.session_state.doc_name = doc_name
                st.session_state.last_file_key = file_key
                st.session_state.messages = []
        st.rerun()

if st.session_state.doc_name:
    st.success(f"Document loaded: **{st.session_state.doc_name}** — you can ask questions below.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input and reply
if prompt := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.collection is None:
        with st.chat_message("assistant"):
            st.markdown("Upload a PDF or text file above to get started.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Upload a PDF or text file above to get started.",
        })
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # History for memory (excluding this turn)
                history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")][:-1]
                answer = chat_with_rag(st.session_state.collection, prompt, history)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
