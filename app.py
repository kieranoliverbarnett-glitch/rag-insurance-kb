import streamlit as st
import anthropic
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --- Page config ---
st.set_page_config(
    page_title="Life Insurance Support Assistant",
    page_icon="🛡️",
    layout="centered"
)

# --- Configuration ---
KB_PATH = "knowledge_base"
TOP_K = 3


# --- Simple vector store (numpy-based, no external DB needed) ---
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.embeddings = None

    def add(self, documents, embeddings, metadatas):
        self.documents = documents
        self.metadatas = metadatas
        self.embeddings = np.array(embeddings)

    def query(self, query_embedding, n_results=3):
        query_vec = np.array(query_embedding)
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        similarities = np.dot(self.embeddings, query_vec) / (norms + 1e-10)
        top_indices = np.argsort(similarities)[::-1][:n_results]
        return {
            "documents": [[self.documents[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]]
        }


# --- Helper functions ---
def parse_article(filepath):
    """Split a markdown article into chunks at ## boundaries,
    with FAQ entries split into individual chunks."""
    with open(filepath, "r") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    article_title = filename.replace(".md", "").replace("_", " ").title()

    chunks = []
    sections = re.split(r'\n(?=## )', content)

    for section in sections:
        lines = section.strip().split("\n")
        if not lines:
            continue
        header = lines[0].strip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue

        if "common customer questions" in header.lower():
            qa_pairs = re.split(r'\n(?=\*\*)', body)
            for qa in qa_pairs:
                qa = qa.strip()
                if qa:
                    chunk_text = f"From the {article_title} article, FAQ: {qa}"
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "article": article_title,
                            "section": "FAQ",
                            "source": filename
                        }
                    })
        else:
            chunk_text = f"From the {article_title} article, section '{header}': {body}"
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "article": article_title,
                    "section": header,
                    "source": filename
                }
            })

    return chunks


# --- Cached resource initialization ---
@st.cache_resource(show_spinner=False)
def initialize_rag():
    """Load embedding model and build vector store from KB articles.
    Cached per session — runs once, shared across all users."""
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    store = SimpleVectorStore()

    kb_path = Path(KB_PATH)
    all_chunks = []

    for filepath in sorted(kb_path.glob("*.md")):
        chunks = parse_article(str(filepath))
        all_chunks.extend(chunks)

    if all_chunks:
        texts = [c["text"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        store.add(documents=texts, embeddings=embeddings, metadatas=metadatas)

    return model, store


def retrieve(query, model, store, top_k=TOP_K):
    """Convert query to embedding and find most similar chunks."""
    query_embedding = model.encode(query).tolist()
    return store.query(query_embedding, n_results=top_k)


def generate_response(query, chunks):
    """Pass retrieved chunks and query to Claude and return response."""
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
    client = anthropic.Anthropic(api_key=api_key)

    context = "\n\n".join(chunks)

    prompt = f"""Be professional, empathetic, and clear. Avoid casual language and filler phrases like "Great question!"
Acknowledge the customer's situation with care, provide a direct and accurate answer, and always
offer to connect them with a representative if their needs are complex or time-sensitive.

Knowledge base excerpts:
{context}

Customer question: {query}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


# --- UI ---
st.title("🛡️ Life Insurance Support Assistant")
st.caption(
    "A retrieval-augmented generation (RAG) demo — responses are grounded in a "
    "structured knowledge base, not general model knowledge."
)

st.divider()

# Load RAG system
with st.spinner("Loading knowledge base and embedding model..."):
    model, store = initialize_rag()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("View sources", expanded=False):
                for src in message["sources"]:
                    st.caption(f"📄 {src['article']} — {src['section']}")

# Chat input
if prompt := st.chat_input("Ask a life insurance question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            results = retrieve(prompt, model, store)
            chunks = results["documents"][0]
            metadatas = results["metadatas"][0]
            response = generate_response(prompt, chunks)

        st.write(response)

        with st.expander("View sources", expanded=False):
            for meta in metadatas:
                st.caption(f"📄 {meta['article']} — {meta['section']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": metadatas
    })
