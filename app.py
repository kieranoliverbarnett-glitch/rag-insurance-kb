import streamlit as st
import chromadb
import anthropic
import os
import re
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --- Page config ---
st.set_page_config(
    page_title="Life Insurance Support Assistant",
    page_icon="🛡️",
    layout="centered"
)

# --- Configuration ---
COLLECTION_NAME = "insurance_kb"
KB_PATH = "knowledge_base"
TOP_K = 3


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

    # In-memory ChromaDB — rebuilt each session from source articles
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    kb_path = Path(KB_PATH)
    all_chunks = []

    for filepath in sorted(kb_path.glob("*.md")):
        chunks = parse_article(str(filepath))
        all_chunks.extend(chunks)

    if all_chunks:
        texts = [c["text"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]
        ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    return model, collection


def retrieve(query, model, collection, top_k=TOP_K):
    """Convert query to embedding and find most similar chunks."""
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return results


def generate_response(query, chunks, metadatas):
    """Pass retrieved chunks and query to Claude and return response."""
    # Works with Streamlit secrets (cloud) or environment variable (local)
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

# Load RAG system with visible spinner
with st.spinner("Loading knowledge base and embedding model..."):
    model, collection = initialize_rag()

# Initialize chat history in session state
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

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Retrieve and generate
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            results = retrieve(prompt, model, collection)
            chunks = results["documents"][0]
            metadatas = results["metadatas"][0]
            response = generate_response(prompt, chunks, metadatas)

        st.write(response)

        with st.expander("View sources", expanded=False):
            for meta in metadatas:
                st.caption(f"📄 {meta['article']} — {meta['section']}")

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": metadatas
    })
