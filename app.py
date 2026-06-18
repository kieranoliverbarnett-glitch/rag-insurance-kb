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

# How many prior conversation turns to send to Claude.
# Each "turn" is one user message + one assistant response.
# Keeping this bounded prevents very long conversations from
# hitting token limits or slowing down API calls.
MAX_HISTORY_TURNS = 5


# --- Simple vector store (numpy-based) ---
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
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        similarities = np.dot(self.embeddings, query_vec) / (norms + 1e-10)
        top_indices = np.argsort(similarities)[::-1][:n_results]
        return {
            "documents": [[self.documents[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]]
        }


# --- Article parsing ---
def parse_article(filepath):
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
                    chunks.append({
                        "text": f"From the {article_title} article, FAQ: {qa}",
                        "metadata": {"article": article_title, "section": "FAQ", "source": filename}
                    })
        else:
            chunks.append({
                "text": f"From the {article_title} article, section '{header}': {body}",
                "metadata": {"article": article_title, "section": header, "source": filename}
            })
    return chunks


# --- Initialization ---
@st.cache_resource(show_spinner=False)
def initialize_rag():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    store = SimpleVectorStore()
    all_chunks = []
    for filepath in sorted(Path(KB_PATH).glob("*.md")):
        all_chunks.extend(parse_article(str(filepath)))
    if all_chunks:
        texts = [c["text"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        store.add(documents=texts, embeddings=embeddings, metadatas=metadatas)
    return model, store


def build_retrieval_query(current_query, conversation_history):
    """
    For follow-up questions, retrieval needs context or it searches blindly.
    If there's prior conversation, prepend the last user question to the
    current query before embedding. "Why did that happen?" becomes
    "What causes a policy lapse? Why did that happen?" — which retrieves well.
    """
    prior_user_messages = [
        m for m in conversation_history if m["role"] == "user"
    ]
    if prior_user_messages:
        last_question = prior_user_messages[-1]["content"]
        return f"{last_question} {current_query}"
    return current_query


def retrieve(query, model, store, conversation_history):
    retrieval_query = build_retrieval_query(query, conversation_history)
    query_embedding = model.encode(retrieval_query).tolist()
    return store.query(query_embedding, n_results=TOP_K)


def generate_response(query, chunks, conversation_history):
    """
    Passes the full conversation history to Claude so it can maintain
    continuity across turns. The system parameter sets persistent framing
    that applies to the whole conversation, not just the current message.
    """
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
    client = anthropic.Anthropic(api_key=api_key)

    context = "\n\n".join(chunks)

    # Build the messages array from prior conversation turns.
    # We only pass role and content — the Anthropic API doesn't
    # accept any other keys, so we strip out "sources" and anything else.
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in conversation_history
        if m["role"] in ("user", "assistant")
    ]

    # Add the current question as the final user turn,
    # with the retrieved KB chunks attached as context.
    messages.append({
        "role": "user",
        "content": f"Knowledge base excerpts for this question:\n{context}\n\nCustomer question: {query}"
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        # system sets the persistent behavioral frame for the whole conversation.
        # Previously this was stuffed into the user message and lost after turn one.
        system="""Be professional, empathetic, and clear. Avoid casual language and filler phrases like "Great question!"
Acknowledge the customer's situation with care, provide a direct and accurate answer, and always
offer to connect them with a representative if their needs are complex or time-sensitive.""",
        messages=messages
    )

    return response.content[0].text


# --- UI ---
st.title("🛡️ Life Insurance Support Assistant")
st.caption(
    "A retrieval-augmented generation (RAG) demo — responses are grounded in a "
    "structured knowledge base, not general model knowledge."
)

st.divider()

with st.spinner("Loading knowledge base and embedding model..."):
    model, store = initialize_rag()

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

if prompt := st.chat_input("Ask a life insurance question..."):

    # Add user message to history first (for display)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            # Pass history *excluding* the current message we just appended,
            # bounded to the last MAX_HISTORY_TURNS turns (each turn = 2 messages).
            history_for_retrieval = st.session_state.messages[:-1][-(MAX_HISTORY_TURNS * 2):]
            history_for_claude = st.session_state.messages[:-1][-(MAX_HISTORY_TURNS * 2):]

            results = retrieve(prompt, model, store, history_for_retrieval)
            chunks = results["documents"][0]
            metadatas = results["metadatas"][0]
            response = generate_response(prompt, chunks, history_for_claude)

        st.write(response)

        with st.expander("View sources", expanded=False):
            for meta in metadatas:
                st.caption(f"📄 {meta['article']} — {meta['section']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": metadatas
    })
