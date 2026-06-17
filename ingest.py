import os
import re
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
KB_PATH = "./knowledge_base"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "insurance_kb"

# --- Load embedding model ---
print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# --- Set up ChromaDB ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

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

# --- Process all articles ---
all_chunks = []
for filename in sorted(os.listdir(KB_PATH)):
    if filename.endswith(".md"):
        filepath = os.path.join(KB_PATH, filename)
        print(f"Parsing: {filename}")
        chunks = parse_article(filepath)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")

# --- Embed and store ---
print("\nEmbedding and storing chunks...")
texts = [c["text"] for c in all_chunks]
metadatas = [c["metadata"] for c in all_chunks]
ids = [f"chunk_{i}" for i in range(len(all_chunks))]

embeddings = model.encode(texts, show_progress_bar=True).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f"\nDone. {len(all_chunks)} chunks stored in ChromaDB.")
