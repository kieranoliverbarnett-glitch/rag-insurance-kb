# Life Insurance Support RAG Chatbot

**[Live Demo →](https://your-app-name.streamlit.app)** ← update this after deployment

---

## The Problem

Support teams at life insurance companies handle a high volume of repetitive, policy-related questions — coverage details, beneficiary changes, underwriting requirements, claim timelines. Agents either memorize this information over time or interrupt their workflow to search through documentation. Both paths introduce inconsistency and slow resolution times.

The deeper problem isn't access to information. It's that most knowledge bases are designed to be read, not queried. An agent asking "what happens to a policy when a beneficiary dies?" doesn't want a list of documents — they want an answer.

## The Solution

A retrieval-augmented generation (RAG) chatbot that grounds Claude's responses in a structured internal knowledge base, rather than general model knowledge. When a question comes in, the system retrieves the most relevant KB sections and passes them as context to the model, which synthesizes a response calibrated to the company's actual policies.

This approach keeps responses accurate and auditable — the sources are visible for every answer — while dramatically reducing the time it takes to surface the right information.

## How It Works

```
User question
     │
     ▼
Convert to embedding (BGE-small-en-v1.5)
     │
     ▼
Vector similarity search (ChromaDB)
     │
     ▼
Top 3 most relevant KB chunks retrieved
     │
     ▼
Chunks + question passed to Claude as context
     │
     ▼
Grounded, professional response returned
```

## Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers |
| Vector store | ChromaDB (in-memory, rebuilt from source on startup) |
| Generation | Anthropic Claude (claude-sonnet-4-6) |
| Deployment | Streamlit Community Cloud |

## Project Structure

```
├── app.py              # Streamlit demo — retrieval + generation pipeline
├── ingest.py           # Local utility: parse and embed KB articles into ChromaDB
├── retrieve.py         # Local utility: test retrieval without generation
├── generate.py         # Local utility: test full pipeline in the terminal
├── knowledge_base/     # Synthetic life insurance KB articles (.md)
├── requirements.txt
└── .gitignore
```

`ingest.py`, `retrieve.py`, and `generate.py` document the build progression — each layer of the pipeline developed and tested independently before being integrated into the deployed app.

## Running Locally

**1. Clone the repo and install dependencies**
```bash
git clone https://github.com/your-username/rag-insurance-kb.git
cd rag-insurance-kb
pip install -r requirements.txt
```

**2. Set your Anthropic API key**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

The app builds the vector store from the `knowledge_base/` articles on first load. This takes ~30 seconds. Subsequent interactions are fast.

## Deployment Notes

Deployed to Streamlit Community Cloud. The `ANTHROPIC_API_KEY` is stored as a Streamlit secret and never committed to the repo.

The vector store is rebuilt in memory at startup rather than committed as binary files, keeping the repository clean and the deployment self-contained.

---

*Built as a portfolio project demonstrating RAG architecture for CX knowledge management use cases.*
