# ingest_posts.py
import os
import glob
import fitz
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR   = "data"                     # folder with one or more PDFs
DB_PATH    = os.getenv("DB_PATH", "./db")
COLLECTION = "linkedin_posts"
EMBED_MODEL = "text-embedding-3-small"  # must match app.py

CHUNK_SIZE = 650
OVERLAP    = 130

def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text").strip())
    doc.close()
    return "\n\n".join(parts)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks, i = [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += step
    return [c for c in chunks if c.strip()]

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY env var before running.")

    # Find PDFs
    pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}/")

    # Chroma setup
    client = chromadb.PersistentClient(path=os.getenv("DB_PATH", "./db"))

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name=EMBED_MODEL,
    )
    try:
        col = client.get_collection(name=COLLECTION, embedding_function=openai_ef)
    except Exception:
        col = client.create_collection(name=COLLECTION, embedding_function=openai_ef)

    
    total_chunks = 0

    for path in pdf_paths:
        fname = os.path.basename(path)
        stem  = os.path.splitext(fname)[0]

        text = load_pdf_text(path)
        chunks = chunk_text(text)

        # Deterministic IDs so repeated runs don't duplicate
        ids   = [f"{stem}-chunk-{i}" for i in range(len(chunks))]
        metas = [{"source": fname, "idx": i} for i in range(len(chunks))]

        # Upsert replaces existing IDs or adds new ones
        col.upsert(documents=chunks, ids=ids, metadatas=metas)

        print(f"✅ Indexed {len(chunks)} chunks from {fname}")
        total_chunks += len(chunks)

    # Optional: report count if available
    try:
        print(f"📦 Collection '{COLLECTION}' now has ~{col.count()} items")
    except Exception:
        pass

    print(f"🎉 Done. Total upserted chunks: {total_chunks}")

if __name__ == "__main__":
    main()
