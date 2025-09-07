# ingest_posts.py
import os, glob, fitz, chromadb
from chromadb.utils import embedding_functions

DATA_DIR    = "data"                         # PDFs folder
DB_PATH     = os.getenv("DB_PATH", "./db")
COLLECTION  = "linkedin_posts"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # must match app.py
SCHEMA_VER  = "v1"
CHUNK_SIZE  = 320
OVERLAP     = 64

def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    parts = [page.get_text("text").strip() for page in doc]
    doc.close()
    return "\n\n".join(parts)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words, chunks, i = text.split(), [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += step
    return [c for c in chunks if c.strip()]

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running.")

    pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}/")

    client = chromadb.PersistentClient(path=DB_PATH)

    # Build the embedding function used for BOTH adds and queries
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL,
    )

    # Get or create the collection, with metadata declaring the embed model
    col = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=openai_ef,
        metadata={"embed_model": EMBED_MODEL, "schema_version": SCHEMA_VER},
    )

    # ---- Hard check: existing collection model must match ----
    existing_meta = (col.metadata or {})
    existing_model = existing_meta.get("embed_model")
    if existing_model and existing_model != EMBED_MODEL and os.getenv("FORCE_REINDEX") != "1":
        raise RuntimeError(
            f"[ERROR] Collection '{COLLECTION}' was built with embed_model='{existing_model}', "
            f"but you are ingesting with EMBED_MODEL='{EMBED_MODEL}'.\n"
            f"Fix one of:\n"
            f"  1) Set EMBED_MODEL='{existing_model}' to match the collection/app, OR\n"
            f"  2) Recreate the collection (new name), OR\n"
            f"  3) Run again with FORCE_REINDEX=1 to overwrite."
        )

    # If forcing reindex, wipe and reset metadata
    if os.getenv("FORCE_REINDEX") == "1":
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=openai_ef,
            metadata={"embed_model": EMBED_MODEL, "schema_version": SCHEMA_VER},
        )
        print(f"⚠️  FORCE_REINDEX=1: Recreated collection '{COLLECTION}'")

    total_chunks = 0
    for path in pdf_paths:
        fname = os.path.basename(path)
        stem  = os.path.splitext(fname)[0]

        text   = load_pdf_text(path)
        chunks = chunk_text(text)

        ids    = [f"{stem}-chunk-{i}" for i in range(len(chunks))]
        metas  = [{"source": fname, "idx": i, "embed_model": EMBED_MODEL, "schema_version": SCHEMA_VER}
                  for i in range(len(chunks))]

        # Upsert is idempotent for same IDs (updates doc/metadata+re-embeds if model matches)
        col.upsert(documents=chunks, ids=ids, metadatas=metas)

        print(f"✅ Indexed {len(chunks)} chunks from {fname}")
        total_chunks += len(chunks)

    try:
        print(f"📦 Collection '{COLLECTION}' now has ~{col.count()} items")
    except Exception:
        pass

    print(f"🎉 Done. Total upserted chunks: {total_chunks}")
    # Final assert: ensure app and ingest are aligned
    print(f"🔐 embed_model asserted: {EMBED_MODEL}")

if __name__ == "__main__":
    main()
