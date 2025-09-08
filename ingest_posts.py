# ingest_posts.py
import os, glob, json, hashlib, fitz, chromadb
from chromadb.utils import embedding_functions

DATA_DIR    = "data"                         # PDFs folder
STATE_PATH  = os.path.join(DATA_DIR, ".ingest_state.json")
DB_PATH     = os.getenv("DB_PATH", "./db")
COLLECTION  = "linkedin_posts"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # must match app.py
SCHEMA_VER  = "v1"
CHUNK_SIZE  = 320
OVERLAP     = 64

def _read_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _sanitize_text(t: str) -> str:
    # Light cleanup to stabilize hashing and chunking
    import re
    t = re.sub(r'[\u200b-\u200d\uFEFF]', '', t)      # zero-widths
    t = re.sub(r'[ \t]+', ' ', t)                    # collapse spaces
    t = re.sub(r'\s+\n', '\n', t)                    # trim line tails
    t = re.sub(r'\n{3,}', '\n\n', t)                 # limit blank lines
    return t.strip()

def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    parts = [page.get_text("text").strip() for page in doc]
    doc.close()
    return _sanitize_text("\n\n".join(parts))

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

    # Embedding fn used for both adds and queries
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL,
    )

    # Get/create collection with metadata declaring embed model + schema
    col = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=openai_ef,
        metadata={"embed_model": EMBED_MODEL, "schema_version": SCHEMA_VER},
    )

    # Hard check: existing collection model must match unless forcing reindex
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

    # Optional full reset
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

    # Incremental ingest with per-file hashing
    state = _read_state()
    total_chunks = 0
    total_skipped = 0
    total_replaced = 0

    for path in pdf_paths:
        fname = os.path.basename(path)
        stem  = os.path.splitext(fname)[0]

        # Extract and hash full text to detect changes
        text = load_pdf_text(path)
        file_hash = _sha256(text)
        prev_hash = (state.get(fname) or {}).get("file_hash")

        if prev_hash == file_hash:
            print(f"⏭  Unchanged, skipping: {fname}")
            total_skipped += 1
            continue

        # New or changed → re-chunk
        chunks = chunk_text(text)

        # Versioned doc_id keeps IDs stable per content version
        doc_id = f"{stem}:{file_hash[:12]}"
        ids    = [f"{doc_id}-chunk-{i}" for i in range(len(chunks))]
        metas  = [{
            "source": fname,
            "idx": i,
            "doc_id": doc_id,
            "file_hash": file_hash,
            "embed_model": EMBED_MODEL,
            "schema_version": SCHEMA_VER
        } for i in range(len(chunks))]

        # If we had an older version for this file, delete its chunks by metadata
        if prev_hash:
            try:
                # Remove previous version (by source + schema) to avoid bloat
                col.delete(where={"source": fname, "schema_version": SCHEMA_VER})
                total_replaced += 1
                print(f"♻️  Replacing older version of: {fname}")
            except Exception as e:
                print(f"Delete previous version warning ({fname}):", e)

        # Upsert the new version
        col.upsert(documents=chunks, ids=ids, metadatas=metas)
        print(f"✅ Indexed {len(chunks)} chunks from {fname} (doc_id={doc_id})")
        total_chunks += len(chunks)

        # Persist new hash in state
        state[fname] = {"file_hash": file_hash}

    # Save updated state
    _write_state(state)

    try:
        print(f"📦 Collection '{COLLECTION}' now has ~{col.count()} items")
    except Exception:
        pass

    print(f"🎉 Done. New/updated chunks: {total_chunks} | Unchanged skipped: {total_skipped} | Files replaced: {total_replaced}")
    print(f"🔐 embed_model asserted: {EMBED_MODEL} | schema: {SCHEMA_VER}")

if __name__ == "__main__":
    main()
