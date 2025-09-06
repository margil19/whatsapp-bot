# app.py
import os
import time
import json
import hashlib
from datetime import datetime, timezone
from math import sqrt
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import re

# Prefer Render’s mount if available, else fallback to ./logs
LOG_DIR = os.getenv("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "interactions.jsonl")


def sanitize_plain(text: str) -> str:
    if not text: return text
    # Remove bold/italic markers and odd zero-width chars
    text = re.sub(r'[*_`~]+', '', text)              # strip markdown markers
    text = re.sub(r'[\u200b-\u200d\uFEFF]', '', text) # zero-width chars
    # Normalize bullets to "- "
    text = re.sub(r'^[•\-\u2022\>]\s*', '- ', text, flags=re.MULTILINE)
    # Collapse extra spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def _ensure_log_dir():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        pass

def _mask_sender(sender: str) -> str:
    # avoid storing raw phone numbers
    h = hashlib.sha256((sender or "").encode()).hexdigest()
    return f"u_{h[:10]}"

def log_interaction(sender: str, payload: dict):
    """Append one JSON line to logs/interactions.jsonl"""
    _ensure_log_dir()
    path = os.path.join(LOG_DIR, "interactions.jsonl")
    try:
        payload = dict(payload)  # shallow copy
        payload["ts"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload["sender_masked"] = _mask_sender(sender)
        # never store raw sender id
        payload.pop("sender", None)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print("log_interaction error:", e)

# -------------------- Flask --------------------
app = Flask(__name__)

# -------------------- Config --------------------
DB_PATH = os.getenv("DB_PATH", "./db")
COLLECTION = "linkedin_posts"
EMBED_MODEL = "text-embedding-3-small"  # for RAG
CHAT_MODEL = "gpt-4o-mini"              # switchable chat model
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")          # set on Render to protect /stats
FAST_MODE = os.getenv("FAST_MODE") == "1"

# Retrieval tuning
K = 6
CONF_THRESH = 1.05

# Conversational memory
MAX_HISTORY = 10

# In-memory stores
USER_PROFILE = {}
CONV_HISTORY = {}
CONV_SUMMARY = {}
# Track who has already seen the welcome menu
WELCOME_SEEN = set()


# ---- Welcome menu / examples ----
WELCOME_QUESTIONS = [
    "How can I make the most of LinkedIn filters to find opportunities?",
    "Give me 10 subject lines for cold outreach to a recruiter.",
    "How should I structure a cold email to a recruiter?",
    "What are the key phases of a job search and what should I do in each?",
    "How do I ask for a referral effectively?",
    "How do I use the STAR method to answer behavioral questions?",
    "How can I stay consistent in my job search without burning out?"
]

WELCOME_MENU = (
    "Hi! I can help with job search.\n"
    "Reply with a number or type your own question:\n\n"
    "1) LinkedIn filters (find live opportunities)\n"
    "2) Subject lines (recruiter outreach)\n"
    "3) Cold email structure\n"
    "4) Phases of a job search\n"
    "5) Referral ask\n"
    "6) STAR interview stories\n"
    "7) Stay consistent / avoid burnout\n\n"
    "Or type 'menu' anytime to see this again."
)
# -------------------- Safety --------------------
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY environment variable before starting Flask.")

# -------------------- Clients --------------------
openai_client = OpenAI(timeout=8)

# Chroma vector store
chroma_client = chromadb.PersistentClient(path=DB_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name=EMBED_MODEL
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION, embedding_function=openai_ef
)

# Startup sanity log
try:
    count = collection.count() if hasattr(collection, "count") else "n/a"
    print(f"Chroma collection '{COLLECTION}' count:", count)
except Exception as e:
    print("Could not count Chroma docs:", e)

# -------------------- Safe wrapper --------------------
def safe_chat_completion(model, messages, temperature=None):
    kwargs = {"model": model, "messages": messages}
    # Only pass temperature if model supports it (o1-family requires default)
    if temperature is not None and not model.startswith("o1"):
        kwargs["temperature"] = temperature
    return openai_client.chat.completions.create(**kwargs)

# -------------------- Utilities --------------------
def remember_turn(sender: str, role: str, text: str) -> None:
    hist = CONV_HISTORY.get(sender) or []
    hist.append({"role": role, "text": text})
    CONV_HISTORY[sender] = hist[-MAX_HISTORY:]

def get_profile(sender: str) -> dict:
    prof = USER_PROFILE.get(sender) or {
        "role": None, "level": None, "location": None,
        "years": None, "industry": None, "goals": None,
        "links": [], "prefs": {}
    }
    USER_PROFILE[sender] = prof
    return prof

def cosine_sim(a, b) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(y*y for y in b))
    denom = na * nb
    return dot / denom if denom > 1e-9 else 0.0

def embed_text(text: str):
    emb = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return emb.data[0].embedding

# ---- Lightweight keyword sanity check for RAG ----
def rough_keyword_match(q: str, doc: str, min_overlap: int = 2) -> bool:
    """Require a minimal overlap of 4+ letter words to accept a chunk as on-topic."""
    q_words = set(re.findall(r"[a-zA-Z]{4,}", (q or "").lower()))
    d_words = set(re.findall(r"[a-zA-Z]{4,}", (doc or "").lower()))
    return len(q_words & d_words) >= min_overlap

# -------------------- Memory updaters --------------------
def summarize_history(sender: str) -> str:
    last = CONV_HISTORY.get(sender, [])
    prior = CONV_SUMMARY.get(sender, "")
    sys = (
        "You maintain a 1–2 sentence running summary for a careers chat. "
        "Capture the user's target role/level/industry/location, constraints, and current topic."
    )
    content = (
        f"Prior summary:\n{prior}\n\n"
        "Recent messages:\n" + "\n".join(f"{t['role']}: {t['text']}" for t in last[-4:]) +
        "\n\nUpdate the summary in <= 40 words:"
    )
    chat = safe_chat_completion(
        CHAT_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":content}],
        temperature=0.2
    )
    new_summary = chat.choices[0].message.content.strip()
    CONV_SUMMARY[sender] = new_summary
    return new_summary

def extract_profile_updates(sender: str, user_text: str) -> None:
    sys = (
        "Extract a careers profile as strict JSON (no code block). "
        "Allowed keys: role, level, location, years, industry, goals, links."
    )
    chat = safe_chat_completion(
        CHAT_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":user_text}]
    )
    prof = get_profile(sender)
    try:
        data = json.loads(chat.choices[0].message.content)
        text_l = (user_text or "").lower()
        for k, v in data.items():
            if not v:
                continue
            if k == "links" and isinstance(v, list):
                prof["links"] = list({*prof.get("links", []), *v})
            elif k == "role":
                # Only update role if the user hints at it explicitly in THIS message
                if any(t in text_l for t in ["role", "target", "position", "title", "aiming", "interested in"]):
                    prof["role"] = v
            else:
                prof[k] = v
        USER_PROFILE[sender] = prof
    except Exception:
        # tolerate parsing issues silently
        pass

# -------------------- Planner & answering --------------------
def build_effective_query(user_msg: str, profile: dict, summary: str) -> str:
    parts = [user_msg]
    if profile.get("role"): parts.append(f"Target role: {profile['role']}")
    if profile.get("level"): parts.append(f"Level: {profile['level']}")
    if profile.get("industry"): parts.append(f"Industry: {profile['industry']}")
    if profile.get("location"): parts.append(f"Location: {profile['location']}")
    if summary: parts.append(f"Context: {summary}")
    return " | ".join(parts)

def rag_answer_from_posts(question: str, docs: list[str]) -> str:
    context = "\n\n".join(docs)
    sys = (
        "You are Margil Gandhi’s assistant. Answer ONLY using the context below. "
        "If the answer isn't in the context, say you don't have that information. "
        "Use plain text only."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    chat = safe_chat_completion(
        CHAT_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.2
    )
    return chat.choices[0].message.content.strip()

def best_practice_answer(query: str) -> str:
    sys = (
        "You are a careers/job-search assistant. Provide a concise, practical answer. "
        "Use plain text only. Keep it under 120 words. "
        "Do not assume a specific role, level, or industry unless the user states it explicitly in THIS message."
    )
    chat = safe_chat_completion(
        CHAT_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":query}],
        temperature=0.3
    )
    return chat.choices[0].message.content.strip()

def answer_from_pdf_or_fallback(raw_question: str, effective_query: str):
    """Retrieve from Chroma and answer; fall back to best-practice if low confidence or slow."""
    import time

    # Use fewer neighbors in FAST_MODE to keep latency safely under Twilio's 15s
    k = 3 if FAST_MODE else K

    # 1) Retrieve with RAW user question
    t0 = time.time()
    res = collection.query(
        query_texts=[raw_question],
        n_results=k,
        include=["documents", "distances"],
    )
    t1 = time.time()
    print(f"[TIMING] chroma.query took {t1 - t0:.2f}s")

    docs  = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    top_distance = dists[0] if dists else None

    # Debug logs
    print("Distances:", dists)
    if top_distance is not None:
        print(f"Top distance={top_distance:.3f} | approx cosine_similarity={1 - top_distance:.3f}")
    if docs:
        print("Top doc snippet:", (docs[0] or "")[:240])

    # If nothing returned, go straight to best-practice
    if not docs or not dists:
        text = best_practice_answer(raw_question)
        telemetry = {
            "from_pdf": False,
            "confident": False,
            "top_distance": top_distance,
            "distances": dists,
            "used_fallback": True,
        }
        return text, False, telemetry

    # 2) Distance gate
    confident = (top_distance is not None) and (top_distance <= CONF_THRESH)

    # 3) Lightweight keyword sanity check
    if confident and not rough_keyword_match(raw_question, docs[0]):
        confident = False

    # Optional extra guard: if we're far beyond threshold, bail early
    if not confident and top_distance is not None and top_distance > (CONF_THRESH + 0.10):
        text = best_practice_answer(raw_question)
        telemetry = {
            "from_pdf": False,
            "confident": False,
            "top_distance": top_distance,
            "distances": dists,
            "used_fallback": True,
        }
        print("[Fallback] Distance well above threshold; skipping RAG.")
        return text, False, telemetry

    if confident:
        # 4) Generate from posts context
        t2 = time.time()
        text = rag_answer_from_posts(effective_query, docs)
        t3 = time.time()
        print(f"[TIMING] rag_answer_from_posts (chat) took {t3 - t2:.2f}s")

        # 5) Post-generation guard: if RAG says missing info, fall back
        if any(p in (text or "").lower() for p in [
            "don't have that information",
            "not in the context",
            "context does not cover",
        ]):
            print("[RAG->Fallback] RAG reported missing info; using best-practice.")
            text = best_practice_answer(raw_question)
            telemetry = {
                "from_pdf": False,
                "confident": False,
                "top_distance": top_distance,
                "distances": dists,
                "used_fallback": True,
            }
            return text, False, telemetry

        print("[RAG] Answered from PDF.")
        telemetry = {
            "from_pdf": True,
            "confident": True,
            "top_distance": top_distance,
            "distances": dists,
            "used_fallback": False,
        }
        return text, True, telemetry

    # If not confident, use best-practice fallback
    print("[Fallback] Weak distance or keyword overlap; using best-practice.")
    text = best_practice_answer(raw_question)
    telemetry = {
        "from_pdf": False,
        "confident": False,
        "top_distance": top_distance,
        "distances": dists,
        "used_fallback": True,
    }
    return text, False, telemetry



def is_first_turn(sender: str) -> bool:
    return sender not in WELCOME_SEEN

def map_menu_choice_to_query(text: str) -> str | None:
    t = (text or "").strip().lower()
    if t == "menu": return "__MENU__"
    if t.isdigit():
        i = int(t) - 1
        if 0 <= i < len(WELCOME_QUESTIONS):
            return WELCOME_QUESTIONS[i]
    return None


# -------------------- Routes --------------------
@app.route("/", methods=["GET"])
def health():
    return "OK"

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    user_text = (request.form.get("Body") or "").strip()
    # ultra-fast ping to confirm Twilio <-> Render path
    if user_text.strip().lower() == "ping":
        resp = MessagingResponse()
        resp.message("pong")
        return str(resp)
    # quick path to test chat-only (skips RAG)
    if user_text.strip().lower().startswith("!fast "):
        query = user_text.split(" ", 1)[1]
        reply_text = best_practice_answer(query)  # single chat call
        resp = MessagingResponse(); resp.message(sanitize_plain(reply_text))
        return str(resp)

    sender = (request.form.get("From") or "").strip()

    # --- Show menu on first turn or when user types 'menu' ---
    # --- Show menu on first turn or when user types 'menu' ---
    choice = map_menu_choice_to_query(user_text)
    if is_first_turn(sender) or choice == "__MENU__":
        WELCOME_SEEN.add(sender)  # mark that the welcome has been shown
        # optional: put a tiny marker in history to avoid empty-history edge cases
        remember_turn(sender, "assistant", "(menu shown)")
        resp = MessagingResponse()
        resp.message(WELCOME_MENU)
        return str(resp)

    # If user picked a number 1..7, replace their text with the mapped example
    if choice is not None and choice != "__MENU__":
        user_text = choice

    # --- continue with your existing pipeline ---
    remember_turn(sender, "user", user_text)

    if FAST_MODE:
        # Skip extra OpenAI calls to avoid Twilio 15s timeout
        profile = get_profile(sender)   # keep whatever we already know
        summary = ""                    # skip summarize_history
    else:
        extract_profile_updates(sender, user_text)
        summary = summarize_history(sender)
        profile = get_profile(sender)

    effective_query = build_effective_query(user_text, profile, summary)
    answer, from_pdf, telemetry = answer_from_pdf_or_fallback(user_text, effective_query)

    tail = ""
    history = CONV_HISTORY.get(sender, [])
    if not from_pdf and len(history) >= 2:
        missing = []
        if not profile.get("role"):  missing.append("target role")
        if not profile.get("level"): missing.append("level")
        if missing:
            tail = f"\n\n(Optional: Share your {', '.join(missing)} to get more personalized answers.)"

    reply_text = sanitize_plain(answer + tail)
    remember_turn(sender, "assistant", reply_text)

    # --- telemetry log ---
    log_interaction(sender, {
    "question": user_text,
    "effective_query": effective_query,
    "from_pdf": telemetry.get("from_pdf"),
    "confident": telemetry.get("confident"),
    "top_distance": telemetry.get("top_distance"),
    "distances": telemetry.get("distances"),
    "answer_len": len(reply_text),
})

    resp = MessagingResponse()
    resp.message(reply_text)
    return str(resp)

@app.route("/stats", methods=["GET"])
def stats():
    token = request.args.get("token", "")
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        return "Unauthorized", 401

    path = os.path.join(LOG_DIR, "interactions.jsonl")
    if not os.path.exists(path):
        return "No logs yet.", 200

    counts = {}
    total = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    q = (item.get("question") or "").strip().lower()
                    if not q:
                        continue
                    counts[q] = counts.get(q, 0) + 1
                    total += 1
                except Exception:
                    continue
        # top 20
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
        # simple text response
        lines = [f"Total logged: {total}", "Top questions:"]
        for q, c in top:
            lines.append(f"{c:>3}  {q}")
        return "\n".join(lines), 200
    except Exception as e:
        return f"Error reading logs: {e}", 500


# -------------------- Run --------------------

if __name__ == "__main__":
    print("✅ Flask is starting...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=True)

