import os
import time
import json
import hashlib
import threading
import queue
from datetime import datetime, timezone
from math import sqrt
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import re

# -------------------- Paths & logging --------------------
LOG_DIR = os.getenv("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "interactions.jsonl")

def _ensure_log_dir():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        pass

def _mask_sender(sender: str) -> str:
    h = hashlib.sha256((sender or "").encode()).hexdigest()
    return f"u_{h[:10]}"

def sanitize_plain(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'[*_`~]+', '', text)               # strip markdown markers
    text = re.sub(r'[\u200b-\u200d\uFEFF]', '', text)  # zero-width chars
    text = re.sub(r'^[•\-\u2022\>]\s*', '- ', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# Async logging (non-blocking)
_LOG_QUEUE: "queue.Queue[tuple[str, dict]]" = queue.Queue(maxsize=1000)

def _log_writer():
    while True:
        try:
            sender, payload = _LOG_QUEUE.get()
            if sender is None:
                continue
            _ensure_log_dir()
            payload = dict(payload)
            payload["ts"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            payload["sender_masked"] = _mask_sender(sender)
            payload.pop("sender", None)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print("log_writer error:", e)
        finally:
            try:
                _LOG_QUEUE.task_done()
            except Exception:
                pass

threading.Thread(target=_log_writer, daemon=True).start()

def log_interaction(sender: str, payload: dict):
    try:
        _LOG_QUEUE.put_nowait((sender, payload))
    except Exception:
        print("log_interaction: queue full; dropping log line.")

# --- Defer only profile extraction (no summarization here) ---
_DEFER_QUEUE: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=1000)

def _defer_worker():
    """Background worker: extract profile fields from messages without blocking the webhook."""
    while True:
        try:
            sender, user_text = _DEFER_QUEUE.get()
            if not sender or not user_text:
                continue

            # Only run if the message looks like it contains profile info
            if _looks_like_profile_update(user_text):
                try:
                    extract_profile_updates(sender, user_text)
                except Exception as e:
                    print("defer extract_profile_updates error:", e)

        except Exception as e:
            print("defer worker error:", e)
        finally:
            try:
                _DEFER_QUEUE.task_done()
            except Exception:
                pass

# start the worker
threading.Thread(target=_defer_worker, daemon=True).start()


# -------------------- Flask --------------------
app = Flask(__name__)

# -------------------- Config --------------------
DB_PATH = os.getenv("DB_PATH", "./db")
COLLECTION = "linkedin_posts"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
DEBUG = os.getenv("DEBUG") == "1"
LAST_MENU_AT = {}          # sender -> unix timestamp
CHOICE_WINDOW = 120        # seconds
RETRIEVAL_TIME_BUDGET = float(os.getenv("RETRIEVAL_TIME_BUDGET", "1.0"))  # seconds

# Retrieval / budget tuning
K = int(os.getenv("K", "4"))
CONF_DIST = float(os.getenv("CONF_DIST", "1.10"))   # tuned for your store; lower=closer
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))
RETRIEVAL_BUDGET = float(os.getenv("RETRIEVAL_BUDGET", "1.5"))      # token multiplier
RETRIEVAL_TIME_BUDGET = float(os.getenv("RETRIEVAL_TIME_BUDGET", "1.0"))  # seconds
TURN_BUDGET = float(os.getenv("TURN_BUDGET", "10.0"))


# Clients
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY before starting Flask.")
openai_client = OpenAI(timeout=5, max_retries=0)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name=EMBED_MODEL
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION, embedding_function=openai_ef
)

# Stores
MAX_HISTORY = 10
USER_PROFILE = {}
CONV_HISTORY = {}
CONV_SUMMARY = {}
WELCOME_SEEN = set()
_CHROMA_CACHE = {}


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
    "1) LinkedIn filters\n"
    "2) Subject lines (recruiter outreach)\n"
    "3) Cold email structure\n"
    "4) Phases of a job search\n"
    "5) Referral ask\n"
    "6) STAR interview stories\n"
    "7) Stay consistent\n\n"
    "Or type 'menu' anytime to see this again."
)


# -------------------- Helpers --------------------
def safe_chat_completion(model, messages, temperature=None, max_tokens=None):
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens or MAX_TOKENS}
    if temperature is not None and not model.startswith("o1"):
        kwargs["temperature"] = temperature
    return openai_client.chat.completions.create(**kwargs)

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

def rough_keyword_match(q: str, doc: str, min_overlap: int = 2) -> bool:
    q_words = set(re.findall(r"[a-zA-Z]{4,}", (q or "").lower()))
    d_words = set(re.findall(r"[a-zA-Z]{4,}", (doc or "").lower()))
    return len(q_words & d_words) >= min_overlap

def summarize_history(sender: str, max_turns: int = 4, max_words: int = 40) -> str:
    """
    Produce a 1–2 sentence running summary of the conversation.
    - Pulls only the last `max_turns` exchanges to keep the prompt tiny.
    - Returns previous summary on any error (never blocks the reply).
    - Keeps fields we care about: role, level, industry, location, constraints, current topic.
    """
    try:
        last = CONV_HISTORY.get(sender, [])[-max_turns:]
        prior = CONV_SUMMARY.get(sender, "")

        # Tiny prompt: compact, deterministic ask
        sys = (
            "You maintain a concise running summary (<= {mw} words) for a careers chat. "
            "Capture: target role, level, industry, location, key constraints, and current topic. "
            "Be factual, no fluff.".format(mw=max_words)
        )
        user = (
            f"Prior summary:\n{prior}\n\n"
            "Recent messages (role: text):\n" +
            "\n".join(f"{t['role']}: {t['text']}" for t in last) +
            f"\n\nUpdate the summary in <= {max_words} words."
        )

        chat = safe_chat_completion(
            CHAT_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2
        )
        new_summary = (chat.choices[0].message.content or "").strip()

        # Minimal sanitization & fallback
        if not new_summary:
            return prior
        CONV_SUMMARY[sender] = new_summary
        if 'DEBUG' in globals() and DEBUG:
            print("[summary]", new_summary)
        return new_summary

    except Exception as e:
        # Never break the request path because of summary
        if 'DEBUG' in globals() and DEBUG:
            print("summarize_history error:", e)
        return CONV_SUMMARY.get(sender, "")


# Heuristic: only try to extract when text likely contains profile info
PROFILE_HINT_RE = re.compile(
    r"\b(role|title|position|level|senior|junior|intern|manager|lead|location|remote|onsite|"
    r"city|country|years|experience|exp|industry|domain|goal|target|aim|looking for|"
    r"switch|transition|linkedin|portfolio|resume|cv|github|url|http[s]?://)\b",
    re.IGNORECASE,
)

def _looks_like_profile_update(text: str) -> bool:
    return bool(text and PROFILE_HINT_RE.search(text))

def extract_profile_updates(sender: str, user_text: str) -> None:
    """
    Pulls structured profile info from the user's message and merges it into USER_PROFILE[sender].
    Keys: role, level, location, years, industry, goals, links.
    - Skips OpenAI call unless the text looks relevant (cheap heuristic).
    - Uses a tiny JSON-only prompt and tolerates bad JSON.
    - Never raises; on any error, does nothing.
    """
    try:
        if not user_text or not _looks_like_profile_update(user_text):
            return  # no-op if the message doesn't look like profile info

        prof = get_profile(sender)

        # --- Tiny, deterministic prompt (JSON only) ---
        sys = (
            "Extract a careers profile from the user's SINGLE message. "
            "Return STRICT JSON ONLY (no code fences). "
            "Allowed keys: role, level, location, years, industry, goals, links. "
            "Rules:\n"
            "- 'years' should be a number if stated (e.g., 3, 5.5), else omit.\n"
            "- 'links' must be a list of URLs if present.\n"
            "- Omit any key you cannot infer from THIS message alone."
        )
        user = f"User message:\n{user_text}\n\nReturn JSON only."

        chat = safe_chat_completion(
            CHAT_MODEL,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.1
        )

        raw = (chat.choices[0].message.content or "").strip()

        # --- Parse JSON defensively ---
        try:
            data = json.loads(raw)
        except Exception:
            # lenient recovery: try to locate a JSON object in the text
            m = re.search(r"\{.*\}", raw, flags=re.S)
            data = json.loads(m.group(0)) if m else {}

        if not isinstance(data, dict):
            return

        # --- Normalize & merge ---
        def _clean_url(u: str) -> str | None:
            if not isinstance(u, str):
                return None
            u = u.strip()
            if not u:
                return None
            if not re.match(r"^https?://", u, flags=re.I):
                # Accept bare domains for common cases
                if re.match(r"^[\w.-]+\.[a-z]{2,}(/.*)?$", u, flags=re.I):
                    u = "https://" + u
                else:
                    return None
            return u

        # Merge links (dedupe)
        if "links" in data and isinstance(data["links"], list):
            new_links = []
            for u in data["links"]:
                cu = _clean_url(u)
                if cu:
                    new_links.append(cu)
            if new_links:
                prof_links = set(prof.get("links", []) or [])
                prof["links"] = list(prof_links.union(new_links))

        # Merge scalar fields if present & non-empty
        for key in ("role", "level", "location", "industry", "goals"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                prof[key] = val.strip()

        # years → numeric if possible
        yrs = data.get("years")
        if isinstance(yrs, (int, float)):
            prof["years"] = yrs
        elif isinstance(yrs, str):
            yrs_m = re.search(r"\d+(\.\d+)?", yrs)
            if yrs_m:
                try:
                    prof["years"] = float(yrs_m.group(0))
                except Exception:
                    pass

        USER_PROFILE[sender] = prof

        if 'DEBUG' in globals() and DEBUG:
            print("[profile]", json.dumps(prof, ensure_ascii=False))

    except Exception as e:
        if 'DEBUG' in globals() and DEBUG:
            print("extract_profile_updates error:", e)
        # swallow errors to avoid impacting the reply path
        return


def build_effective_query(user_msg: str, profile: dict, summary: str) -> str:
    """Build a compact query string to send to retrieval/LLM."""
    parts = [user_msg]

    # Only include the most impactful profile hints
    if profile.get("role"):
        parts.append(f"Target role: {profile['role']}")
    if profile.get("industry"):
        parts.append(f"Industry: {profile['industry']}")

    # Keep summary short, only if it exists
    if summary:
        parts.append(f"Context: {summary}")

    return " | ".join(parts)

# === helpers for compact, fast RAG ===
AVG_TOKENS_PER_WORD = 1.3  # rough; fine for budgeting
def clip_words(txt: str, max_words: int) -> str:
    w = (txt or "").split()
    return " ".join(w[:max_words])

def _budgeted_context(docs: list[str]) -> str:
    # Limit total context size based on RETRIEVAL_BUDGET × MAX_TOKENS
    # Convert token budget to words to keep things model-agnostic.
    total_token_budget = int(RETRIEVAL_BUDGET * MAX_TOKENS)
    total_word_budget  = max(200, int(total_token_budget / AVG_TOKENS_PER_WORD))

    # Per-chunk soft cap so one long chunk doesn’t hog the budget
    per_chunk_cap = max(220, int(total_word_budget * 0.45))

    kept, used = [], 0
    for d in docs:
        if used >= total_word_budget:
            break
        remain = total_word_budget - used
        take = min(per_chunk_cap, remain)
        piece = clip_words(d or "", take)
        if piece.strip():
            kept.append(piece)
            used += len(piece.split())
    return "\n\n".join(kept)


def rag_answer_from_posts(question: str, docs: list[str]) -> str:
    """Answer ONLY from retrieved docs; compact, budgeted context; resilient to errors."""
    try:
        context = _budgeted_context(docs)
        sys = (
            "You are Margil Gandhi’s assistant. Answer ONLY using the provided context. "
            "If the answer is not present, explicitly say you don't have that information. "
            f"Keep the answer concise (<= {MAX_TOKENS} tokens, ~{int(MAX_TOKENS/1.3)} words). "
            "Plain text only."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}"
        chat = safe_chat_completion(
            CHAT_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2
        )
        text = (chat.choices[0].message.content or "").strip()
        # Final safety: hard truncate by words to avoid overruns on WhatsApp
        return clip_words(text, int(MAX_TOKENS / 1.3))
    except Exception as e:
        if 'DEBUG' in globals() and DEBUG:
            print("rag_answer_from_posts error:", e)
        return "Sorry—I ran into an issue using the PDF context. Try asking again in a moment."


def best_practice_answer(query: str) -> str:
    """Fast, general fallback; short, practical; resilient to timeouts/errors."""
    try:
        sys = (
            "You are a careers/job-search assistant. Give a direct, practical answer. "
            f"Keep it concise (<= {MAX_TOKENS} tokens, ~{int(MAX_TOKENS/1.3)} words). "
            "No fluff, plain text."
        )
        chat = safe_chat_completion(
            CHAT_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":query}],
            temperature=0.3
        )
        text = (chat.choices[0].message.content or "").strip()
        return clip_words(text, int(MAX_TOKENS / 1.3))
    except Exception as e:
        if 'DEBUG' in globals() and DEBUG:
            print("best_practice_answer error:", e)
        return "Sorry—I’m a bit busy right now. Please try again."

# ---- heuristics to control expensive work ----
def looks_generic(msg: str) -> bool:
    if not msg: return True
    m = msg.strip().lower()
    # Loosen the short cutoff so we don't over-fallback
    if len(m) < 50: 
        return True
    return any(w in m for w in ["help", "tips", "advice", "how to start"]) and len(m) < 140

def wants_rag(msg: str) -> bool:
    if not msg: return False
    m = msg.lower()
    # Long, specific asks likely benefit from RAG
    if len(m) >= 120:
        return True
    # Explicit RAG cues
    return any(k in m for k in [
        "from my posts", "as i wrote", "from the pdf", "context above",
        "according to your post", "based on your guide"
    ])

def is_first_turn(sender: str) -> bool:
    return sender not in WELCOME_SEEN


def map_menu_choice_to_query(text: str, sender: str | None = None) -> str | None:
    t = (text or "").strip().lower()
    if t == "menu":
        if sender:
            LAST_MENU_AT[sender] = time.time()
        return "__MENU__"

    m = re.match(r'^\s*(\d+)\s*\)?\s*$', t)
    if m and sender:
        if time.time() - LAST_MENU_AT.get(sender, 0) <= CHOICE_WINDOW:
            i = int(m.group(1)) - 1
            if 0 <= i < len(WELCOME_QUESTIONS):
                return WELCOME_QUESTIONS[i]
    return None


# -------------------- Routes --------------------
@app.route("/", methods=["GET"])
def health():
    return "OK"

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    t0 = time.time()
    user_text = (request.form.get("Body") or "").strip()
    sender = (request.form.get("From") or "").strip()

    # ultra-fast exits
    if user_text.lower() == "ping":
        resp = MessagingResponse(); resp.message("pong"); return str(resp)

    # menu handling (now gated by recent menu window inside map_menu_choice_to_query)
    choice = map_menu_choice_to_query(user_text, sender)
    if choice == "__MENU__" or (is_first_turn(sender) and not user_text):
        WELCOME_SEEN.add(sender)
        LAST_MENU_AT[sender] = time.time()   # record menu show time
        remember_turn(sender, "assistant", "(menu shown)")
        resp = MessagingResponse(); resp.message(WELCOME_MENU); return str(resp)

    # quick dev command that bypasses RAG
    if user_text.lower().startswith("!fast "):
        query = user_text.split(" ", 1)[1]
        txt = best_practice_answer(query)
        resp = MessagingResponse(); resp.message(sanitize_plain(txt)); return str(resp)

    # normalize menu number selection (only valid shortly after menu)
    if choice is not None and choice != "__MENU__":
        user_text = choice

    remember_turn(sender, "user", user_text)

    elapsed = lambda: time.time() - t0
    summary = summarize_history(sender)
    profile = get_profile(sender)

    # A) short/generic → single quick chat
    if looks_generic(user_text) or elapsed() > 0.5:
        txt = best_practice_answer(user_text)
        reply = sanitize_plain(txt)
        remember_turn(sender, "assistant", reply)
        _log_reply(sender, user_text, reply, from_pdf=False, confident=False)
        _enqueue_deferred(sender, user_text)  # defer profile extraction only
        resp = MessagingResponse(); resp.message(reply); return str(resp)

    # B) maybe-RAG → quick retrieval, then decide
    confident = False; docs = []
    if wants_rag(user_text) and elapsed() < TURN_BUDGET * 0.4:
        try:
            tR = time.time()
            cache_key = (user_text, K)
            if cache_key in _CHROMA_CACHE:
                res = _CHROMA_CACHE[cache_key]
            else:
                res = collection.query(
                    query_texts=[user_text],
                    n_results=K,
                    include=["documents", "distances"],
                )
                _CHROMA_CACHE[cache_key] = res
            tr = time.time() - tR

            dists = res.get("distances", [[]])[0]
            docs  = res.get("documents", [[]])[0]
            top   = dists[0] if dists else None

            # require: good distance, have docs, and keyword overlap
            confident = (
                top is not None and
                top <= CONF_DIST and
                bool(docs) and
                rough_keyword_match(user_text, docs[0])
            )

            # if retrieval took too long, skip RAG to protect time budget
            if tr > RETRIEVAL_TIME_BUDGET:
                confident = False
        except Exception as e:
            print("retrieval error:", e)
            confident = False
            docs = []

    # C) generate (one chat max)
    eq = build_effective_query(user_text, profile, summary)

    if confident and docs and elapsed() < TURN_BUDGET * 0.7:
        txt = rag_answer_from_posts(eq, docs)
        if not txt or "not in the context" in (txt or "").lower():
            txt = best_practice_answer(user_text)
        reply = sanitize_plain(txt)
        remember_turn(sender, "assistant", reply)
        _log_reply(sender, user_text, reply, from_pdf=True, confident=True)
        _enqueue_deferred(sender, user_text)
        resp = MessagingResponse(); resp.message(reply); return str(resp)
    else:
        txt = best_practice_answer(user_text)
        reply = sanitize_plain(txt)
        remember_turn(sender, "assistant", reply)
        _log_reply(sender, user_text, reply, from_pdf=False, confident=False)
        _enqueue_deferred(sender, user_text)
        resp = MessagingResponse(); resp.message(reply); return str(resp)

def _log_reply(sender, question, reply_text, from_pdf, confident):
    log_interaction(sender, {
        "question": question,
        "from_pdf": from_pdf,
        "confident": confident,
        "answer_len": len(reply_text),
    })

def _enqueue_deferred(sender: str, user_text: str):
    try:
        _DEFER_QUEUE.put_nowait((sender, user_text))
    except Exception:
        # If saturated, skip deferred work rather than affecting next turn latency
        pass


@app.route("/stats", methods=["GET"])
def stats():
    token = request.args.get("token", "")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return "Unauthorized", 401
    path = LOG_FILE
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
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
        lines = [f"Total logged: {total}", "Top questions:"]
        for q, c in top:
            lines.append(f"{c:>3}  {q}")
        return "\n".join(lines), 200
    except Exception as e:
        return f"Error reading logs: {e}", 500

# -------------------- Run --------------------
if __name__ == "__main__":
    print("✅ Flask is starting...")
    try:
        count = collection.count() if hasattr(collection, "count") else "n/a"
        print(f"Chroma collection '{COLLECTION}' count:", count)
    except Exception as e:
        print("Could not count Chroma docs:", e)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=DEBUG)
