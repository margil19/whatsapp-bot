from flask import Response
import os
import time
import json
import hashlib
import threading
import queue
from datetime import datetime, timezone
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from collections import OrderedDict
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import re

# ==================== Paths & logging ====================
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

# Insert a newline before inline numbered items like " 1. " or " 2) "
LISTIFY_RE = re.compile(r'(?<!\n)\s(\d{1,2}[.)])\s')

def enforce_numbered_lines(text: str) -> str:
    if not text:
        return text
    # Normalize weird spaces first (NBSP, narrow NBSP)
    text = re.sub(r'[\u00A0\u202F]', ' ', text)
    # Insert newline before numbered tokens glued to a paragraph
    text = LISTIFY_RE.sub(r'\n\1 ', text)
    # Compress excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def sanitize_plain(text: str) -> str:
    if not text:
        return text
    # strip markdown & zero-widths (incl. WORD JOINER \u2060)
    text = re.sub(r'[*_`~]+', '', text)
    text = re.sub(r'[\u200b-\u200d\uFEFF\u2060]', '', text)
    # normalize NBSPs to real spaces
    text = re.sub(r'[\u00A0\u202F]', ' ', text)
    # normalize bullet markers and compress whitespace
    text = re.sub(r'^[•\-\u2022\>]\s*', '- ', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def _open_log_file():
    # Simple size-based rollover at 10MB
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 10 * 1024 * 1024:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            os.replace(LOG_FILE, f"{LOG_FILE}.{ts}")
    except Exception:
        pass
    return open(LOG_FILE, "a", encoding="utf-8")

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
            with _open_log_file() as f:
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

threading.Thread(target=_defer_worker, daemon=True).start()

# ==================== Flask ====================
app = Flask(__name__)

# ==================== Config ====================
DB_PATH = os.getenv("DB_PATH", "./db")
COLLECTION = "linkedin_posts"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
DEBUG = os.getenv("DEBUG") == "1"
LAST_MENU_AT = {}          # sender -> unix timestamp
CHOICE_WINDOW = 60         # seconds

# Retrieval / budget tuning
K = int(os.getenv("K", "4"))
CONF_DIST = float(os.getenv("CONF_DIST", "0.40"))   # tuned for cosine distance; smaller is better
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))
RETRIEVAL_BUDGET = float(os.getenv("RETRIEVAL_BUDGET", "1.5"))      # token multiplier
RETRIEVAL_TIME_BUDGET = float(os.getenv("RETRIEVAL_TIME_BUDGET", "5.0"))  # seconds
TURN_BUDGET = float(os.getenv("TURN_BUDGET", "10.0"))

# ==================== Clients ====================
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY before starting Flask.")
# Slightly softer client on transient issues
openai_client = OpenAI(timeout=8, max_retries=2)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name=EMBED_MODEL
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION,
    embedding_function=openai_ef,
    metadata={
        "hnsw:space": "cosine",       # <-- sets cosine distance
        "embed_model": EMBED_MODEL,   # keep model alignment info
    },
)

# ==================== Bounded in-memory stores ====================
MAX_HISTORY = 10

class SenderMap(dict):
    """LRU-ish per-sender container to avoid unbounded growth."""
    def __init__(self, max_senders=5000):
        super().__init__()
        self.order = OrderedDict()
        self.max_senders = max_senders
        self.lock = threading.Lock()

    def touch(self, sender):
        with self.lock:
            self.order.pop(sender, None)
            self.order[sender] = True
            if len(self.order) > self.max_senders:
                old, _ = self.order.popitem(last=False)
                self.pop(old, None)

CONV_HISTORY = SenderMap()
CONV_SUMMARY = SenderMap()
USER_PROFILE = SenderMap()
WELCOME_SEEN = set()

def remember_turn(sender: str, role: str, text: str) -> None:
    hist = CONV_HISTORY.get(sender) or []
    hist.append({"role": role, "text": text})
    CONV_HISTORY[sender] = hist[-MAX_HISTORY:]
    CONV_HISTORY.touch(sender)

def get_profile(sender: str) -> dict:
    prof = USER_PROFILE.get(sender) or {
        "role": None, "level": None, "location": None,
        "years": None, "industry": None, "goals": None,
        "links": [], "prefs": {}
    }
    USER_PROFILE[sender] = prof
    USER_PROFILE.touch(sender)
    return prof

# Bounded, thread-safe LRU cache for retrieval results
_cache_lock = threading.Lock()
class LRUCache(OrderedDict):
    def __init__(self, maxsize=256):
        super().__init__()
        self.maxsize = maxsize
    def get_set(self, key, compute_fn):
        with _cache_lock:
            if key in self:
                self.move_to_end(key)
                return self[key]
        # compute outside lock to avoid blocking others
        value = compute_fn()
        with _cache_lock:
            self[key] = value
            self.move_to_end(key)
            if len(self) > self.maxsize:
                self.popitem(last=False)
        return value

_CHROMA_CACHE = LRUCache(maxsize=int(os.getenv("CHROMA_CACHE_SIZE", "256")))

# ==================== Welcome menu / examples ====================
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
    "Tips: type 'menu' anytime to see this again, or '!fast <question>' for a quick reply."
)

# ==================== Helpers ====================
def safe_chat_completion(model, messages, temperature=None, max_tokens=None):
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens or MAX_TOKENS}
    if temperature is not None and not model.startswith("o1"):
        kwargs["temperature"] = temperature
    return openai_client.chat.completions.create(**kwargs)

def rough_keyword_match(q: str, doc: str, min_overlap: int = 2) -> bool:
    q_words = set(re.findall(r"[a-zA-Z]{4,}", (q or "").lower()))
    d_words = set(re.findall(r"[a-zA-Z]{4,}", (doc or "").lower()))
    return len(q_words & d_words) >= min_overlap

def summarize_history(sender: str, max_turns: int = 4, max_words: int = 40) -> str:
    """
    Produce a 1–2 sentence running summary of the conversation.
    - Pulls only the last `max_turns` exchanges to keep the prompt tiny.
    - Returns previous summary on any error (never blocks the reply).
    """
    try:
        last = CONV_HISTORY.get(sender, [])[-max_turns:]
        prior = CONV_SUMMARY.get(sender, "")

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
        if not new_summary:
            return prior
        CONV_SUMMARY[sender] = new_summary
        CONV_SUMMARY.touch(sender)
        if DEBUG:
            print("[summary]", new_summary)
        return new_summary

    except Exception as e:
        if DEBUG:
            print("summarize_history error:", e)
        return CONV_SUMMARY.get(sender, "")

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
            return

        prof = get_profile(sender)

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
            m = re.search(r"\{.*\}", raw, flags=re.S)
            data = json.loads(m.group(0)) if m else {}

        if not isinstance(data, dict):
            return

        def _clean_url(u: str) -> str | None:
            if not isinstance(u, str):
                return None
            u = u.strip()
            if not u:
                return None
            if not re.match(r"^https?://", u, flags=re.I):
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
        USER_PROFILE.touch(sender)

        if DEBUG:
            print("[profile]", json.dumps(prof, ensure_ascii=False))

    except Exception as e:
        if DEBUG:
            print("extract_profile_updates error:", e)
        return

def build_effective_query(user_msg: str, profile: dict, summary: str) -> str:
    parts = [user_msg]
    if profile.get("role"):
        parts.append(f"Target role: {profile['role']}")
    if profile.get("industry"):
        parts.append(f"Industry: {profile['industry']}")
    if summary:
        parts.append(f"Context: {summary}")
    return " | ".join(parts)

# === helpers for compact, fast RAG ===
AVG_TOKENS_PER_WORD = 1.3  # rough; fine for budgeting
def clip_words(txt: str, max_words: int) -> str:
    w = (txt or "").split()
    return " ".join(w[:max_words])

def _budgeted_context(docs: list[str]) -> str:
    total_token_budget = int(RETRIEVAL_BUDGET * MAX_TOKENS)
    total_word_budget  = max(200, int(total_token_budget / AVG_TOKENS_PER_WORD))
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

def _extract_answer_and_update_summary(sender: str, raw: str, max_words: int = 40) -> str:
    """
    Parse a single-call model response that contains both the 'answer' and a compact 'summary'.
    Updates CONV_SUMMARY[sender] inline; returns just the answer text.
    Accepts JSON {"answer": "...", "summary": "..."} or a fallback ANSWER/SUMMARY block.
    """
    ans, summ = None, None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            ans = obj.get("answer")
            summ = obj.get("summary")
    except Exception:
        pass

    if not ans:
        # Fallback to labeled blocks, robust to minor formatting
        m_ans = re.search(r"ANSWER:\s*(.+?)(?:\n\s*SUMMARY:|\Z)", raw, re.S | re.I)
        if m_ans:
            ans = m_ans.group(1).strip()
        else:
            ans = raw.strip()

    if not summ:
        m_sum = re.search(r"SUMMARY:\s*(.+)$", raw, re.S | re.I)
        if m_sum:
            summ = m_sum.group(1).strip()

    # Clip and store the summary inline (keeps personalization fresh with zero extra calls)
    if summ:
        summ = clip_words(summ, max_words)
        CONV_SUMMARY[sender] = summ
        if hasattr(CONV_SUMMARY, "touch"):
            CONV_SUMMARY.touch(sender)

    return ans

def rag_answer_from_posts(sender: str, question: str, docs: list[str]) -> str:
    try:
        context = _budgeted_context(docs)
        sys = (
            "You are Margil Gandhi’s assistant. Use ONLY the provided context."
            " Return STRICT JSON with keys: answer, summary.\n"
            " - 'answer': the concise reply for the user (<= {mx} tokens).\n"
            " - 'summary': <= 40 words updating the conversation state "
            "(target role/level/industry/location/constraints/current topic)."
        ).format(mx=MAX_TOKENS)
        user = f"Context:\n{context}\n\nQuestion: {question}\n\nReturn JSON only."
        chat = safe_chat_completion(
            CHAT_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2
        )
        raw = (chat.choices[0].message.content or "").strip()
        return _extract_answer_and_update_summary(sender, raw)
    except Exception as e:
        if DEBUG: print("rag_answer_from_posts error:", e)
        return "Sorry—I ran into an issue using the PDF context. Try asking again in a moment."

def best_practice_answer(sender: str, query: str) -> str:
    try:
        sys = (
            "You are a careers/job-search assistant. Return STRICT JSON with keys: answer, summary.\n"
            " - 'answer': direct, concise; numbered list with ONE item per line when listing.\n"
            " - 'summary': <= 40 words updating the conversation state (role/level/industry/location/constraints/topic)."
        )
        chat = safe_chat_completion(
            CHAT_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":query}],
            temperature=0.3
        )
        raw = (chat.choices[0].message.content or "").strip()
        return _extract_answer_and_update_summary(sender, raw)
    except Exception as e:
        if DEBUG: print("best_practice_answer error:", e)
        return "Sorry—I’m a bit busy right now. Please try again."

# ---- heuristics to control expensive work ----
def looks_generic(msg: str) -> bool:
    if not msg: return True
    m = msg.strip().lower()
    if len(m) < 20:
        return True
    # only treat "tips/advice/help" as generic when *very short*
    return any(w in m for w in ["help", "tips", "advice", "how to start"]) and len(m) < 80

def wants_rag(msg: str) -> bool:
    if not msg:
        return False
    m = msg.lower()
    if len(m) >= 120:
        return True
    return any(k in m for k in [
        "from my posts", "as i wrote", "from the pdf", "context above",
        "according to your post", "based on your guide"
    ])

def is_first_turn(sender: str) -> bool:
    return sender not in WELCOME_SEEN

def map_menu_choice_to_query(text: str, sender: str | None = None) -> str | None:
    t = (text or "").strip().lower()
    if t in ("menu", "help", "?"):
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

def _enqueue_deferred(sender: str, user_text: str):
    try:
        _DEFER_QUEUE.put_nowait((sender, user_text))
    except Exception:
        # If saturated, skip deferred work rather than affecting next turn latency
        pass

def _log_reply(sender, question, reply_text, from_pdf, confident,
               top_dist=None, top_source=None, retrieval_ms=None):
    log_interaction(sender, {
        "question": question,
        "from_pdf": from_pdf,           # True if RAG answer path used
        "confident": confident,
        "answer_len": len(reply_text),
        "top_dist": top_dist,           # float or None
        "top_source": top_source,       # e.g., PDF filename
        "retrieval_ms": retrieval_ms,   # int ms or None
    })

# ==================== Retrieval with cache ====================
def _query_chroma_cached(q_texts, n_results):
    def _compute():
        return collection.query(
            query_texts=q_texts,
            n_results=n_results,
            include=["documents", "distances", "metadatas"],  # ← important
        )
    key = (tuple(q_texts), n_results)
    return _CHROMA_CACHE.get_set(key, _compute)

def twiml_message(msg: str) -> Response:
    resp = MessagingResponse()
    resp.message(msg)
    return Response(str(resp), mimetype="application/xml", status=200)

@app.errorhandler(Exception)
def handle_any_error(e):
    app.logger.exception("Unhandled error in request")
    return twiml_message("Sorry—something went wrong. Send 'menu' to continue.")

# ==================== Routes ====================
@app.route("/", methods=["GET"])
def health():
    return "OK"

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    t0 = time.time()
    user_text = (request.form.get("Body") or "").strip()
    sender = (request.form.get("From") or "").strip()

    # Ultra-fast exits
    if user_text.lower() == "ping":
        return twiml_message("pong")

    # Menu/help handling (within window logic)
    choice = map_menu_choice_to_query(user_text, sender)
    if choice == "__MENU__" or (is_first_turn(sender) and not user_text):
        WELCOME_SEEN.add(sender)
        LAST_MENU_AT[sender] = time.time()
        remember_turn(sender, "assistant", "(menu shown)")
        return twiml_message(WELCOME_MENU)

    # Quick dev command that bypasses RAG
    if user_text.lower().startswith("!fast "):
        query = user_text.split(" ", 1)[1]
        txt = best_practice_answer(sender, query)
        return twiml_message(sanitize_plain(txt))

    # Normalize menu number selection (only valid shortly after menu)
    mapped = map_menu_choice_to_query(user_text, sender)
    is_menu_pick = (mapped is not None and mapped != "__MENU__")
    if is_menu_pick:
        user_text = mapped

    # --- ALWAYS record user turn + (optionally) refresh summary inline ---
    remember_turn(sender, "user", user_text)
    # Only summarize when it's NOT a quick menu pick and we have headroom
    if (not is_menu_pick) and (time.time() - t0 < 0.5):
        try:
            _ = summarize_history(sender, max_turns=3, max_words=40)
        except Exception:
            pass

    # EARLY EXIT BEFORE ANY LLM HEAVY WORK
    if looks_generic(user_text):
        txt = best_practice_answer(sender, user_text)
        txt = enforce_numbered_lines(txt)
        reply = sanitize_plain(txt)
        remember_turn(sender, "assistant", reply)
        _log_reply(sender, user_text, reply, from_pdf=False, confident=False)
        _enqueue_deferred(sender, user_text)  # defer profile extraction only
        return twiml_message(reply)

    # From here on, we can afford a bit more work
    elapsed = lambda: time.time() - t0

    # ---- Cheap context (make sure these exist) ----
    try:
        profile = get_profile(sender)
    except Exception:
        profile = {}
    # summary was refreshed earlier; read whatever we have
    summary = CONV_SUMMARY.get(sender, "")

    # B) maybe-RAG → quick retrieval, then decide
    confident = False
    docs = []
    top = None
    top_src = None
    retrieval_ms = None  # duration in milliseconds

    if elapsed() < TURN_BUDGET * 0.4:
        try:
            tR = time.time()
            res = _query_chroma_cached([user_text], K)  # includes metadatas
            retrieval_ms = int((time.time() - tR) * 1000)

            dists = (res.get("distances", [[]]) or [[]])[0]
            docs  = (res.get("documents", [[]]) or [[]])[0]
            metas = (res.get("metadatas", [[]]) or [[]])[0]

            top = dists[0] if dists else None
            top_src = (metas[0].get("source") if metas and isinstance(metas[0], dict) else None)

            app.logger.info(
                "[rag] top_dist=%s src=%s k=%d time_ms=%s",
                (round(top, 3) if top is not None else None),
                top_src, K, retrieval_ms
            )

            # confidence gate (smaller distance is better for cosine)
            confident = (
                top is not None and
                top <= CONF_DIST and
                bool(docs) and
                rough_keyword_match(user_text, docs[0])
            )

            # latency guard
            if retrieval_ms is not None and (retrieval_ms / 1000.0) > RETRIEVAL_TIME_BUDGET:
                confident = False

        except Exception:
            app.logger.exception("retrieval error")
            confident = False
            docs = []
            top = None
            top_src = None
            retrieval_ms = None

    # C) generate (one chat max)
    eq = build_effective_query(user_text, profile, summary)

    if confident and docs and elapsed() < TURN_BUDGET * 0.7:
        # ---------- RAG path ----------
        txt = rag_answer_from_posts(sender, eq, docs)
        if not txt or "not in the context" in (txt or "").lower():
            txt = best_practice_answer(sender, user_text)

        txt = enforce_numbered_lines(txt)
        reply = sanitize_plain(txt)

        if DEBUG:
            reply += (
                f"\n\n[diag] rag=yes"
                f"{' top=' + str(round(top,3)) if top is not None else ''}"
                f"{' src=' + top_src if top_src else ''}"
            )

        remember_turn(sender, "assistant", reply)
        _log_reply(
            sender, user_text, reply,
            from_pdf=True, confident=True,
            top_dist=top, top_source=top_src, retrieval_ms=retrieval_ms
        )
        _enqueue_deferred(sender, user_text)
        return twiml_message(reply)

    else:
        # ---------- Fallback path (no RAG used) ----------
        txt = best_practice_answer(sender, user_text)
        txt = enforce_numbered_lines(txt)
        reply = sanitize_plain(txt)

        if DEBUG:
            # even in fallback, include top/src if we have them
            reply += (
                "\n\n[diag] rag=no"
                f"{' top=' + str(round(top,3)) if top is not None else ''}"
                f"{' src=' + top_src if top_src else ''}"
            )

        remember_turn(sender, "assistant", reply)
        _log_reply(
            sender, user_text, reply,
            from_pdf=False, confident=False,
            top_dist=None, top_source=None, retrieval_ms=None
        )
        _enqueue_deferred(sender, user_text)
        return twiml_message(reply)

@app.route("/stats", methods=["GET"])
def stats():
    token = request.args.get("token", "")
    if not ADMIN_TOKEN:
        return "Admin token not configured.", 400
    if token != ADMIN_TOKEN:
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

# ==================== Run ====================
if __name__ == "__main__":
    print("✅ Flask is starting...")
    try:
        count = collection.count() if hasattr(collection, "count") else "n/a"
        print(f"Chroma collection '{COLLECTION}' count:", count)
        if DEBUG:
            # Quick note to validate distance semantics empirically if needed
            print("Note: CONF_DIST assumes cosine distance (smaller is better).")
    except Exception as e:
        print("Could not count Chroma docs:", e)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=DEBUG)
