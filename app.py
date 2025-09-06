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
from functools import lru_cache

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

# -------------------- Deferred worker (post-reply) --------------------
_DEFER_QUEUE: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=1000)

def _defer_worker():
    # Runs OUTSIDE the Twilio webhook response path
    while True:
        try:
            sender, user_text = _DEFER_QUEUE.get()
            if not sender:
                continue
            # Heuristics: only run one of these per deferred cycle
            if _looks_like_profile_update(user_text):
                try:
                    extract_profile_updates(sender, user_text)
                except Exception as e:
                    print("defer extract_profile_updates error:", e)
            else:
                # Summarize every ~3 user turns if we have some history
                if len(CONV_HISTORY.get(sender, [])) >= 3:
                    try:
                        summarize_history(sender)
                    except Exception as e:
                        print("defer summarize_history error:", e)
        except Exception as e:
            print("defer worker error:", e)
        finally:
            try:
                _DEFER_QUEUE.task_done()
            except Exception:
                pass

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

# Retrieval / budget tuning
K = int(os.getenv("K", "3"))
CONF_DIST = float(os.getenv("CONF_DIST", "0.35"))   # cosine distance; lower=closer
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
RETRIEVAL_BUDGET = float(os.getenv("RETRIEVAL_BUDGET", "1.2"))
TURN_BUDGET = float(os.getenv("TURN_BUDGET", "10.0"))  # leave ~5s headroom for network/Twilio

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
SENDER_TURN_COUNT = {}
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
    "1) LinkedIn filters (find live opportunities)\n"
    "2) Subject lines (recruiter outreach)\n"
    "3) Cold email structure\n"
    "4) Phases of a job search\n"
    "5) Referral ask\n"
    "6) STAR interview stories\n"
    "7) Stay consistent / avoid burnout\n\n"
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

def cosine_sim(a, b) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(y*y for y in b))
    denom = na * nb
    return dot / denom if denom > 1e-9 else 0.0

def embed_text(text: str):
    emb = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return emb.data[0].embedding

def rough_keyword_match(q: str, doc: str, min_overlap: int = 2) -> bool:
    q_words = set(re.findall(r"[a-zA-Z]{4,}", (q or "").lower()))
    d_words = set(re.findall(r"[a-zA-Z]{4,}", (doc or "").lower()))
    return len(q_words & d_words) >= min_overlap

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
    try:
        chat = safe_chat_completion(
            CHAT_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": content}],
            temperature=0.2
        )
        new_summary = chat.choices[0].message.content.strip()
        CONV_SUMMARY[sender] = new_summary
        return new_summary
    except Exception as e:
        print("summarize_history error:", e)
        return prior or ""

def extract_profile_updates(sender: str, user_text: str) -> None:
    sys = (
        "Extract a careers profile as strict JSON (no code block). "
        "Allowed keys: role, level, location, years, industry, goals, links."
    )
    try:
        chat = safe_chat_completion(
            CHAT_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user_text}]
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
                    if any(t in text_l for t in ["role", "target", "position", "title", "aiming", "interested in"]):
                        prof["role"] = v
                else:
                    prof[k] = v
            USER_PROFILE[sender] = prof
        except Exception:
            pass
    except Exception as e:
        print("extract_profile_updates error:", e)

PROFILE_HINT_RE = re.compile(
    r"\b(role|position|title|senior|junior|mid|years|experience|exp|location|remote|hybrid|onsite|industry|domain|target|aiming|interested|looking)\b|https?://",
    re.I
)
def _looks_like_profile_update(text: str) -> bool:
    return bool(PROFILE_HINT_RE.search(text or ""))

def build_effective_query(user_msg: str, profile: dict, summary: str) -> str:
    parts = [user_msg]
    if profile.get("role"): parts.append(f"Target role: {profile['role']}")
    if profile.get("level"): parts.append(f"Level: {profile['level']}")
    if profile.get("industry"): parts.append(f"Industry: {profile['industry']}")
    if profile.get("location"): parts.append(f"Location: {profile['location']}")
    if summary: parts.append(f"Context: {summary}")
    return " | ".join(parts)

@lru_cache(maxsize=256)
def best_practice_answer_cached(query: str) -> str:
    return best_practice_answer(query)

def rag_answer_from_posts(question: str, docs: list[str]) -> str:
    context = "\n\n".join(docs)
    sys = (
        "You are Margil Gandhi’s assistant. Answer ONLY using the context below. "
        "If the answer isn't in the context, say you don't have that information. "
        "Use plain text only."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        chat = safe_chat_completion(
            CHAT_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=MAX_TOKENS
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        print("rag_answer_from_posts error:", e)
        return ""

def best_practice_answer(query: str) -> str:
    sys = (
        "You are a careers/job-search assistant. Provide a concise, practical answer. "
        "Use plain text only. Keep it under 120 words. "
        "Do not assume a specific role, level, or industry unless the user states it explicitly in THIS message."
    )
    try:
        chat = safe_chat_completion(
            CHAT_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": query}],
            temperature=0.3,
            max_tokens=MAX_TOKENS
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        print("best_practice_answer error:", e)
        return "Here’s a quick take: clarify your goal, focus on what you can control this week, and take one small step today. Share role/level if you want a more tailored answer."

# ---- heuristics to control expensive work ----
def looks_generic(msg: str) -> bool:
    if not msg: return True
    m = msg.strip().lower()
    if len(m) < 80: return True
    return any(w in m for w in ["help", "tips", "advice", "how to start"]) and len(m) < 140

def wants_rag(msg: str) -> bool:
    if not msg: return False
    m = msg.lower()
    if len(m) >= 120: return True
    return any(k in m for k in ["from my posts", "as i wrote", "from the pdf", "context above"])

def is_first_turn(sender: str) -> bool:
    return sender not in WELCOME_SEEN

def map_menu_choice_to_query(text: str) -> str | None:
    t = (text or "").strip().lower()
    if t == "menu": return "__MENU__"
    m = re.match(r'^\s*(\d+)\s*\)?\s*$', t)
    if m:
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
    choice = map_menu_choice_to_query(user_text)
    if choice == "__MENU__" or (is_first_turn(sender) and not user_text):
        WELCOME_SEEN.add(sender); remember_turn(sender, "assistant", "(menu shown)")
        resp = MessagingResponse(); resp.message(WELCOME_MENU); return str(resp)
    if user_text.lower().startswith("!fast "):
        query = user_text.split(" ", 1)[1]
        txt = best_practice_answer(query)
        resp = MessagingResponse(); resp.message(sanitize_plain(txt)); return str(resp)

    # normalize menu number selection
    if choice is not None and choice != "__MENU__":
        user_text = choice

    remember_turn(sender, "user", user_text)
    SENDER_TURN_COUNT[sender] = SENDER_TURN_COUNT.get(sender, 0) + 1

    elapsed = lambda: time.time() - t0
    profile = get_profile(sender)
    summary = CONV_SUMMARY.get(sender, "")

    # A) short/generic → single chat
    if looks_generic(user_text) or elapsed() > 0.5:
        txt = best_practice_answer(user_text)
        reply = sanitize_plain(txt)
        remember_turn(sender, "assistant", reply)
        _log_reply(sender, user_text, reply, from_pdf=False, confident=False)
        _enqueue_deferred(sender, user_text)  # AFTER reply is decided
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
            docs = res.get("documents", [[]])[0]
            top = dists[0] if dists else None
            confident = (top is not None and top <= CONF_DIST) and (not docs or rough_keyword_match(user_text, docs[0]))
            if tr > RETRIEVAL_BUDGET:
                confident = False
        except Exception as e:
            print("retrieval error:", e)
            confident = False
            docs = []

    # C) generate (one chat max)
    if confident and docs and elapsed() < TURN_BUDGET * 0.7:
        txt = rag_answer_from_posts(build_effective_query(user_text, profile, summary), docs)
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
