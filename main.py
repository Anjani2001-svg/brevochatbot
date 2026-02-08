# main.py  (FastAPI backend — copy/paste ready)
# Fixes:
# 1) "send some IT courses" triggers recommendation mode
# 2) Hard subject filter (prevents Animal Care when user asked for IT)
# 3) Basic cleanup + safe .env loading

import os
import re
import json
import threading
import traceback
from datetime import datetime
from urllib.parse import urlparse

import requests
import pandas as pd
import pytz
from bs4 import BeautifulSoup
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, HTMLResponse

from db_setup import build_db
from chatbot_core import set_db, generate_reply


# ---------------- ENV ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from same folder as this file (recommended)
load_dotenv(os.path.join(BASE_DIR, ".env"))


# ---------------- APP ----------------
app = FastAPI()

# ---------------- Memory (RAM + Disk) ----------------
CHAT_MEMORY: dict[str, list[dict[str, str]]] = {}
MAX_TURNS = 20

COURSE_STATE: dict[str, dict[str, str]] = {}  # convo_id -> {"name":..., "url":...}

MEMORY_FILE = os.path.join(BASE_DIR, "chat_memory_store.json")
MEMORY_LOCK = threading.Lock()


def remember_course(convo_id: str, name: str, url: str) -> None:
    if convo_id and (name or url):
        COURSE_STATE[convo_id] = {"name": name or "", "url": url or ""}


def get_remembered_course(convo_id: str) -> dict[str, str]:
    return COURSE_STATE.get(convo_id, {}) or {}


def load_persistent_memory() -> None:
    global CHAT_MEMORY, COURSE_STATE
    if not os.path.exists(MEMORY_FILE):
        return
    try:
        with MEMORY_LOCK:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        CHAT_MEMORY = data.get("chat_memory", {}) or {}
        COURSE_STATE = data.get("course_state", {}) or {}
        print(f"✅ Loaded persistent memory: {len(CHAT_MEMORY)} conversations")
    except Exception as e:
        print("⚠️ Failed to load persistent memory:", repr(e))


def save_persistent_memory() -> None:
    try:
        with MEMORY_LOCK:
            data = {"chat_memory": CHAT_MEMORY, "course_state": COURSE_STATE}
            tmp = MEMORY_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, MEMORY_FILE)
    except Exception as e:
        print("⚠️ Failed to save persistent memory:", repr(e))


# ---------------- Weekend rule (UK time) ----------------
UK_TZ = pytz.timezone("Europe/London")


def is_weekend_now() -> bool:
    return datetime.now(UK_TZ).weekday() >= 5


# ---------------- Website fetch ----------------
ALLOWED_DOMAINS = {"southlondoncollege.org", "www.southlondoncollege.org"}
PAGE_CACHE = TTLCache(maxsize=256, ttl=1800)  # 30 min


def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if u.endswith("/"):
        u = u[:-1]
    return u


def safe_fetch_page_text(url: str) -> str:
    if not url:
        return ""
    url = normalize_url(url)
    host = urlparse(url).netloc.lower()
    if host not in ALLOWED_DOMAINS:
        return ""

    if url in PAGE_CACHE:
        return PAGE_CACHE[url]

    r = requests.get(url, timeout=15, headers={"User-Agent": "SLCBot/1.0"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = text[:9000]
    PAGE_CACHE[url] = text
    return text


def extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s)]+", text)
    return [normalize_url(u) for u in urls]


def extract_course_facts(page_text: str) -> dict:
    facts = {}

    m = re.search(r"Current price is:\s*£\s*([0-9,]+(?:\.[0-9]{2})?)", page_text, flags=re.I)
    if m:
        facts["current_price"] = f"£{m.group(1)}"

    m = re.search(r"Original price was:\s*£\s*([0-9,]+(?:\.[0-9]{2})?)", page_text, flags=re.I)
    if m:
        facts["original_price"] = f"£{m.group(1)}"

    m = re.search(r"\b(\d{1,3})\s*credits\b", page_text, flags=re.I)
    if m:
        facts["credits"] = m.group(1)

    m = re.search(r"\b(\d+\s*-\s*\d+\s*months)\b", page_text, flags=re.I)
    if m:
        facts["duration"] = m.group(1)

    m = re.search(r"\b(\d+\s+year(?:s)?)\b", page_text, flags=re.I)
    if m and not facts.get("duration"):
        facts["duration"] = m.group(1)

    m = re.search(r"Standard Plan.*?(\d+\s*-\s*\d+\s*Months)", page_text, flags=re.I)
    if m:
        facts["standard_schedule"] = m.group(1).strip()

    m = re.search(r"Fast-track Plan.*?(\d+\s*Months)", page_text, flags=re.I)
    if m:
        facts["fast_schedule"] = m.group(1).strip()

    return facts


def fetch_course_page_context(url: str) -> tuple[dict, str]:
    if not url:
        return {}, ""
    page_text = safe_fetch_page_text(url)
    facts = extract_course_facts(page_text)
    web_context = (
        f"URL: {url}\n"
        f"Extracted facts: current_price={facts.get('current_price','not found')}, "
        f"duration={facts.get('duration','not found')}, "
        f"credits={facts.get('credits','not found')}\n\n"
        f"PAGE TEXT:\n{page_text}"
    ).strip()
    return facts, web_context


# ---------------- Price update sheet ----------------
PRICE_FILE = os.path.join(BASE_DIR, "SLC Full Site Price Update.xlsx")
PRICE_SHEET_NAME = os.getenv("PRICE_SHEET_NAME", "").strip() or None


def load_price_map() -> dict:
    """
    price_map[url] = {"standard": "...", "fast": "...", "instalment": "..."}
    also fallback: price_map[course_name_lower] = {...}
    """
    if not os.path.exists(PRICE_FILE):
        print(f"⚠️ Price file not found: {PRICE_FILE} (will use tracker prices)")
        return {}

    try:
        df = pd.read_excel(PRICE_FILE, sheet_name=PRICE_SHEET_NAME or 0)
    except Exception as e:
        print("❌ Failed reading price file:", PRICE_FILE, repr(e))
        return {}

    df = df.fillna("")
    df.columns = df.columns.str.strip()

    def pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    url_col = pick_col(["Course URL", "URL", "Link"])
    name_col = pick_col(["Course Name", "Name", "Course"])

    std_col = pick_col(["Standard Sale Price", "Standard Price", "Standard"])
    fast_col = pick_col(["Fast Track Sale Price", "Fast Track Price", "Fast Track"])
    inst_col = pick_col(["Instalment Price", "Installment Price", "Instalments", "Monthly Price"])

    if not (url_col or name_col):
        print("⚠️ Price sheet must contain Course URL or Course Name.")
        print("Columns:", list(df.columns))
        return {}

    price_map = {}
    count = 0

    for _, r in df.iterrows():
        url = normalize_url(str(r.get(url_col, "")).strip()) if url_col else ""
        name = str(r.get(name_col, "")).strip() if name_col else ""
        name_key = name.lower().strip() if name else ""

        rec = {
            "standard": str(r.get(std_col, "")).strip() if std_col else "",
            "fast": str(r.get(fast_col, "")).strip() if fast_col else "",
            "instalment": str(r.get(inst_col, "")).strip() if inst_col else "",
        }

        if not (rec["standard"] or rec["fast"] or rec["instalment"]):
            continue

        if url:
            price_map[url] = rec
            count += 1

        if name_key and name_key not in price_map:
            price_map[name_key] = rec

    print(f"✅ Loaded price updates: {count} URL rows (+ name fallbacks)")
    return price_map


# ---------------- Intent rules (FIXED) ----------------
def is_recommend_intent(text: str) -> bool:
    """
    FIX: also catches "send some IT courses" / "show me IT courses" / "list IT courses"
    """
    t = (text or "").lower()

    has_course_word = any(w in t for w in ["course", "courses", "programme", "program", "qualification"])

    browse_phrases = [
        "send some",
        "show me",
        "list",
        "give me",
        "recommend",
        "suggest",
        "which course",
        "what course",
        "best course",
        "suitable course",
        "courses for",
    ]
    has_browse_phrase = any(p in t for p in browse_phrases)

    return has_course_word and has_browse_phrase


def should_use_website(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "price",
        "cost",
        "fee",
        "duration",
        "how long",
        "entry requirements",
        "requirements",
        "modules",
        "units",
        "syllabus",
        "awarding body",
        "credits",
        "assessment",
        "what will i learn",
        "course details",
        "start date",
        "intake",
    ]
    return any(k in t for k in keywords)


def is_quality_intent(text: str) -> bool:
    t = (text or "").lower()
    phrases = [
        "is this course good",
        "is it good",
        "is this good",
        "is it worth it",
        "worth it",
        "should i do",
        "should i take",
        "should i enroll",
        "should i enrol",
        "do you recommend",
        "recommend this",
        "good course",
        "is this right for me",
        "is it right for me",
    ]
    return any(p in t for p in phrases)


def is_basic_question(text: str) -> bool:
    t = (text or "").lower()
    return any(
        k in t
        for k in [
            "accredited",
            "accreditation",
            "certificate",
            "online",
            "study online",
            "distance learning",
            "enrol",
            "enroll",
            "apply",
            "entry requirements",
            "requirements",
            "modules",
            "units",
            "assessment",
            "assignments",
            "exams",
            "what will i learn",
            "who is this for",
            "who is it for",
        ]
    )


def requested_fields(text: str) -> set[str]:
    t = (text or "").lower()
    fields = set()

    if any(k in t for k in ["price", "cost", "fee", "fees", "tuition"]):
        fields.add("price")
    if any(k in t for k in ["duration", "how long", "length"]):
        fields.add("duration")
    if "credit" in t:
        fields.add("credits")
    if any(k in t for k in ["awarding body", "awarding"]):
        fields.add("awarding")
    if any(k in t for k in ["level", "rqf"]):
        fields.add("level")

    if "only" in t:
        if "price" in t:
            return {"price"}
        if "duration" in t or "how long" in t:
            return {"duration"}
        if "credit" in t:
            return {"credits"}

    return fields


def is_followup_without_course(text: str) -> bool:
    t = (text or "").lower()

    if "http://" in t or "https://" in t:
        return False

    if re.search(r"\blevel\s*\d\b", t):
        return False
    if any(w in t for w in ["diploma", "certificate", "award"]):
        return False

    if len(t.strip()) <= 60 and any(
        w in t for w in ["price", "cost", "fee", "duration", "how long", "credits", "only", "it", "this", "worth", "good"]
    ):
        return True

    return False


def resolve_course_for_message(convo_id: str, text: str) -> str:
    urls = extract_urls_from_text(text)
    if urls:
        return urls[0]

    remembered = get_remembered_course(convo_id)
    if remembered.get("url") and (is_followup_without_course(text) or is_quality_intent(text) or is_basic_question(text)):
        return remembered["url"]

    return ""


# ---------------- Course data helpers ----------------
SUBJECT_KEYWORDS = {
    "accounting": ["account", "accounting", "bookkeep", "bookkeeping", "finance", "payroll", "sage", "tax"],
    "childcare": ["childcare", "early years", "eyfs", "nursery", "teaching assistant", "sen", "safeguarding"],
    "it": ["it", "cyber", "security", "network", "programming", "python", "data", "ai", "cloud", "software", "computing", "sql", "microsoft", "excel", "office", "coding"],
    "business": ["business", "management", "leadership", "hr", "marketing", "project", "operations"],
    "health_social_care": ["health", "social care", "care", "adult care", "nursing", "mental health"],
    "education": ["education", "teaching", "teacher", "training", "assessor", "iqa", "quality assurance", "pta", "ta"],
}

KNOWN_AWARDING_BODIES = ["ATHE", "OTHM", "NCFE", "QLS", "Pearson", "City & Guilds", "CACHE", "AQA", "ILM", "CIM", "CIPD"]


def _extract_goal_subject(text: str) -> str | None:
    t = (text or "").lower()
    best = None
    best_hits = 0
    for subject, kws in SUBJECT_KEYWORDS.items():
        hits = sum(1 for k in kws if k in t)
        if hits > best_hits:
            best_hits = hits
            best = subject
    return best if best_hits > 0 else None


def extract_user_preferences(text: str) -> dict:
    return {"goal_subject": _extract_goal_subject(text)}


def get_field(meta: dict, *names: str) -> str:
    meta = meta or {}
    key_map = {}
    for k in meta.keys():
        if isinstance(k, str):
            key_map[k.strip().lower()] = k

    for n in names:
        if n in meta:
            v = meta.get(n)
            s = "" if v is None else str(v).strip()
            if s and s.lower() not in {"n/a", "na", "nan"}:
                return s

        if isinstance(n, str):
            k2 = key_map.get(n.strip().lower())
            if k2 is not None:
                v = meta.get(k2)
                s = "" if v is None else str(v).strip()
                if s and s.lower() not in {"n/a", "na", "nan"}:
                    return s
    return ""


def faiss_top_courses(db, query: str, k: int = 8):
    return db.similarity_search_with_score(query, k=k)


def best_course_hit(db, query: str):
    try:
        hits = faiss_top_courses(db, query, k=10)
        for doc, score in hits:
            meta = getattr(doc, "metadata", {}) or {}
            url = normalize_url(get_field(meta, "Course URL", "course_url", "URL", "Link"))
            name = get_field(meta, "Course Name", "Course Title", "Name", "Course")
            if url or name:
                return doc, score
    except Exception as e:
        print("⚠️ Search lookup failed:", repr(e))
    return None, None


def merge_course_record(meta: dict, price_map: dict, url: str, name: str, page_facts: dict | None = None) -> dict:
    meta = meta or {}
    page_facts = page_facts or {}

    base_price = get_field(meta, "Price", "Base Price", "Course Price", "Standard Price")
    standard_price = get_field(meta, "Standard Sale Price", "Standard Price", "Standard")
    fast_price = get_field(meta, "Fast Track Sale Price", "Fast Track Price", "Fast Track")
    instalment_price = get_field(meta, "Instalment Price", "Installment Price", "Instalments", "Monthly Price")

    duration = get_field(meta, "Duration", "Course Duration", "Standard Duration", "Standard Duration ")
    credits = get_field(meta, "Credits", "Credit Value", "Credit value", "Total Credits")
    level = get_field(meta, "Qualification Level", "Level")
    awarding = get_field(meta, "Awarding Body", "Awarding body")

    # sheet override
    p = price_map.get(url) or price_map.get((name or "").lower().strip())
    if p:
        standard_price = p.get("standard") or standard_price
        fast_price = p.get("fast") or fast_price
        instalment_price = p.get("instalment") or instalment_price

    # website override
    current_price = page_facts.get("current_price", "")
    original_price = page_facts.get("original_price", "")
    credits = page_facts.get("credits") or credits
    duration = page_facts.get("duration") or duration

    return {
        "name": name,
        "url": url,
        "duration": duration,
        "credits": credits,
        "level": level,
        "awarding": awarding,
        "base_price": base_price,
        "standard_price": standard_price,
        "fast_price": fast_price,
        "instalment_price": instalment_price,
        "current_price": current_price,
        "original_price": original_price,
    }


def format_reco_shortlist(courses: list[dict]) -> str:
    lines = []
    for c in courses:
        lines.append(
            f"- {c.get('name','')}\n"
            f"  URL: {c.get('url','')}\n"
            f"  Level: {c.get('level','') or 'not listed'}\n"
            f"  Awarding: {c.get('awarding','') or 'not listed'}\n"
            f"  Duration: {c.get('duration','') or 'not listed'}\n"
            f"  Prices: current={c.get('current_price','') or 'n/a'}, "
            f"standard={c.get('standard_price','') or 'n/a'}, "
            f"fast={c.get('fast_price','') or 'n/a'}, "
            f"instalment={c.get('instalment_price','') or 'n/a'}"
        )
    return "\n\n".join(lines).strip()


# ---------------- Startup / Shutdown ----------------
@app.on_event("startup")
def startup():
    load_persistent_memory()

    try:
        db = build_db()
        set_db(db)
        app.state.course_db = db
        print("✅ Course tracker DB loaded")
    except Exception as e:
        print("❌ Failed to load course tracker DB:", repr(e))
        set_db(None)
        app.state.course_db = None

    app.state.price_map = load_price_map()
    print("✅ Application startup complete")


@app.on_event("shutdown")
def shutdown():
    save_persistent_memory()
    print("✅ Saved persistent memory on shutdown")


# ---------------- Local test chat (UI -> Streamlit uses this) ----------------
@app.post("/test-chat")
async def test_chat(payload: dict):
    text = (payload.get("text") or "").strip()
    convo_id = (payload.get("convo_id") or "").strip() or f"convo-{datetime.utcnow().timestamp()}"

    if not text:
        return {"reply": "", "convo_id": convo_id}

    history = CHAT_MEMORY.setdefault(convo_id, [])
    history.append({"role": "user", "content": text})

    # simple conversation context
    recent = history[-(MAX_TURNS * 2) :]
    conversation_context = "Conversation so far:\n" + "\n".join(
        [("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in recent]
    )

    db = getattr(app.state, "course_db", None)
    price_map = getattr(app.state, "price_map", {}) or {}

    try:
        # --------- RECOMMENDATION MODE (FIXED) ---------
        if is_recommend_intent(text):
            if db is None:
                reply = "Course tracker database is not loaded. Please check your tracker Excel file and restart the server."
            else:
                prefs = extract_user_preferences(text)
                goal = (prefs.get("goal_subject") or "").strip().lower()

                hits = faiss_top_courses(db, text, k=50)
                candidates: list[dict] = []
                seen = set()

                for doc, faiss_score in hits:
                    meta = getattr(doc, "metadata", {}) or {}
                    name = get_field(meta, "Course Name", "Course Title", "Name", "Course")
                    url = normalize_url(get_field(meta, "Course URL", "course_url", "URL", "Link"))
                    if not url or url in seen:
                        continue

                    # ✅ HARD SUBJECT FILTER (prevents Animal Care when asked for IT)
                    if goal and goal in SUBJECT_KEYWORDS:
                        kws = SUBJECT_KEYWORDS[goal]
                        combined = f"{name} {url}".lower()
                        if not any(k in combined for k in kws):
                            continue

                    seen.add(url)
                    merged = merge_course_record(meta, price_map, url, name, page_facts=None)
                    candidates.append(merged)

                candidates = candidates[:5]

                if not candidates:
                    reply = "I couldn’t find matching courses. What IT area do you want (Cyber Security, Data, Programming, Networking, Microsoft Office)?"
                else:
                    shortlist_text = format_reco_shortlist(candidates)
                    extra = f"{conversation_context}\n\nSHORTLIST:\n{shortlist_text}"

                    user_for_llm = (
                        f"{text}\n\n"
                        "You are a friendly South London College course advisor.\n"
                        "Recommend 3–5 suitable courses from the SHORTLIST only.\n"
                        "Use bullet points only. One course per bullet.\n"
                        "For each bullet: Course name + URL + level + duration + awarding body + price if present.\n"
                        "If a detail is missing, say 'not listed here'."
                    )

                    reply = generate_reply(
                        user_text=user_for_llm,
                        extra_context=extra,
                        retrieval_query=text,
                        use_knowledge=False,
                    )

        # --------- NORMAL MODE (single course Q&A) ---------
        else:
            if db is None:
                reply = "Course tracker database is not loaded. Please check your tracker Excel file and restart the server."
            else:
                course_ref = resolve_course_for_message(convo_id, text)
                lookup_query = course_ref or text

                doc, score = best_course_hit(db, lookup_query)
                if doc is None:
                    reply = "Which course do you mean? Please type the course name or paste the course link."
                else:
                    meta = getattr(doc, "metadata", {}) or {}
                    name = get_field(meta, "Course Name", "Course Title", "Name", "Course")
                    url = normalize_url(get_field(meta, "Course URL", "course_url", "URL", "Link"))

                    remember_course(convo_id, name, url)

                    # Optional website fetch for details
                    web_context = ""
                    page_facts = {}
                    if url and (should_use_website(text) or is_quality_intent(text) or is_basic_question(text)):
                        try:
                            page_facts, web_context = fetch_course_page_context(url)
                        except Exception as e:
                            print("⚠️ website fetch failed:", url, repr(e))

                    merged = merge_course_record(meta, price_map, url, name, page_facts)

                    merged_context = (
                        "COURSE FACTS:\n"
                        f"Name: {merged.get('name','')}\n"
                        f"URL: {merged.get('url','')}\n"
                        f"Level: {merged.get('level','not listed')}\n"
                        f"Awarding body: {merged.get('awarding','not listed')}\n"
                        f"Duration: {merged.get('duration','not listed')}\n"
                        f"Price: {merged.get('current_price') or merged.get('standard_price') or merged.get('base_price') or 'not listed'}\n"
                    )

                    extra = f"{conversation_context}\n\n{merged_context}"
                    if web_context:
                        extra += f"\n\n====================\n\n{web_context}"

                    reply = generate_reply(
                        user_text=f"{text}\n\nUse ONLY the provided context. If missing, say 'not listed here'.",
                        extra_context=extra,
                        retrieval_query=text,
                        use_knowledge=False,
                    )

    except Exception as e:
        traceback.print_exc()
        reply = f"Server error: {repr(e)}"

    reply = (reply or "").strip() or "Thanks — I didn’t catch that. Could you rephrase?"

    history.append({"role": "assistant", "content": reply})
    if len(history) > MAX_TURNS * 2:
        CHAT_MEMORY[convo_id] = history[-(MAX_TURNS * 2) :]

    save_persistent_memory()
    return {"reply": reply, "convo_id": convo_id}


# ---------------- Simple local web UI (optional) ----------------
@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Local Bot Chat</title>
</head>
<body>
  <h3>Local Bot Chat</h3>
  <p>Open this page at: <b>http://127.0.0.1:8000/chat</b></p>
  <div id="box" style="border:1px solid #ccc; padding:12px; height:420px; overflow-y:auto;"></div>
  <div style="margin-top:10px; display:flex; gap:10px;">
    <input id="msg" placeholder="Type a message..." style="width:70%; padding:10px;" />
    <button onclick="send()">Send</button>
  </div>

<script>
function escapeHtml(s){
  return s.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}
function getConvoId(){
  let id = localStorage.getItem("convo_id");
  if(!id){
    id = "convo-" + crypto.randomUUID();
    localStorage.setItem("convo_id", id);
  }
  return id;
}
async function send(){
  const input = document.getElementById("msg");
  const text = input.value.trim();
  if(!text) return;

  const box = document.getElementById("box");
  box.innerHTML += `<div><b>You:</b> ${escapeHtml(text)}</div>`;
  input.value = "";

  const convo_id = getConvoId();
  const res = await fetch("/test-chat", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({text, convo_id})
  });
  const data = await res.json();
  box.innerHTML += `<div><b>Bot:</b> ${escapeHtml((data.reply||"").toString())}</div>`;
  box.scrollTop = box.scrollHeight;
}
document.getElementById("msg").addEventListener("keydown", (e)=>{ if(e.key==="Enter") send(); });
</script>
</body>
</html>
"""


# ---------------- Health ----------------
@app.get("/healthcheck")
def healthcheck():
    return {"ok": True}


@app.get("/")
def root():
    return {"status": "ok"}
