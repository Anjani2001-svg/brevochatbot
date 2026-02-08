import os
import json
import uuid
from pathlib import Path

import requests
import streamlit as st


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

# Backend URL (set this in terminal env or Streamlit secrets if you want)
# Example: http://127.0.0.1:8000/test-chat
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/test-chat")

# Local file just for UI chat history persistence (optional)
STORE_FILE = BASE_DIR / "streamlit_chat_store.json"


# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="SLC Bot", page_icon="🎓", layout="centered")
st.title("🎓 SLC Course Advisor")


# =========================
# STORE (optional)
# =========================
def load_store() -> dict:
    if not STORE_FILE.exists():
        return {}
    try:
        return json.loads(STORE_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def save_store(store: dict) -> None:
    tmp = STORE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STORE_FILE)

store = load_store()


# =========================
# CONVERSATION ID (stable via URL)
# =========================
try:
    qp = st.query_params
    cid = qp.get("cid")
    if isinstance(cid, list):
        cid = cid[0] if cid else None
except Exception:
    cid = st.experimental_get_query_params().get("cid", [None])[0]

if not cid:
    cid = f"st-{uuid.uuid4()}"
    try:
        st.query_params["cid"] = cid
    except Exception:
        st.experimental_set_query_params(cid=cid)

st.session_state.convo_id = cid


# Load messages for this conversation
if "messages" not in st.session_state:
    st.session_state.messages = store.get(cid, [])


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.subheader("Settings")
    st.caption("Backend endpoint")
    st.code(BACKEND_URL)

    if st.button("🆕 New conversation"):
        new_cid = f"st-{uuid.uuid4()}"
        try:
            st.query_params["cid"] = new_cid
        except Exception:
            st.experimental_set_query_params(cid=new_cid)

        st.session_state.convo_id = new_cid
        st.session_state.messages = []
        store[new_cid] = []
        save_store(store)
        st.rerun()

    if st.button("🩺 Healthcheck"):
        try:
            health_url = BACKEND_URL.replace("/test-chat", "/healthcheck")
            r = requests.get(health_url, timeout=10)
            if r.status_code == 200:
                st.success(r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
            else:
                st.error(f"Healthcheck failed ({r.status_code}): {r.text}")
        except Exception as e:
            st.error(f"Healthcheck error: {e}")

    st.divider()
    st.caption("Tip: If backend is running on a different machine, set BACKEND_URL accordingly.")


# =========================
# RENDER CHAT HISTORY
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# =========================
# CHAT INPUT
# =========================
user_text = st.chat_input("Ask about a course (type a course name or paste a link)…")

if user_text:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Call backend
    reply = ""
    error_msg = None

    payload = {
        "text": user_text,
        "convo_id": st.session_state.convo_id,
    }

    try:
        res = requests.post(BACKEND_URL, json=payload, timeout=60)

        # Handle non-JSON responses cleanly
        content_type = res.headers.get("content-type", "")
        if res.status_code != 200:
            error_msg = f"Backend error ({res.status_code}): {res.text}"
        elif "application/json" in content_type:
            data = res.json()
            reply = (data.get("reply") or "").strip()
        else:
            # If backend returns plain text
            reply = res.text.strip()

    except Exception as e:
        error_msg = f"❌ Error calling backend: {e}"

    if error_msg:
        reply = error_msg

    if not reply:
        reply = "No reply returned from backend."

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Persist messages
    store[st.session_state.convo_id] = st.session_state.messages
    save_store(store)
