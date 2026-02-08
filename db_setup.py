from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

TRACKER_FILE = r"C:\Users\GEL User\Desktop\huggingface\slcchatbot\SLC Full Course Tracker Sheet.xlsx"
DB_FILE = Path(__file__).resolve().parent / "courses.db"


# ---------------- SQLite (Excel -> courses_base + view) ----------------
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS courses_base (
        course_id   TEXT PRIMARY KEY,
        url         TEXT,
        title       TEXT,
        duration    TEXT,
        credits     TEXT,
        base_price  TEXT,
        overview    TEXT
    )""")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS courses_web (
        course_id       TEXT PRIMARY KEY,
        current_price   TEXT,
        original_price  TEXT,
        standard_price  TEXT,
        standard_schedule TEXT,
        fast_price      TEXT,
        fast_schedule   TEXT,
        last_crawled    TEXT,
        status          TEXT,
        error           TEXT
    )""")

    conn.execute("""
    CREATE VIEW IF NOT EXISTS courses_merged AS
    SELECT
        b.course_id,
        b.url,
        b.title,
        b.duration,
        b.credits,
        b.base_price,
        w.current_price,
        w.original_price,
        w.standard_price,
        w.standard_schedule,
        w.fast_price,
        w.fast_schedule,
        w.last_crawled,
        w.status,
        w.error,
        b.overview
    FROM courses_base b
    LEFT JOIN courses_web w ON w.course_id = b.course_id
    """)
    conn.commit()


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def make_course_id(url: str, title: str) -> str:
    if url and isinstance(url, str) and url.strip():
        return url.strip().lower()
    return (title or "").strip().lower()


def load_excel_to_db() -> None:
    df = pd.read_excel(TRACKER_FILE, engine="openpyxl").fillna("")
    df.columns = [c.strip() for c in df.columns]

    title_col = _pick_col(df, ["Course Title", "Course Name", "Title", "Course"])
    url_col = _pick_col(df, ["course_url", "Course URL", "URL", "Link"])

    duration_col = _pick_col(df, ["Duration", "Course Duration"])
    credits_col = _pick_col(df, ["Credits", "Credit Value", "Credit value"])
    price_col = _pick_col(df, ["Price", "Base Price", "Course Price"])
    overview_col = _pick_col(df, ["Overview", "Description", "Course Overview"])

    if not title_col or not url_col:
        raise ValueError(
            f"Excel must contain a title + url column. "
            f"Found columns: {list(df.columns)}"
        )

    conn = sqlite3.connect(DB_FILE)
    init_db(conn)

    conn.execute("DELETE FROM courses_base")
    for _, row in df.iterrows():
        url = str(row.get(url_col, "")).strip()
        title = str(row.get(title_col, "")).strip()
        course_id = make_course_id(url, title)

        conn.execute("""
            INSERT OR REPLACE INTO courses_base(course_id, url, title, duration, credits, base_price, overview)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            course_id,
            url,
            title,
            str(row.get(duration_col, "")).strip() if duration_col else "",
            str(row.get(credits_col, "")).strip() if credits_col else "",
            str(row.get(price_col, "")).strip() if price_col else "",
            str(row.get(overview_col, "")).strip() if overview_col else "",
        ))

    conn.commit()
    conn.close()
    print("✅ Loaded Excel into DB:", DB_FILE)


# ---------------- In-memory search DB (drop-in replacement for FAISS) ----------------
@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]


def _tokenize(s: str) -> set:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    parts = [p for p in s.split() if len(p) > 1]
    return set(parts)


class SimpleSearchDB:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self._doc_tokens = [_tokenize(d.page_content) for d in docs]

    def similarity_search_with_score(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        q = (query or "").strip().lower()
        qtoks = _tokenize(q)

        scored: List[Tuple[int, float]] = []
        for i, doc in enumerate(self.docs):
            mt = doc.metadata or {}
            title = str(mt.get("Course Name") or mt.get("Course Title") or mt.get("title") or "")
            title = title.lower()

            dtoks = self._doc_tokens[i]
            union = len(qtoks | dtoks) or 1
            inter = len(qtoks & dtoks)
            jacc = inter / union  # 0..1

            title_sim = SequenceMatcher(None, q, title).ratio() if title else 0.0
            sim = (0.65 * jacc) + (0.35 * title_sim)  # 0..1

            # Return "distance-like" score to resemble FAISS (lower is better)
            score = 1.0 - sim
            scored.append((i, score))

        scored.sort(key=lambda x: x[1])
        out = [(self.docs[i], score) for i, score in scored[:k]]
        return out

    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k)]


def build_db() -> SimpleSearchDB:
    """
    This is what main.py imports.
    It also refreshes courses.db from Excel so your data stays consistent.
    """
    load_excel_to_db()

    df = pd.read_excel(TRACKER_FILE, engine="openpyxl").fillna("")
    df.columns = [c.strip() for c in df.columns]

    title_col = _pick_col(df, ["Course Title", "Course Name", "Title", "Course"])
    url_col = _pick_col(df, ["course_url", "Course URL", "URL", "Link"])

    docs: List[Document] = []

    for _, row in df.iterrows():
        title = str(row.get(title_col, "")).strip() if title_col else ""
        url = str(row.get(url_col, "")).strip() if url_col else ""

        # Metadata keys your main.py already expects:
        meta = {c: row.get(c, "") for c in df.columns}
        meta["Course Name"] = meta.get("Course Name") or meta.get("Course Title") or title
        meta["Course URL"] = meta.get("Course URL") or meta.get("course_url") or url

        # Build searchable text from row
        parts = [title, url]
        for key in ["Overview", "Description", "Course Overview", "Duration", "Credits",
                    "Qualification Level", "Awarding Body"]:
            if key in meta and str(meta[key]).strip():
                parts.append(f"{key}: {meta[key]}")

        page_content = "\n".join([p for p in parts if p])
        docs.append(Document(page_content=page_content, metadata=meta))

    print(f"✅ Built in-memory search DB with {len(docs)} courses")
    return SimpleSearchDB(docs)


if __name__ == "__main__":
    load_excel_to_db()
