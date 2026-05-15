"""
reasoning/knowledge.py
──────────────────────
Medical knowledge retrieval for Aegis.

Primary source: data/firstaid.sqlite3 populated by scripts/scrape_mayo_firstaid.py
  - 55 Mayo Clinic first aid articles
  - Title embedded at scrape time; retrieved by cosine sim at query time
  - Full article content fetched from DB on demand

Fallback: small in-memory fact set (used only when DB not yet populated).

At query time:
  1. Embed the user query
  2. Cosine-score against all stored title embeddings
  3. Return top-k article content strings to the generator
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from utils.embedding import cosine_similarity, get_embedding

DB_PATH      = Path("data/firstaid.sqlite3")
MIN_SIM      = 0.45
MAX_ARTICLES = 3

# ── SQLite retrieval ───────────────────────────────────────────────────────

def _db_available() -> bool:
    if not DB_PATH.exists():
        return False
    try:
        conn  = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


_article_cache: list[dict] | None = None


def _get_articles() -> list[dict]:
    global _article_cache
    if _article_cache is not None:
        return _article_cache
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT title, url, content, title_embedding FROM articles"
    ).fetchall()
    conn.close()
    articles = []
    for title, url, content, emb_json in rows:
        try:
            emb = np.array(json.loads(emb_json))
        except Exception:
            continue
        articles.append({"title": title, "url": url, "content": content, "embedding": emb})
    _article_cache = articles
    return articles


def _query_db(query: str, top_k: int) -> list[str]:
    articles = _get_articles()
    if not articles:
        return []
    try:
        q_emb = get_embedding(query)
    except Exception:
        return []
    scored = []
    for article in articles:
        try:
            sim = float(cosine_similarity(q_emb, article["embedding"]))
        except Exception:
            continue
        if sim >= MIN_SIM:
            scored.append((sim, article))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        f"[{a['title']}]\n{a['content']}"
        for _, a in scored[:top_k]
    ]


# ── Fallback in-memory facts ───────────────────────────────────────────────

_FALLBACK_FACTS = [
    ("sprain first aid", "Ankle sprains: RICE — Rest, Ice (20 min on/off), Compression, Elevation. No heat, alcohol, or massage for 72 hours. Can't bear weight? X-ray needed to rule out fracture."),
    ("headache first aid", "Dehydration headache: drink 1-2 glasses of water, rest in quiet dark room. Sudden severe 'thunderclap' headache or headache after head injury requires emergency care immediately."),
    ("nausea food poisoning", "Food poisoning: symptoms 2-72 hours after eating contaminated food. Stay hydrated with small sips. Seek care if symptoms last 48+ hours, high fever, blood in stool, or severe dehydration."),
    ("anaphylaxis allergic reaction first aid", "Anaphylaxis: use EpiPen immediately, inject outer thigh, call emergency services. Don't rely on antihistamines alone. Go to emergency room even after EpiPen — symptoms can return."),
    ("migraine first aid", "Migraine: take triptans or NSAIDs early. Rest in dark quiet room. 4+ migraines per month = chronic; see a doctor for preventive medication."),
    ("hypoglycemia low blood sugar diabetes", "Hypoglycaemia: 15g fast-acting carbs (glucose tablets, juice). Recheck in 15 min. Unconscious — don't give food, call emergency services. Metformin alone doesn't cause hypoglycaemia."),
    ("burn first aid", "Minor burns: cool running water 10-20 min, no ice/butter/toothpaste. Non-stick sterile dressing. See doctor for burns larger than 3cm, on face/hands, or deep burns."),
    ("fracture broken bone first aid", "Fracture: immobilise in position found, don't straighten. Ice in cloth. Open fracture (bone through skin): cover with clean cloth, emergency care immediately."),
    ("choking first aid", "Choking adult: 5 back blows between shoulder blades, then 5 abdominal thrusts. Alternate until cleared or unconscious. If unconscious: CPR."),
    ("heart attack first aid", "Heart attack: chest pain/pressure, arm/jaw pain, shortness of breath, nausea. Call emergency services immediately. Aspirin 300mg if conscious and not allergic."),
    ("stroke first aid", "Stroke FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency. Note time symptoms started. Don't give food or water."),
    ("fever first aid", "Fever in adults: paracetamol or ibuprofen, fluids, rest. Seek care if above 39.4°C, lasts 3+ days, or accompanied by stiff neck, rash, or confusion."),
    ("severe bleeding first aid", "Severe bleeding: firm direct pressure with clean cloth for 10+ min. Don't remove soaked cloth — add more on top. Elevate limb. Call emergency services if bleeding won't stop."),
    ("CPR first aid", "CPR adults: 30 chest compressions (hard and fast, 100-120/min), then 2 rescue breaths. Continue until help arrives or person recovers. Hands-only CPR is acceptable if not trained in rescue breathing."),
    ("frostbite hypothermia", "Frostbite: get to warm environment, remove wet clothing. Warm affected area in warm (not hot) water 38-42°C. Don't rub frostbitten tissue. Hypothermia: warm core first, not extremities."),
]

_fallback_cache: list[dict] | None = None


def _get_fallback() -> list[dict]:
    global _fallback_cache
    if _fallback_cache is not None:
        return _fallback_cache
    entries = []
    for query_hint, text in _FALLBACK_FACTS:
        try:
            emb = get_embedding(query_hint)
        except Exception:
            emb = None
        entries.append({"text": text, "embedding": emb})
    _fallback_cache = entries
    return entries


def _query_fallback(query: str, top_k: int) -> list[str]:
    entries = _get_fallback()
    try:
        q_emb = get_embedding(query)
    except Exception:
        return []
    scored = []
    for e in entries:
        if e["embedding"] is None:
            continue
        try:
            sim = float(cosine_similarity(q_emb, e["embedding"]))
        except Exception:
            continue
        if sim >= MIN_SIM:
            scored.append((sim, e["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:top_k]]


# ── Public API ─────────────────────────────────────────────────────────────

def query_knowledge_base(query: str, top_k: int = MAX_ARTICLES) -> list[str]:
    if _db_available():
        results = _query_db(query, top_k)
        if results:
            return results
    return _query_fallback(query, top_k)


def clear_knowledge_base():
    """Deletes the SQLite DB file and resets the in-memory cache."""
    global _article_cache
    _article_cache = None
    if DB_PATH.exists():
        DB_PATH.unlink()
    print(f"Knowledge base DB deleted: {DB_PATH}")


def warmup():
    if _db_available():
        articles = _get_articles()
        print(f"Knowledge base: {len(articles)} Mayo Clinic articles loaded from DB.")
    else:
        _get_fallback()
        print("Knowledge base: DB not found. Using fallback.\n"
              "Run: python scripts/scrape_mayo_firstaid.py")


def db_stats() -> dict:
    if not _db_available():
        return {"source": "fallback", "article_count": len(_FALLBACK_FACTS)}
    return {"source": "mayo_clinic", "article_count": len(_get_articles()), "db_path": str(DB_PATH)}